import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv


class HyperedgeAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(HyperedgeAttention, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size, bias=True),
            # -----------
            nn.BatchNorm1d(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=True),
            # ------------
        )

    def forward(self, X, H):
        N = X.shape[0]
        X = X.reshape(N, -1)
        X = torch.mm(H.T, X)
        w = self.layers(X)
        beta = torch.softmax(w, dim=0)
        return beta * X


class AttentionLayer(nn.Module):
    def __init__(self, in_size, out_size, layer_num_heads, dropout):
        super(AttentionLayer, self).__init__()
        self.gat_layer = GATConv(in_size, out_size, layer_num_heads, dropout, dropout, activation=F.relu)
        self.hyperedge_attention = HyperedgeAttention(in_size=out_size * layer_num_heads)
        # self.linear = nn.Linear(in_size, out_size * layer_num_heads)

    def forward(self, g, h, H):  # G
        hyperedge_embeddings = self.gat_layer(g, h)  # self.hgnn(h, G)
        hyperedge_embeddings = torch.relu(hyperedge_embeddings)

        return self.hyperedge_attention(hyperedge_embeddings, H)


class Multi_view_Attention(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_heads, dropout):
        super(Multi_view_Attention, self).__init__()
        self.dropout = dropout
        self.layer = nn.ModuleList()
        self.layer.append(AttentionLayer(in_size, hidden_size, num_heads[0], dropout))
        self.layer.append(AttentionLayer(hidden_size * num_heads[0], hidden_size, num_heads[1], dropout))
        # self.layer.append(AttentionLayer(hidden_size * num_heads[1], hidden_size, num_heads[2], dropout))
        self.fc = nn.Linear(hidden_size * num_heads[-1], out_size)
        # ------------------------
        self.semantic_emb = nn.Linear(32, 32)
        self.ent_attn = nn.Linear(out_size, 1, bias=False)
        self.ent_attn.requires_grad_(True)

    def forward(self, g, h, H):  # G
        for layer in self.layer:
            h = layer(g, h, H)  # G
            F.dropout(h, self.dropout)
        return self.fc(h)

    def get_joint_embeddings(self, struct_emb, emb):
        # semantic_emb = self.semantic_emb(emb)
        e = torch.stack((struct_emb, emb), dim=1)
        u = torch.tanh(e)
        scores = self.ent_attn(u).squeeze(-1)
        attention_weights = torch.softmax(scores, dim=-1)
        joint_embedding = torch.sum(attention_weights.unsqueeze(-1) * e, dim=1)
        return joint_embedding

    def get_loss(self, scores, ave_true_score):
        max_n = torch.max(scores, 1, keepdim=True)[0]
        loss = torch.sum(-ave_true_score + max_n.squeeze() + torch.logsumexp(scores - max_n, 1))
        return loss
