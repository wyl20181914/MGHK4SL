import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, noise_dim, structure_dim, g_out_dim):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.model = nn.Sequential(
            nn.Linear(noise_dim + structure_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, g_out_dim),
            nn.Sigmoid()
        )

    def forward(self, batch_ent_emb):
        random_noise = torch.randn((batch_ent_emb.shape[0], self.noise_dim)).cuda()
        batch_data = torch.cat((random_noise, batch_ent_emb), dim=-1)
        out = self.model(batch_data)
        return out




class Discriminator(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)
        # self.model = nn.Sequential(
        #     nn.Linear(h_feats * 2, h_feats),
        #     nn.LeakyReLU(),
        #     nn.Linear(h_feats, 1),
        #     nn.Sigmoid()
        # )

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': F.sigmoid(self.W2(F.relu(self.W1(h))))}
        # return {'score': self.W2(F.relu(self.W1(h)))}

    @torch.no_grad()
    def get_test_score_mat(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']
