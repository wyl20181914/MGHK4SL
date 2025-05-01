import copy
import time

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy import sparse
from Hypergraph import construct_hypergraph
from torch.utils.data import Dataset


# from sklearn.metrics import average_precision_score


def get_pos_neg_graph(g):
    u, v = g.edges()
    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)  # shuffle
    test_size = int(len(eids) * 0.3)
    # train_size = g.number_of_edges() - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]

    combined_list = list(test_pos_u.numpy()) + list(test_pos_v.numpy())
    unique_list = list(set(combined_list))
    gene_list = sorted(unique_list)
    gene_local = {}
    local_gene = {}
    for i in range(len(gene_list)):
        gene_local[gene_list[i]] = i
        local_gene[i] = gene_list[i]

    # 构造测试数据字典
    test_data_dict = {}
    for i in range(len(test_pos_u)):
        id1 = test_pos_u[i].item()
        id2 = test_pos_v[i].item()
        if id1 not in test_data_dict.keys():
            test_data_dict[id1] = {}
        if id2 not in test_data_dict.keys():
            test_data_dict[id2] = {}
        test_data_dict[id1][id2] = 1
        test_data_dict[id2][id1] = 1

    # 构造测试评分矩阵图
    n = len(gene_list)
    H = torch.ones(n, n)
    test_mat_u, test_mat_v = np.where(H == 1)
    test_mat_g = dgl.graph((test_mat_u, test_mat_v), num_nodes=n)

    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]
    # Find all negative edges and split them for training and testing
    adj = sparse.coo_matrix((np.ones(len(u)), (u.cpu().numpy(), v.cpu().numpy())))
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), g.number_of_edges())  # default replace=True
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())
    return dgl.add_self_loop(train_pos_g), dgl.add_self_loop(train_neg_g), \
        dgl.add_self_loop(test_pos_g), dgl.add_self_loop(test_neg_g), eids[
                                                                      :test_size], test_data_dict, gene_local, local_gene, test_mat_g


def load_data(net_name, device):
    adjacency = np.load(net_name)
    adjacency = np.maximum(adjacency, adjacency.T)
    primary_gene, partner_gene = np.where(adjacency == 1)
    true_interaction_graph = dgl.graph((primary_gene, partner_gene), num_nodes=9758)
    # count = np.count_nonzero(adjacency)
    # type = np.allclose(adjacency, adjacency.T)

    label = []
    for i in range(adjacency.shape[0]):
        label.append(np.count_nonzero(adjacency[i]))
    label = np.array(label)
    edges = sum(label)

    graph = nx.from_numpy_array(adjacency)
    n = adjacency.shape[0]
    subject_g = dgl.from_networkx(graph)
    train_pos_g, train_neg_g, test_pos_g, test_neg_g, test_link, test_data_dict, gene_local, local_gene, test_mat_g = get_pos_neg_graph(
        subject_g)
    train_subject_g = dgl.remove_edges(subject_g, test_link)
    train_network = dgl.add_self_loop(train_subject_g).to_networkx()
    # train_network = train_subject_g.to_networkx()
    graph, H = construct_hypergraph(nx.to_numpy_array(train_network), device)  # adjacency_matrix
    # src, dst = hypergraph_adj.nonzero(as_tuple=True)
    # edges = (src, dst)
    # train_g = dgl.graph(edges, num_nodes=hypergraph_adj.shape[0])

    train_g = dgl.from_networkx(graph)
    # train_g = dgl.from_networkx(H_graph)
    train_g = dgl.add_self_loop(train_g)
    train_g = train_g.to(device)
    gene_semantic_information = load_pretrain_embeddings('./data/all_entities_pretrain_emb.npy').to(device)
    train_g.ndata['feature'] = nn.Embedding(n, 32).weight.to(device)
    nn.init.xavier_uniform_(train_g.ndata['feature'])
    # train_g.ndata['feature'] = torch.zeros(n, 64, device=device)
    # train_g.ndata['feature'] = gene_semantic_information
    # print('dataset loaded')
    return H, train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g, test_data_dict, gene_local, local_gene, subject_g, label, true_interaction_graph, primary_gene, gene_semantic_information, adjacency


def load_pretrain_embeddings(embeddings_path):
    all_entities = pd.read_csv('./data/entities.txt', delimiter=' ')
    bert_path = './data/all_entities_pretrain_emb.npy'
    # print(bert_path)
    all_entities_pretrain_emb_org = np.load(bert_path)
    # PCA dimension reduction
    from sklearn.decomposition import PCA
    pca = PCA(n_components=32)
    pca.fit(all_entities_pretrain_emb_org)
    all_entities_pretrain_emb = pca.transform(all_entities_pretrain_emb_org)
    # Gene节点特征
    Gene = all_entities.loc[all_entities['genres'] == 'Gene']
    Gene_id = torch.tensor(Gene['id'].values)
    Mapping_Gene_id = pd.DataFrame(data={
        'name': Gene['name'],
        # 'Gene_id': Gene_id,
        'mappedID': pd.RangeIndex(len(Gene_id)),
    })
    select_ids = []
    for id in Gene_id:
        select_ids.append(id)
    Gene_feat = torch.tensor(all_entities_pretrain_emb[select_ids])
    return Gene_feat


def evaluate(test_data_dict, sorted_mat):
    metrics = {
        'P10': [], 'P20': [], 'P50': [], 'P100': [],
        'R10': [], 'R20': [], 'R50': [], 'R100': [],
        'N10': [], 'N20': [], 'N50': [], 'N100': []
    }
    results = {'test': copy.deepcopy(metrics)}

    for test_gene in test_data_dict.keys():
        sorted_list = []
        partner = list(sorted_mat[:, test_gene])
        for item in partner:
            sorted_list.append(item)
            if len(sorted_list) == 100: break

        for topk in [10, 20, 50, 100]:
            hit_topk = len(set(sorted_list[:topk]) & set(test_data_dict[test_gene].keys()))

            # ndcg topk
            denom = np.log2(np.arange(2, topk + 2))
            dcg_topk = np.sum(np.in1d(sorted_list[:topk], list(test_data_dict[test_gene].keys())) / denom)
            idcg_topk = np.sum((1 / denom)[:min(len(list(test_data_dict[test_gene].keys())), topk)])

            results['test'][f'P{topk}'].append(
                0 if hit_topk == 0 or len(test_data_dict[test_gene].keys()) == 0 else hit_topk / min(topk, len(
                    test_data_dict[test_gene].keys())))
            results['test'][f'R{topk}'].append(
                0 if hit_topk == 0 or len(test_data_dict[test_gene].keys()) == 0 else hit_topk / len(
                    test_data_dict[test_gene].keys()))
            results['test'][f'N{topk}'].append(0 if dcg_topk == 0 or idcg_topk == 0 else dcg_topk / idcg_topk)

    for topk in [10, 20, 50, 100]:
        results['test']['P' + str(topk)] = float(round(np.asarray(results['test']['P' + str(topk)]).mean(), 4))
        results['test']['R' + str(topk)] = float(round(np.asarray(results['test']['R' + str(topk)]).mean(), 4))
        results['test']['N' + str(topk)] = float(round(np.asarray(results['test']['N' + str(topk)]).mean(), 4))
    return results


def get_true_score(label, score):
    ave_true_score = []
    idx = 0
    for i in range(len(label)):
        num = label[i]
        if num == 0:
            ave_true_score.append(0)
        else:
            ave_true_score.append(torch.mean(score[idx:idx + num].float()).item())
            idx = idx + num

    ave_true_score = torch.tensor(np.array(ave_true_score))
    return ave_true_score


def compute_score_mat(h, pred, batch_index, device):
    start_time = time.time()
    # n = len(local_gene)
    n = h.shape[0]
    score_mat = torch.empty(len(batch_index), n).to(device)
    row = 0
    for j in batch_index:
        head = []
        tail = []
        for i in range(n):
            head.append(h[j, :])
            tail.append(h[i, :])
            # head.append(h[local_gene[j], :])
            # tail.append(h[local_gene[i], :])
        head = torch.stack(head).to(device)
        tail = torch.stack(tail).to(device)
        score = pred.get_test_score_mat(head, tail)
        score_mat[row] = score
        row = row + 1
    # score_mat = score_mat.t()
    end_time = time.time()
    print(f'本轮耗时 {end_time - start_time}秒')
    return score_mat


def complete_graph(batch_idx, primary_gene, k):
    # adjacency = np.load('./data/SL_adj.npy')
    u = np.repeat(primary_gene[batch_idx], k)
    v = np.tile(np.arange(0, k), len(batch_idx))
    subgraph = dgl.graph((u, v), num_nodes=k)

    return subgraph


class SLInteractionDataset(Dataset):
    def __init__(self, file_path, transform=None):
        adjacency = np.load(file_path)
        adjacency = np.maximum(adjacency, adjacency.T)
        self.positive_data = np.where(adjacency == 1)
        self.negative_data = np.where(adjacency == 0)
        # self.data = pd.read_csv(file_path, sep='\t', encoding='utf-8')
        self.transform = transform

    def __len__(self):
        return len(self.positive_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        gene_p = self.positive_data.iloc[idx, 0]
        gene_q = self.positive_data.iloc[idx, 1]

        if self.transform:
            gene_p, gene_q = self.transform((gene_p, gene_q))

        sample = (torch.tensor(gene_p, dtype=torch.long), torch.tensor(gene_q, dtype=torch.long))

        return sample

