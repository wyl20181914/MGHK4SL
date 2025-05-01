import os

import networkx as nx
import numpy as np
import torch



def construct_knowledge_hypergraph(device):
    with open(os.path.join('./KgData/Gene_set.txt')) as f:
        gene_set = []
        for line in f:
            gene_set.append(line.strip())
    with open(os.path.join('./KgData/relations.txt')) as f:
        relation2id = dict()
        id2relation = []
        for line in f:
            relation, rid = line.strip().split()
            relation2id[relation] = int(rid)
            id2relation.append(relation)
    with open(os.path.join('./KgData/all_entities.txt')) as f:
        entities_2id = dict()
        id2entities = []
        for line in f:
            entities, eid = line.strip().split()
            entities_2id[entities] = int(eid)
            id2relation.append(entities)
    kg_triples = []
    with open(os.path.join('./KgData/kg.txt')) as f:
        for line in f:
            h, r, t = line.strip().split()
            h, r, t = entities_2id[h], relation2id[r], entities_2id[t]
            kg_triples.append([h, r, t])
    gene_id = []
    for item in entities_2id.keys():
        if item in gene_set:
            gene_id.append(entities_2id[item])
    # knowledge_hypergraph_incidence_matrix = np.zeros((len(entities_2id), len(relation2id)))  # 知识超图关联矩阵
    knowledge_hypergraph_incidence_matrix = torch.zeros((len(entities_2id), len(relation2id))).to(device)
    for triple in kg_triples:
        knowledge_hypergraph_incidence_matrix[triple[0]][triple[1]] = 1
        knowledge_hypergraph_incidence_matrix[triple[2]][triple[1]] = 1
    H = torch.tensor(knowledge_hypergraph_incidence_matrix, dtype=torch.float32).to(device)
    # H = knowledge_hypergraph_incidence_matrix.clone().detach().requires_grad_(True).to(device)

    Dv = torch.diag(torch.sum(knowledge_hypergraph_incidence_matrix, dim=1)).to(device)
    W = torch.diag(torch.ones(len(knowledge_hypergraph_incidence_matrix[1]), dtype=torch.float32)).to(device)
    knowledge_hypergraph_adjacency_matrix = torch.matmul(H, W).mm(H.t()) - Dv
    # knowledge_hypergraph_adjacency_matrix = torch.matmul(H, W).mm(H.t()) - Dv
    # knowledge_hypergraph_adjacency_matrix = H @ W @ H.transpose() - Dv # 知识超图邻接矩阵
    knowledge_hypergraph_adjacency_matrix = knowledge_hypergraph_adjacency_matrix.cpu().numpy()
    g = nx.from_numpy_array(knowledge_hypergraph_adjacency_matrix)
    return knowledge_hypergraph_adjacency_matrix, knowledge_hypergraph_incidence_matrix
