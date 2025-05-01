import dgl
import numpy as np
import torch
import networkx as nx
from torch import nn



def construct_hypergraph(adjacency, device):
    n = adjacency.shape[0]
    adjacency_tensor = torch.from_numpy(adjacency).to(device)
    alpha = 1
    I = torch.eye(n, device=device)
    Z = alpha * torch.inverse(alpha * torch.matmul(adjacency_tensor.t(), adjacency_tensor) + I)
    Z = torch.matmul(torch.matmul(Z, adjacency_tensor.t()), adjacency_tensor).cpu().numpy()
    row, col = np.where(Z > 0.25)
    H = np.zeros((n, n))
    H[row, col] = Z[row, col]

    # threshold = 0.01 # 可以根据需要调整这个阈值
    # H = (Z > threshold).astype(float)

    H = torch.tensor(H, device=device, dtype=torch.float32)

    W_temp = torch.matmul(H.t(), torch.tensor(Z, device=device, dtype=torch.float32))
    W_norm = torch.norm(W_temp, dim=1)
    W_sum = torch.sum(H, dim=0)

    # W_sum[W_sum == 0] = 0.01

    W = torch.diag(W_norm / W_sum)

    # 计算超图邻接矩阵
    DV = torch.diag(torch.sum(torch.matmul(H, W), dim=1))
    hypergraph_adj = torch.matmul(H, W).mm(H.t()) - DV
    hypergraph_adj = hypergraph_adj.cpu().numpy()
    graph = nx.from_numpy_array(hypergraph_adj)


    return graph, H + adjacency_tensor




