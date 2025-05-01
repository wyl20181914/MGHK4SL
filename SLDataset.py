import random
import numpy as np
import torch
from dgl.dataloading import Sampler
from torch.utils.data import Dataset


class SLInteractionDataset(Dataset):
    def __init__(self, file_path, transform=None):
        """
        初始化自定义数据集。
        :param file_path: 包含邻接矩阵的文件路径
        :param transform: 可选的转换函数/对象，用于对样本进行预处理
        """
        self.adjacency = np.load(file_path)
        self.adjacency = np.maximum(self.adjacency, self.adjacency.T)  # 确保邻接矩阵是对称的

        # 找到所有的正样本和负样本索引
        self.positive_data = list(zip(*np.where(self.adjacency == 1)))
        self.negative_data = list(zip(*np.where(self.adjacency == 0)))

        # 确保负样本数量足够
        if len(self.negative_data) < len(self.positive_data) :
            raise ValueError("负样本数量不足")

        # 随机选择与正样本数量相同的负样本
        self.positive_samples = [(p[0], p[1], 1) for p in self.positive_data]
        self.negative_samples = [(n[0], n[1], 0) for n in random.sample(self.negative_data, len(self.positive_data) )]
        self.transform = transform

    def __len__(self):
        return len(self.positive_samples) + len(self.negative_samples)

    def __getitem__(self, idx):
        if idx < len(self.positive_samples):
            gene_p, gene_q, label = self.positive_samples[idx]
        else:
            gene_p, gene_q, label = self.negative_samples[idx - len(self.positive_samples)]

        if self.transform:
            gene_p, gene_q = self.transform((gene_p, gene_q))

        sample = (torch.tensor(gene_p, dtype=torch.long), torch.tensor(gene_q, dtype=torch.long),
                  torch.tensor(label, dtype=torch.float))

        return sample


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        """
        初始化平衡批采样器。

        :param dataset: 自定义数据集实例
        :param batch_size: 每个 batch 的大小
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_pos_samples = len(dataset.positive_samples)
        self.num_neg_samples = len(dataset.negative_samples)
        # 计算总批次数量，包括可能存在的不完整批次
        self.num_batches = (self.num_pos_samples + self.batch_size - 1) // self.batch_size

        if self.num_batches == 0:
            raise ValueError("正样本数量必须至少为 batch_size")

    def __iter__(self):
        pos_indices = list(range(self.num_pos_samples))
        neg_indices = list(range(self.num_neg_samples))

        for _ in range(self.num_batches):
            batch_indices = []

            # 随机选择正样本
            selected_pos_indices = random.choices(pos_indices, k=self.batch_size)
            batch_indices.extend(selected_pos_indices)

            # 随机选择负样本
            selected_neg_indices = random.choices(neg_indices, k=self.batch_size)
            batch_indices.extend([idx + self.num_pos_samples for idx in selected_neg_indices])

            yield batch_indices

    def __len__(self):
        return self.num_batches

    # def __iter__(self):
    #     # 打乱正样本和负样本的索引
    #     pos_indices = torch.randperm(self.num_pos_samples).tolist()
    #     neg_indices = torch.randperm(self.num_neg_samples).tolist()
    #
    #     # 生成 batch
    #     for i in range(self.num_batches):
    #         batch_indices = []
    #         start_idx = i * self.batch_size
    #         end_idx = min(start_idx + self.batch_size, self.num_pos_samples)
    #
    #         # 添加正样本
    #         batch_indices.extend(pos_indices[start_idx:end_idx])
    #
    #         # 添加负样本
    #         batch_indices.extend([idx + self.num_pos_samples for idx in neg_indices[start_idx:end_idx]])
    #
    #         yield batch_indices
