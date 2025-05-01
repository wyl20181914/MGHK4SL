import gc
import itertools
import random

import dgl
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from Model import Multi_view_Attention
from Util import load_data, load_pretrain_embeddings, evaluate, get_true_score, \
    compute_score_mat, complete_graph, SLInteractionDataset
from GAN import Generator, Discriminator, # UpdateDiscriminator


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # 如果有多个GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


net_name = './data/SL_adj.npy'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




H, g, train_pos_g, train_neg_g, test_pos_g, test_neg_g, test_data_dict, gene_local, local_gene, subject_g, label, true_interaction_graph, primary_gene, gene_semantic_information = load_data(
    net_name, device)
H = H.float()
train_losses = []
test_losses = []
model = Multi_view_Attention(in_size=32, hidden_size=64, out_size=32, num_heads=[128, 32], dropout=0.1).to(device)
generator = Generator(noise_dim=32, structure_dim=32, g_out_dim=32).to(device)
discriminator = Discriminator(h_feats=32).to(device)

loss_fn = torch.nn.MSELoss(reduction='mean')
# labels = torch.ones(70312).to(device)
# weight_decay=0.0011
optimizer = torch.optim.Adam(itertools.chain(model.parameters(), discriminator.parameters()), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.96)
loss_list = []
best_auc = 0.0
best_f1 = 0.0
best_aupr = 0.0
num_epoch = 50
dataset = SLInteractionDataset(
        './data/Sl_Interactive.txt',
        transform=None
    )
dataloader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=True,
        drop_last=True
    )
for epoch in range(num_epoch):
    model.train()
    gc.collect()
    torch.cuda.empty_cache()
    print(f'****Training---epoch: {epoch+1}****')
    for index, batch in enumerate(dataloader):
        optimizer.zero_grad()
        gene_p = batch[0]
        gene_q = batch[1]
        h_struct = model(g, g.ndata['feature'], H)  # Ours      # (9758,32)
        # h_joint = model.get_joint_embeddings(h_struct, gene_semantic_information)
        pred = discriminator(h_struct, gene_p, gene_q)
        loss = loss_fn(pred, torch.ones(512).unsqueeze(1).to(device))
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        # print(f'epoch: {epoch+1}, loss: {loss.item():.4f}')

    with torch.no_grad():
        model.eval()
        score_list = []
        n_batch = 9758
        batch_size = 50
        size = n_batch // batch_size + (n_batch // batch_size > 0)
        for x in range(size):
            batch_idx = np.arange(x * batch_size, min(n_batch, (x + 1) * batch_size))
            u = np.repeat(batch_idx, n_batch)
            v = np.tile(np.arange(0, n_batch), len(batch_idx))
            subgraph = dgl.graph((u, v), num_nodes=n_batch)
            sub_score = discriminator.get_test_score_mat(subgraph.to(device), h_struct).reshape(len(batch_idx), len(h_struct))
            score_list.append(sub_score)
        score_mat = torch.cat(score_list)
        # score_mat = score_mat.T
        score_mat = score_mat.cpu().numpy()
        sorted_score_mat = np.argsort(score_mat, axis=0, kind='stable')[::-1, :]
        eval_result = evaluate(test_data_dict, sorted_score_mat)
        print(f'Epoch:{epoch + 1}, eval result: {eval_result}')
        with open('./result.txt', 'a') as f:
            f.write(f'Epoch:{epoch + 1}, eval result: {eval_result}\n')




# for i in range(50):
#
#     model.train()
#     gc.collect()
#     torch.cuda.empty_cache()
#     # h = model(g, g.ndata['feature'], H)  # Ours      # (9758,32)
#     # true_score = pred(subject_g.to(device), h)
#     # LOSS = loss_fn(true_score, labels)
#     # LOSS.backward()
#     # optimizer.step()
#     # optimizer.zero_grad()
#     # print(f'Epoch: {e + 1}, TrainLoss: {LOSS.item()}')
#
#     n_batch = primary_gene.shape[0]
#     batch_size = 512
#     batch = n_batch // batch_size + (n_batch // batch_size > 0)
#     epoch_loss = 0
#     start_time = time.time()
#     # accumulation_steps = 4
#
#     for b in range(batch):
#         optimizer.zero_grad()
#         h = model(g, g.ndata['feature'], H)  # Ours      # (9758,32)
#         # feature = torch.cat((h, gene_semantic_information), dim=1)
#         h_joint = model.get_joint_embeddings(h, gene_semantic_information)
#         pred = discriminator(h_joint)
#
#
#         true_score = pred(true_interaction_graph.to(device), h_joint)  # (70312,)
#         # batch_idx = np.arange(b * batch_size, min(n_batch, (b + 1) * batch_size))
#         batch_idx = np.random.choice(np.arange(n_batch), size=batch_size, replace=True)
#         subgraph = complete_graph(batch_idx, primary_gene, len(h))
#         # subgraph = dgl.add_self_loop(complete_subgraph(batch_idx, len(h)))
#         sub_score = pred(subgraph.to(device), h_joint).reshape(len(batch_idx), len(h))
#         batch_loss = model.get_loss(sub_score, true_score[batch_idx]) / batch_size
#         batch_loss.backward()
#         optimizer.step()
#         # scheduler.step()
#         print(f'Epoch:{i + 1}, Mini-batch:{b + 1}, batch_loss {batch_loss.item()}')
#         # with open('./result.txt', 'a') as f:
#         #     f.write(f'Epoch:{i + 1}, Mini-batch:{b + 1}, batch_loss {batch_loss.item()}\n')
#         loss_list.append(batch_loss.item())
#         # if (e + 1) % accumulation_steps == 0:
#         #     optimizer.step()
#         #     optimizer.zero_grad()
#     scheduler.step()
#
#     # batch_idx = np.arange(i * batch_size, min(n_batch, (i + 1) * batch_size))
#     # score_mat = compute_score_mat(h, pred, batch_idx, device)
#     # batch_loss = model.get_loss(score_mat, ave_true_score[batch_idx])
#     # optimizer.zero_grad()
#     # batch_loss.backward(retain_graph=True)
#     # optimizer.step()
#     # epoch_loss = epoch_loss + batch_loss.item()
#
#     with torch.no_grad():
#         model.eval()
#         score_list = []
#         n_batch = 9758
#         batch_size = 50
#         size = n_batch // batch_size + (n_batch // batch_size > 0)
#         for x in range(size):
#             batch_idx = np.arange(x * batch_size, min(n_batch, (x + 1) * batch_size))
#             u = np.repeat(batch_idx, n_batch)
#             v = np.tile(np.arange(0, n_batch), len(batch_idx))
#             subgraph = dgl.graph((u, v), num_nodes=n_batch)
#             sub_score = pred.get_test_score_mat(subgraph.to(device), h_joint).reshape(len(batch_idx), len(h))
#             score_list.append(sub_score)
#         score_mat = torch.cat(score_list)
#         score_mat = score_mat.T
#         score_mat = score_mat.cpu().numpy()
#         sorted_score_mat = np.argsort(score_mat, axis=0, kind='stable')[::-1, :]
#         eval_result = evaluate(test_data_dict, sorted_score_mat)
#         print(f'Epoch:{i + 1}, eval result: {eval_result}')
#         with open('./result.txt', 'a') as f:
#             f.write(f'Epoch:{i + 1}, eval result: {eval_result}\n')




        # for topk in [10, 20, 50, 100]:
        #     if eval_result['test']['P' + str(topk)] > best_results['test']['P' + str(topk)]:
        #         best_results['test']['P' + str(topk)] = eval_result['test']['P' + str(topk)]
        #     if eval_result['test']['R' + str(topk)] > best_results['test']['R' + str(topk)]:
        #         best_results['test']['R' + str(topk)] = eval_result['test']['R' + str(topk)]
        #     if eval_result['test']['N' + str(topk)] > best_results['test']['N' + str(topk)]:
        #         best_results['test']['N' + str(topk)] = eval_result['test']['N' + str(topk)]

# print('Best result:', best_results)


# subgraph = complete_subgraph(batch_idx, len(h))
# pos_score = pred(test_pos_g.to(device), h)
# neg_score = pred(test_neg_g.to(device), h)
# test_loss = compute_loss(pos_score.cpu(), neg_score.cpu()) * 10
# a = LOSS.cpu().numpy()
# b = test_loss.detach().cpu().numpy()
# train_losses.append(a.item())
# test_losses.append(b.item())

# metrics = compute_metrics(pos_score.cpu(), neg_score.cpu())
# print(f'Epoch: {e + 1}, AUC: {metrics[0]}, F1: {metrics[1]}, AUPR: {metrics[2]}')
# print('-------------------')
# if metrics[0] > best_auc:
#     best_auc = metrics[0]
# if metrics[1] > best_f1:
#     best_f1 = metrics[1]
# if metrics[2] > best_aupr:
#     best_aupr = metrics[2]

# if (i + 1) % 1001 == 0:
# # 相似度
# h_transpose = h.transpose(0, 1)
# score_mat = torch.matmul(h, h_transpose).cpu().detach().numpy()

# local_feature = []
# for i in range(n):
#     local_feature.append(h[local_gene[i], :])
# local_feature_tensor = torch.stack(local_feature).to(device)

# score_mat = score_mat.cpu().numpy()
# sorted_score_mat = np.argsort(score_mat, axis=0, kind='stable')[::-1, :]
# eval_result = evaluate(test_data_dict, sorted_score_mat, gene_local, local_gene)
# print(eval_result)
# print('-------------------')

# local_feature = torch.tensor(local_feature).to(device)

# score_mat = pred.get_test_score_mat(head, tail)
# sorted_score_mat = np.argsort(score_mat, axis=0, kind='stable')[::-1, :]
# eval_result = evaluate(test_data_dict, sorted_score_mat)
# print(metrics[0], metrics[1])
# print("Best Result:", best_auc, best_f1, best_aupr)

# 绘制训练损失和测试损失
plt.figure(figsize=(10, 6))
# 绘制训练损失
plt.plot(loss_list, label='Training Loss', color='blue')
# 绘制测试损失
# plt.plot(test_losses, label='Test Loss', color='green')
# 添加标题和标签
plt.title('Training Loss over Mini-Batch ')
plt.xlabel('Mini-Batch')
plt.ylabel('Loss')
# 显示图例
plt.legend()
# 显示网格
plt.grid(True)
# 显示图形
plt.show()
