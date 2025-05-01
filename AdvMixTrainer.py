import dgl
import numpy as np
import torch
import itertools
import datetime
# import wandb
from Util import evaluate
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from sklearn.metrics import recall_score,precision_score


from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
class AdvMixTrainer:
    def __init__(self,
                 model=None,
                 generator=None,
                 discriminator=None,
                 data_loader=None,
                 num_epoch=None,
                 g=None,
                 H=None,
                 gene_semantic_information=None,
                 batch_size=None,
                 learning_rate=None,
                 learning_rate_g=None,
                 test_data_dict=None,
                 num_nodes=None,
                 args=None,
                 adjacency = None
                 ):
        self.data_loader = data_loader
        self.model = model
        self.generator = generator
        self.discriminator = discriminator
        self.gene_semantic_information = gene_semantic_information
        self.semantic_proj = nn.Linear(self.gene_semantic_information.shape[1],
                                       self.gene_semantic_information.shape[1]).cuda()
        self.optimizer = torch.optim.Adam(itertools.chain(model.parameters(), discriminator.parameters(), self.semantic_proj.parameters()),
                                          lr=learning_rate)
        self.optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate_g)
        self.g = g
        self.H = H
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.loss_fn = torch.nn.MSELoss()
        self.test_data_dict = test_data_dict
        self.num_nodes = num_nodes
        self.args = args
        self.adjacency = adjacency


    def run(self):
        for epoch in range(self.num_epoch):
            res = 0
            res_g = 0
            for index, batch in enumerate(self.data_loader):
                # Training Discriminator
                sub_graph = dgl.graph((batch[0], batch[1]), num_nodes=self.num_nodes).to(device)
                self.optimizer.zero_grad()
                h_struct = self.model(self.g, self.g.ndata['feature'], self.H)  # (9758, 32)
                h_semantic = self.semantic_proj(self.gene_semantic_information)
                h_joint = self.model.get_joint_embeddings(h_struct, h_semantic)
                t_score = self.discriminator(sub_graph, h_joint)
                p_score = t_score[:self.batch_size]
                n_score = t_score[self.batch_size:]
                l1_loss = self.regularization_loss_l1(h_joint, batch[0], batch[1])
                d_loss = (self.loss_fn(p_score, torch.ones(self.batch_size).cuda().unsqueeze(1))
                          + self.loss_fn(n_score, torch.zeros(self.batch_size).cuda().unsqueeze(1))
                          + 0.1 * l1_loss
                          )
                h_gen_struct = self.generator(h_struct)
                h_gen_semantic = self.generator(h_semantic)
                s_graph = dgl.graph((batch[0][:self.batch_size], batch[1][:self.batch_size]), num_nodes=self.num_nodes).to(device)
                fake_score = self.get_fake_score(h_struct, h_semantic, h_gen_struct.detach(), h_gen_semantic.detach(), s_graph)
                for f_score in fake_score:
                    d_loss += self.loss_fn(f_score, torch.zeros(self.batch_size).cuda().unsqueeze(1))
                d_loss.backward()
                res += d_loss
                self.optimizer.step()


                # Training Generator
                self.optimizer_g.zero_grad()
                h_struct = self.model(self.g, self.g.ndata['feature'], self.H)  # (9758, 32)
                h_semantic = self.semantic_proj(self.gene_semantic_information)
                fake_joint_struct = self.model.get_joint_embeddings(self.generator(h_struct.detach()), h_semantic.detach())
                fake_joint_semantic = self.model.get_joint_embeddings(h_struct.detach(),
                                                                      self.generator(h_semantic.detach())
                                                                      )
                fake_joint_all = self.model.get_joint_embeddings(self.generator(h_struct.detach()),self.generator(h_semantic.detach()))
                g_score = [self.discriminator(s_graph,fake_joint_struct),
                           self.discriminator(s_graph,fake_joint_semantic),
                           self.discriminator(s_graph,fake_joint_all)
                           ]
                g_loss = 0.0
                for score in g_score:
                    g_loss += self.loss_fn(score, torch.ones(self.batch_size).cuda().unsqueeze(1))
                g_loss.backward()
                res_g += g_loss
                self.optimizer_g.step()

            print(f'epoch: {epoch + 1},d_loss: {res}, g_loss: {res_g}')

            with torch.no_grad():
                self.model.eval()
                h_struct = self.model(self.g, self.g.ndata['feature'], self.H)  # (9758, 32)
                h_semantic = self.semantic_proj(self.gene_semantic_information)
                h_joint = self.model.get_joint_embeddings(h_struct, h_semantic)
                score_list = []
                n_batch = 9758
                batch_size = 50
                size = n_batch // batch_size + (n_batch // batch_size > 0)
                for x in range(size):
                    batch_idx = np.arange(x * batch_size, min(n_batch, (x + 1) * batch_size))
                    u = np.repeat(batch_idx, n_batch)
                    v = np.tile(np.arange(0, n_batch), len(batch_idx))
                    subgraph = dgl.graph((u, v), num_nodes=n_batch)
                    sub_score = self.discriminator.get_test_score_mat(subgraph.to(device), h_joint).reshape(len(batch_idx),
                                                                                                       len(h_joint))
                    score_list.append(sub_score)
                score_mat = torch.cat(score_list)
                score_mat = score_mat.T
                score_mat = score_mat.cpu().numpy()
                sorted_score_mat = np.argsort(score_mat, axis=0, kind='stable')[::-1, :]
                eval_result = evaluate(self.test_data_dict, sorted_score_mat)

                for key, value in eval_result['test'].items():
                    if isinstance(value, (np.float32, np.float64)):  # 检查是否是 np.float 类型
                        eval_result[key] = float(value)  # 转换为 Python 的 float

                print(f'Epoch:{epoch + 1}, eval result: {eval_result}')

                # recall = 0
                # precision = 0
                # pred_result = (score_mat>0.5).astype(int).flatten()
                # true_label = self.adjacency.flatten()
                # recall = recall_score(true_label, pred_result)
                # precision = precision_score(true_label, pred_result)
                # print('*******************************************************')
                # print(f'epoch: {epoch + 1},Recall: {recall}, Precision: {precision}')
                # print('*******************************************************')


    def test(self, test_data_dict, model, discriminator, epoch):
        h_struct = self.model(self.g, self.g.ndata['feature'], self.H)  # (9758, 32)
        h_joint = self.model.get_joint_embeddings(h_struct, self.semantic_proj(self.gene_semantic_information))

        # h_gen_struct = self.generator(h_struct)
        # h_gen_semantic = self.generator(self.semantic_proj(self.gene_semantic_information))
        # h_joint = self.model.get_joint_embeddings(h_gen_struct, h_gen_semantic)

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
                sub_score = discriminator.get_test_score_mat(subgraph.to(device), h_joint).reshape(len(batch_idx),
                                                                                                   len(h_joint))
                score_list.append(sub_score)
            score_mat = torch.cat(score_list)
            score_mat = score_mat.T
            score_mat = score_mat.cpu().numpy()
            sorted_score_mat = np.argsort(score_mat, axis=0, kind='stable')[::-1, :]
            eval_result = evaluate(test_data_dict, sorted_score_mat)
            print(f'Epoch:{epoch + 1}, eval result: {eval_result}')


            # # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            #
            # recall = 0
            # precision = 0
            # pred_result = (score_mat>0.9).astype(int).flatten()
            # true_label = self.adjacency.flatten()
            # recall = recall_score(true_label, pred_result)
            # precision = precision_score(true_label, pred_result)
            # print(f'epoch: {epoch + 1},Recall: {recall}, Precision: {precision}')
            #
            # # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

            # *****************
            # wandb.log({'epoch': epoch, 'P10': eval_result['test']['P10'],'eval_result':eval_result, 'batch_size': self.batch_size, 'learning_rate':self.args.learning_rate, 'learning_rate_g': self.args.learning_rate_g,'hidden_size': self.args.hidden_size, 'dropout':self.args.dropout})
            # *****************



            with open('./result.txt', 'a') as f:
                f.write(f'Epoch:{epoch + 1}, eval result: {eval_result}\n')
        return eval_result

    def regularization_loss_l1(self, h, src, dst):
        src_embeddings = h[src]
        dst_embeddings = h[dst]
        regular = (torch.mean(torch.abs(src_embeddings)) + torch.mean(torch.abs(dst_embeddings))) / 2


        return regular

    def get_fake_score(self, h_struct, h_semantic, h_gen_struct, h_gen_semantic, sub_graph):

        fake_joint_struct = self.model.get_joint_embeddings(h_struct, h_gen_semantic)
        fake_joint_semantic = self.model.get_joint_embeddings(h_gen_struct, h_semantic)
        fake_joint_all = self.model.get_joint_embeddings(h_gen_struct, h_gen_semantic)

        fake_score_struct = self.discriminator(sub_graph, fake_joint_struct)
        fake_score_semantic = self.discriminator(sub_graph, fake_joint_semantic)
        fake_score_all = self.discriminator(sub_graph, fake_joint_all)

        return [fake_score_struct, fake_score_semantic, fake_score_all]



