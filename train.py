import argparse
import random
import datetime
from argparse import Namespace
import numpy as np
import torch
from torch.utils.data import DataLoader
from Model import Multi_view_Attention
from Util import load_data, load_pretrain_embeddings, evaluate, get_true_score, \
    compute_score_mat, complete_graph
from GAN import Generator, Discriminator
from AdvMixTrainer import AdvMixTrainer
from data.SLDataset import SLInteractionDataset, BalancedBatchSampler
# import wandb

def print_total_param(model):
    total_params = 0
    for name, parameters in model.named_parameters():
        params = np.prod(list(parameters.size()))
        total_params += params
    print('total parameters: {:.4f}M'.format(total_params / 1e6))

def get_args():
    arg = argparse.ArgumentParser(description="Parameter for OurModel4SL")
    arg.add_argument('-dataset_name', type=str, default='./data/SL_adj.npy')
    # arg.add_argument('--dataset_name', type=str, default='/tmp/pycharm_project_164/data/SL_adj.npy')
    arg.add_argument('--batch_size', type=int, default=1024)
    arg.add_argument('--num_epoch', type=int, default=50)
    arg.add_argument('--noise_dim', type=int, default=32)
    arg.add_argument('--structure_dim', type=int, default=32)
    arg.add_argument('--dropout', type=float, default=0.1)
    arg.add_argument('--in_size', type=int, default=32)
    arg.add_argument('--hidden_size', type=int, default=64)
    arg.add_argument('--out_size', type=int, default=32)
    arg.add_argument('--learning_rate', type=float, default=0.00021)
    arg.add_argument('--learning_rate_g', type=float, default=0.0021)
    arg.add_argument('-net_name', type=str, default='./data/SL_adj.npy')
    # arg.add_argument('--net_name', type=str, default='/tmp/pycharm_project_164/data/SL_adj.npy')
    arg.add_argument('--g_out_dim', type=int, default=32)
    arg.add_argument('--seed', type=int, default=42)
    arg.add_argument('--num_nodes', type=int, default=9758)
    return arg.parse_args()


if __name__ == "__main__":

# def train():
    args = get_args()
    print(args)

    # nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # wandb.init(name=nowtime)
    # print('******************\n' + str(wandb.config) + '\n******************')
    # args = wandb.config

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_heads = [128, 32]

    dataset = SLInteractionDataset(
        args.dataset_name,
        transform=None
    )
    sampler = BalancedBatchSampler(
        dataset,
        args.batch_size
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
    )
    H, g, train_pos_g, train_neg_g, test_pos_g, test_neg_g, test_data_dict, gene_local, local_gene, subject_g, label, true_interaction_graph, primary_gene, gene_semantic_information, adjacency  = load_data(
        args.net_name, device)
    H = H.float()
    # define the model
    model = Multi_view_Attention(
        in_size=args.in_size,
        hidden_size=args.hidden_size,
        out_size=args.out_size,
        num_heads=num_heads,
        dropout=args.dropout
    ).to(device)


    print_total_param(model)
    print(model)

    # define the generator
    generator = Generator(
        noise_dim=args.noise_dim,
        structure_dim=args.structure_dim,
        g_out_dim=args.g_out_dim
    ).to(device)
    # define the discriminator
    discriminator = Discriminator(h_feats=32).to(device)
    trainer = AdvMixTrainer(
        model=model,
        generator=generator,
        discriminator=discriminator,
        num_epoch=args.num_epoch,
        data_loader=dataloader,
        g=g,
        H=H,
        gene_semantic_information=gene_semantic_information,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        learning_rate_g=args.learning_rate_g,
        test_data_dict=test_data_dict,
        num_nodes=args.num_nodes,
        args=args,
        adjacency= adjacency
    )
    trainer.run()

