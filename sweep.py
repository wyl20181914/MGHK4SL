from argparse import Namespace
from train import train



import wandb
wandb.login(key='ec1d27b001b8cefe640d3f6cd09a1b5af22f8c4d')

# args = Namespace(
#         dataset_name = '/tmp/pycharm_project_930/data/SL_adj.npy',
#         batch_size = 512,
#         num_epoch = 10,
#         noise_dim = 32,
#         structure_dim = 32,
#         dropout = 0.1,
#         in_size = 32,
#         hidden_size = 64,
#         out_size =32,
#         learning_rate = 0.0001,
#         learning_rate_g =0.0001,
#         net_name = '/tmp/pycharm_project_930/data/SL_adj.npy',
#         g_out_dim = 32,
#         seed = 42,
#         num_nodes =9758
#     )

sweep_config = {
    'method': 'random',
    # 'method': 'bayes'
}
metric = {
    'name': 'P10',
    'goal': 'maximize'
}
sweep_config['metric'] = {
    'name': 'eval_result[P][10]',
    'goal': 'maximize'
}

sweep_config['parameters'] = {}
# 固定不变的超参
sweep_config['parameters'].update({
    'dataset_name': {'value': '/tmp/pycharm_project_256/data/SL_adj.npy'},
    'num_epoch': {'value': 5},
    'noise_dim':{'value': 32},
    'structure_dim':{'value': 32},
    'in_size':{'value': 32},
    'out_size':{'value': 32},
    'net_name':{'value': '/tmp/pycharm_project_256/data/SL_adj.npy'},
    'g_out_dim': {'value': 32},
    'seed': {'value': 42},
    'num_nodes': {'value': 9758},
    'dropout': {'value': 0.11},
    # 'learning_rate': {'value': 0.01},
    # 'learning_rate_g': {'value': 0.001}

})

# 离散型分布超参
sweep_config['parameters'].update({
    'hidden_size': {
        'values': [16, 32, 48, 64, 80, 96, 112, 128]
    }
})

# 连续型分布超参
sweep_config['parameters'].update({

    'learning_rate': {
        'distribution': 'log_uniform_values',
        'min': 1e-6,
        'max': 0.1
    },
    'learning_rate_g': {
        'distribution': 'log_uniform_values',
        'min': 1e-6,
        'max': 0.1
    },

    'batch_size': {
        'distribution': 'q_uniform',
        'q': 128,
        'min': 256,
        'max': 2048,
    },

    # 'dropout': {
    #     'distribution': 'uniform',
    #     'min': 0,
    #     'max': 0.6,
    # }
})
from pprint import pprint

pprint(sweep_config)

sweep_id = wandb.sweep(sweep_config, project='OurModel4SL')

wandb.agent(sweep_id, function = train, count=50)
