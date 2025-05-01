# MGHK4SL

This repository contains the official implementation for the paper: Multimodal GAN Integrating Hypergraph and Knowledge Graph Representations for Synthetic Lethality.

## Requirements

The recommended requirements for MGHK4SL are specified as follows:
* python==3.10.15
* pytroch==2.4.1
* dgl==2.0.0
* pandas==2.2.3
* scikit-learn==1.5.2
* scipy==1.13.1
* numpy==1.26.4
* numpy-base==1.26.4
* torchvision==0.19.1
* wandb==0.19.1
* tqdm==4.66.5

The dependencies can be installed by:
```bash
pip install -r requirements.txt
```

## Data

The human SL gene pairs are sourced from the public dataset SynLethDB 2.0: https://synlethdb.sist.shanghaitech.edu.cn/v2

## Usage

To train and evaluate MGHK4SL , run the following command:

```train & evaluate
python train.py <dataset_name> <net_name> --batch-size <batch_size> --noise-dims <noise_dims> ...
```

## Acknowledgement
The code is inspired by [AdaMF-MAT](https://github.com/zjukg/AdaMF-MAT) and [NSF4SL](https://github.com/JieZheng-ShanghaiTech/NSF4SL).
>[Unleashing the Power of Imbalanced Modality Information for Multi-modal Knowledge Graph Completion](https://arxiv.org/abs/2402.15444)
>[NSF4SL: negative-sample-free contrastive learning for ranking synthetic lethal partner genes in human cancers](https://doi.org/10.1093/bioinformatics/btac462)

SL data and SynLethKG are constructed based on [SynLethDB 2.0](https://synlethdb.sist.shanghaitech.edu.cn/v2/#/).
>[SynLethDB 2.0: A web-based knowledge graph database on synthetic lethality for novel anticancer drug discovery](https://doi.org/10.1093/database/baac030)