from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
import collections

import random

import pickle

def record_net_data_stats_iid( net_dataidx_map, args):

    save_str = f"dataset_split_info_ds_{args.dataset}_model_{args.model}_n_cli_{args.num_clients}_ds_split_{args.dataset_split}_ds_alpha_{args.dirichlet_alpha}_align_{args.align_loss}_waf_{args.weight_align_factor}_delta_{args.delta}_init_type_{args.weight_init}_same_init_{args.same_initialization}_le_{args.local_epochs}_s_{args.single_model}"

    print(f'Data statistics: {net_dataidx_map}')
    print(net_dataidx_map)

    with open(f'{args.base_dir}weight_alignment_csvs/{save_str}.pkl', 'wb') as f:
        pickle.dump(net_dataidx_map, f)

def record_net_data_stats(y_train, net_dataidx_map, args):

    save_str = f"dataset_split_info_model_{args.model}_n_cli_{args.num_clients}_ds_split_{args.dataset_split}_ds_alpha_{args.dirichlet_alpha}_align_{args.align_loss}_waf_{args.weight_align_factor}_delta_{args.delta}_init_type_{args.weight_init}_same_init_{args.same_initialization}_le_{args.local_epochs}_s_{args.single_model}"

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    print(f'Data statistics: {net_cls_counts}')
    print(net_cls_counts)

    with open(f'{args.base_dir}weight_alignment_csvs/{save_str}.pkl', 'wb') as f:
        pickle.dump(net_cls_counts, f)

    return net_cls_counts


def dirichlet(y_train,  n_nets, alpha=0.5):

    if alpha is None:
        print("Set dirichlet alpha value")
        sys.exit()

    min_size = 0
    K = 10
    N = y_train.shape[0]
    net_dataidx_map = {}

    while min_size < 10:
        idx_batch = [[] for _ in range(n_nets)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets))

            proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]



    return net_dataidx_map,y_train