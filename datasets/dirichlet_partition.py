from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
import collections

import random

def record_net_data_stats(y_train, net_dataidx_map):

	net_cls_counts = {}

	for net_i, dataidx in net_dataidx_map.items():
		unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
		tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
		net_cls_counts[net_i] = tmp

	print(f'Data statistics: {net_cls_counts}')

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

    record_net_data_stats(y_train, net_dataidx_map)

    return net_dataidx_map