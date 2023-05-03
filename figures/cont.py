"""
IMAGES: PEER contribution equation vs accuracy at Epoch K
Convergence Rate versus baseline
"""

import matplotlib.pyplot as plt
import math
import pickle
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
#Example
#x=[0.10,0.11,0.10 ,0.09, 0.09,0.11  ,  0.10 ,   0.10 ,   0.12 ,   0.08]
#np.uniformity=1-(np.linalg.norm(x)*math.sqrt(10)-1)/(math.sqrt(10)-1)

plt.clf()

ds = 'cifar10'

seeds=[1,4,6]
for seed in seeds:
    if ds=='cifar10':
        class_dict=f'dataset_splits/dataset_split_info_model_Conv4_seed_{seed}_n_cli_20_ds_split_dirichlet_ds_alpha_0.25_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_True_le_5_s_False.pkl'
        classes = pickle.load( open( class_dict, "rb" ) )
        results=f'client_results/peer_contrib/client_results_ds_CIFAR10_model_Conv4_ds_CIFAR10_seed_{seed}_n_cli_20_ds_split_dirichlet_ds_alpha_0.25_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_True_le_5_s_False_rand_top_True.csv'

    else:
        results=f'client_results/peer_contrib/client_results_ds_Fashion_MNIST_model_MLP_ds_Fashion_MNIST_seed_{seed}_n_cli_20_ds_split_dirichlet_ds_alpha_0.25_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_True_le_1_s_False_rand_top_True.csv'
        class_dict=f'dataset_splits/dataset_split_info_model_MLP_seed_{seed}_n_cli_20_ds_split_dirichlet_ds_alpha_0.25_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_True_le_1_s_False.pkl'
        classes = pickle.load( open( class_dict, "rb" ) )

    df=pd.read_csv(results)


    df=df[(df['iter_list']>120)&(df['iter_list']<200) ]
    df=df.groupby(['client_list'])['test_losses'].mean()#.agg({'test_accuracy_list':['mean','std']})



    total_samples=0
    for peer in classes.items():
        peer, class_ls=peer[0],peer[1]
        total_samples+=sum(class_ls.values())


    freq_dict={}
    label_len=10

    def pad_or_truncate(some_list, target_len):
        return some_list[:target_len] + [0]*(target_len - len(some_list))

    uniformity_ls=[]
    size_ls=[]
    for peer in classes.items():
        peer, class_ls=peer[0],peer[1]
        vals=list(class_ls.values())

        vals=pad_or_truncate(vals,10)
        size_ls.append(sum(vals))
        uniformity_ls.append(np.std(vals))

    #print(uniformity_ls)


    #print(np.mean(df.values))
    #print(np.std(df.values))

    #plt.plot([i / j for i, j in zip(size_ls, uniformity_ls)], df.values, '.')
    plt.plot(uniformity_ls, df.values, '.')
    #plt.plot(uniformity_ls,size_ls, '.')

    corr, _ = pearsonr(uniformity_ls, size_ls)
    print('\nSTD to Size correlation %.3f' % corr)

    corr, _ = pearsonr(size_ls, df.values)
    print('Size to Accuracy correlation %.3f' % corr)

    corr, _ = pearsonr(uniformity_ls, df.values)
    print('STD to Accuracy correlation: %.3f' % corr)


    #plt.plot([i for i in range(len(df.values))], df.values, '.')
plt.savefig('a2.png')