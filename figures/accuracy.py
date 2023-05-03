import pandas as pd
import os
import numpy as np


'''
df=pd.read_csv('client_results/client_results_ds_CIFAR10_model_Conv4_n_cli_10_ds_split_dirichlet_ds_alpha_0.3_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_False_le_5_s_False.csv')
print(df[df['iter_list']==180]['test_accuracy_list'].values)
print(np.mean(df[df['iter_list']==180]['test_accuracy_list'].values))
print(np.std(df[df['iter_list']==180]['test_accuracy_list'].values))
df=pd.read_csv('client_results/client_results_ds_CIFAR10_model_Conv4_n_cli_10_ds_split_iid_ds_alpha_0.3_align_se_waf_1_delta_None_init_type_kaiming_normal_same_init_False_le_5_s_False.csv')
print(np.mean(df[df['iter_list']==50]['test_accuracy_list'].values))
print(np.std(df[df['iter_list']==50]['test_accuracy_list'].values))
'''

df=pd.read_csv(
    'client_results/diffs/client_results_ds_Fashion_MNIST_model_MLP_n_cli_10_ds_split_dirichlet_ds_alpha_0.3_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_True_le_1_s_False_rand_top_False.csv')
print(df[df['iter_list']==200]['test_accuracy_list'].values)
print(np.mean(df[df['iter_list']==200]['test_accuracy_list'].values))
print(np.std(df[df['iter_list']==200]['test_accuracy_list'].values))

df=pd.read_csv(
    'client_results/diffs/client_results_ds_Fashion_MNIST_model_MLP_n_cli_10_ds_split_iid_ds_alpha_0.3_align_se_waf_1_delta_None_init_type_kaiming_normal_same_init_True_le_1_s_False.csv')
print(np.mean(df[df['iter_list']==50]['test_accuracy_list'].values))
print(np.std(df[df['iter_list']==50]['test_accuracy_list'].values))
