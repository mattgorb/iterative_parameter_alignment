import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pickle
import pandas as pd
import numpy as np
from scipy.stats import pearsonr


rule='classimbalance'
#rule='powerlaw'
#rule='uniform'

metric='test_accuracy_list'
#metric='test_losses'

ipa=pd.read_csv(f'client_results/peer_contrib/benchmarks/client_results_ds_CIFAR10_model_Conv4_ds_CIFAR10_seed_1_n_cli_10_None_ds_split_{rule}_ds_alpha_0.3_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_False_le_5_s_False_rand_top_True.csv')
baseline=pd.read_csv(f'client_results/peer_contrib/benchmarks/client_results_ds_CIFAR10_model_Conv4_ds_CIFAR10_seed_1_n_cli_10_None_ds_split_{rule}_ds_alpha_0.3_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_False_le_50_s_True_rand_top_True.csv')
#ipa=pd.read_csv(f'client_results/peer_contrib/benchmarks/client_results_ds_MNIST_model_MLP_ds_MNIST_seed_1_n_cli_20_None_ds_split_{rule}_ds_alpha_0.3_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_False_le_5_s_False_rand_top_False.csv')
#baseline=pd.read_csv(f'client_results/peer_contrib/benchmarks/client_results_ds_MNIST_model_MLP_ds_MNIST_seed_1_n_cli_20_None_ds_split_{rule}_ds_alpha_0.3_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_False_le_25_s_True_rand_top_False.csv')


baseline=baseline.groupby(['client_list'])[metric].mean()


for epoch in np.unique(ipa['iter_list'].values):
    df=ipa[ipa['iter_list']==epoch]
    df=df.groupby(['client_list'])[metric].mean()

    corr, _ = pearsonr(df, baseline)
    print(f'Epoch {epoch} corr {corr}')
    #print(corr)

total = ipa[(ipa['iter_list'] > 4) & (ipa['iter_list'] < 11)]
total=total.groupby(['client_list'])[metric].mean()
corr, _ = pearsonr(total, baseline)

print(f'Total: {corr}')