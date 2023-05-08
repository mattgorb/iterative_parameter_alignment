import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pickle
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

import seaborn as sns
sns.set_style("whitegrid")


ds='mnist'

#baseline=pd.read_csv(f'client_results/peer_contrib/benchmarks/client_results_ds_MNIST_model_MLP_ds_MNIST_seed_1_n_cli_20_None_ds_split_classimbalance_ds_alpha_0.3_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_False_le_25_s_True_rand_top_False.csv')
ipa=pd.read_csv(f'client_results/peer_contrib/benchmarks/client_results_ds_MNIST_model_MLP_ds_MNIST_seed_1_n_cli_20_None_ds_split_classimbalance_ds_alpha_0.3_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_False_le_5_s_False_rand_top_False.csv')

#baseline=pd.read_csv(f'client_results/peer_contrib/benchmarks/client_results_ds_CIFAR10_model_Conv4_ds_CIFAR10_seed_1_n_cli_10_None_ds_split_classimbalance_ds_alpha_0.3_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_False_le_50_s_True_rand_top_True.csv')
#ipa=pd.read_csv(f'client_results/peer_contrib/benchmarks/client_results_ds_CIFAR10_model_Conv4_ds_CIFAR10_seed_1_n_cli_10_None_ds_split_classimbalance_ds_alpha_0.3_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_False_le_5_s_False_rand_top_True.csv')

with sns.color_palette("crest", n_colors=10):
    corrs=[]
    for client in np.unique(ipa['client_list'].values):
        df=ipa[ipa['client_list']==client]
        #df=df.groupby(['client_list'])[metric].mean()
        print(df['test_accuracy_list'].values)
        #sys.exit()
        #plt.scatter(t, y, c='Blues', marker='.')
        #plt.plot([i for i in range(len(df['test_accuracy_list'].values))][5:30], df['test_accuracy_list'].values[5:30], '-.', cmap="Blues")



        #plt.plot(np.random.rand(5, 10))
        #plt.scatter([i for i in range(len(df['test_accuracy_list'].values))][6:30], df['test_accuracy_list'].values[6:30],)#  c=df["model_num"], cmap="Blues", marker='.')
        #with sns.color_palette("Blues", n_colors=20):
        plt.plot([i for i in range(len(df['test_accuracy_list'].values))][:30], df['test_accuracy_list'].values[:30], '-')


plt.title('MNIST, Convergence Rates of Peers', size=18)

plt.xlim([0,6])
plt.ylabel("Test Accuracy (%)" ,size=16)
plt.xlabel("Communication Round",size=16)

plt.tight_layout()

plt.savefig(f'fairness_by_epoch_{ds}_basic.pdf')