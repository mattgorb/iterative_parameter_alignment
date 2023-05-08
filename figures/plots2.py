#https://github.com/olgabot/prettyplotlib/wiki/Examples-with-code#plot-lines-eg-time-series-with-a-legend%22

import pandas as pd
import numpy as np
# prettyplotlib imports
import matplotlib.pyplot as plt


import seaborn as sns
sns.set_style("whitegrid")


ds='fashion'
seed=43

#df1=pd.read_csv('client_results/full/client_results_ds_CIFAR10_model_Conv4_n_cli_20_ds_split_dirichlet_ds_alpha_0.3_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_True_le_5_s_False_rand_top_False.csv')
#df1=pd.read_csv(f'client_results/full_new/client_results_ds_CIFAR10_model_Conv4_ds_CIFAR10_seed_{seed}_n_cli_20_None_ds_split_dirichlet_ds_alpha_0.25_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_False_le_5_s_False_rand_top_True.csv')

df1=pd.read_csv('client_results/full/client_results_ds_Fashion_MNIST_model_MLP_n_cli_20_ds_split_dirichlet_ds_alpha_0.25_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_True_le_1_s_False_rand_top_True.csv')

#df1=pd.read_csv('client_results/full/client_results_ds_MNIST_model_MLP_ds_MNIST_seed_24_n_cli_20_None_ds_split_dirichlet_ds_alpha_0.25_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_False_le_5_s_False_rand_top_True.csv')

print(np.unique(df1['client_list'].values))

#df.assign(m=df.mean(axis=1)).sort_values('m').drop('m', axis=1)
with sns.color_palette("crest", n_colors=5):
    for peer in list(np.unique(df1['client_list'].values)):
        plt.plot([i for i in range(len(df1[df1['client_list']==peer]['test_accuracy_list'].values[2:],))],
                 df1[df1['client_list']==peer]['test_accuracy_list'].values[2:],'-.')
    #plt.plot(df1[df1['client_list']==2]['test_accuracy_list'].values,label='Peer 2 (Labels 5-9)')

plt.ylabel('Peer Test Accuracy')
plt.xlabel('Communication Round')
plt.legend(loc='lower right')

#plt.xlim([0,100])
plt.ylim([55,90])

plt.tight_layout()

plt.savefig(f'{ds}_{seed}.png', bbox_inches='tight')

