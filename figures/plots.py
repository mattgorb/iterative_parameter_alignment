#https://github.com/olgabot/prettyplotlib/wiki/Examples-with-code#plot-lines-eg-time-series-with-a-legend%22

import pandas as pd
import numpy as np
# prettyplotlib imports
import matplotlib.pyplot as plt


import seaborn as sns


df1=pd.read_csv('split_label_exps/'
               'client_results_ds_CIFAR10_model_Conv4_n_cli_2_ds_split_disjoint_classes_ds_alpha_None_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_True_le_1_s_False.csv')

df2=pd.read_csv(
    'split_label_exps/split2/cifar10/run-FedAvg_cifar10_Conv4_n_cli_2_rule_split_label_rule_arg_0.6_SGD_S200_F1.000000_Lr0.100000_1_1.000000_B50_E5_W0.001000_lrdecay0.998000_seed0_Accuracy_test_Sel clients-tag-Accuracy_test.csv')

df3=pd.read_csv('/split_label_exps/split2/run-FedDyn_cifar10_Conv4_n_cli_2_rule_split_label_rule_arg_0.3_SGD_S200_F1.000000_Lr0.100000_1_1.000000_B50_E5_W0.001000_a0.010000_seed0_lrdecay0.998000_Accuracy_test_Sel clients-tag-Accuracy_test.csv')

df4=pd.read_csv('/split_label_exps/split2/run-FedDC_0.01cifar10_Conv4_n_cli_2_rule_split_label_rule_arg_0.3_SGD_S200_F1.000000_Lr0.100000_1_1.000000_B50_E5_W0.001000_a0.010000_seed0_lrdecay0.998000_Accuracy_test_Sel clients-tag-Accuracy_test.csv')

print(df2.Value.values*100)

sns.set_style("whitegrid")

plt.plot(df2.Value.values*100,label='FedAvg')
#plt.plot(df3.Value.values*100,label='FedDyn')
#plt.plot(df4.Value.values*100,label='FedDC')

plt.plot(df1[df1['client_list']==1]['test_accuracy_list'].values,label="Peer 1 (Labels 0-4)")
plt.plot(df1[df1['client_list']==2]['test_accuracy_list'].values,label='Peer 2 (Labels 5-9)')

plt.ylabel('Peer Test Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('cifar10_split.png', bbox_inches='tight')

