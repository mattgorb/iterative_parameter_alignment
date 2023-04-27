"""
IMAGES: PEER contribution equation vs accuracy at Epoch K
Convergence Rate versus baseline
"""

import matplotlib.pyplot as plt
import math
import pickle
import pandas as pd
import numpy as np

#Example
#x=[0.10,0.11,0.10 ,0.09, 0.09,0.11  ,  0.10 ,   0.10 ,   0.12 ,   0.08]
#np.uniformity=1-(np.linalg.norm(x)*math.sqrt(10)-1)/(math.sqrt(10)-1)



ds='c10'
class_dict=f'/Users/matthewgorbett/PycharmProjects/iterative_weight_alignment/figures/dataset_splits' \
           f'/dataset_split_info_model_Conv4_n_cli_10_ds_split_dirichlet_ds_alpha_0.3_align_ae_' \
           f'waf_1_delta_None_init_type_kaiming_normal_same_init_False_le_5_s_False.pkl'


classes = pickle.load( open( class_dict, "rb" ) )


results=f'/Users/matthewgorbett/PycharmProjects/iterative_weight_alignment/figures/client_results/' \
        f'client_results_ds_CIFAR10_model_Conv4_n_cli_10_ds_split_dirichlet_ds_alpha_0.3_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_False_le_5_s_False.csv'


df=pd.read_csv(results)


df=df[(df['iter_list']>100)&(df['iter_list']<150) ]
df=df.groupby(['client_list'])['test_accuracy_list'].mean()#.agg({'test_accuracy_list':['mean','std']})
print(df.head(12))


print(classes)

total_samples=0
for peer in classes.items():
    peer, class_ls=peer[0],peer[1]
    total_samples+=sum(class_ls.values())

print('Total samples')
print(total_samples)

freq_dict={}

if ds=='c10':
    label_len=10
else:
    label_len=100


def pad_or_truncate(some_list, target_len):
    return some_list[:target_len] + [0]*(target_len - len(some_list))

uniformity_ls=[]
for peer in classes.items():
    peer, class_ls=peer[0],peer[1]
    '''freqs=[]
    for label, num in class_ls.items():
        freqs.append(num/sum(class_ls.values()))
        #freqs.append(num/5000)

    freq_dict[peer]=freqs

    total_num=(sum(class_ls.values()))'''
    #uniformity_ls.append(len(class_ls.keys()))
    #uniformity_ls.append((1-(np.linalg.norm(freqs)*math.sqrt(label_len)-1)/(math.sqrt(label_len)-1)))

    vals=list(class_ls.values())
    #print(vals)
    vals=pad_or_truncate(vals,10)
    #print(vals)
    uniformity_ls.append(1/np.std(vals))
    #print(f'{total_num}, { (1-(np.linalg.norm(freqs)*math.sqrt(label_len)-1)/(math.sqrt(label_len)-1))},'
    #      f' { total_num*(1-(np.linalg.norm(freqs)*math.sqrt(label_len)-1)/(math.sqrt(label_len)-1))}')

print(uniformity_ls)
#sys.exit()

#np.uniformity=1-(np.linalg.norm(x)*math.sqrt(10)-1)/(math.sqrt(10)-1)
print(df)
print(df.values)

plt.clf()
plt.plot(uniformity_ls, df.values, '.')
plt.savefig('/Users/matthewgorbett/PycharmProjects/iterative_weight_alignment/figures/a.png')