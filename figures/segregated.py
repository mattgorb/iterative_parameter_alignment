import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pickle
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pylab as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pickle
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase
sns.set_style('whitegrid')
import random
import numpy as np                                    # v 1.19.2
import matplotlib.pyplot as plt                       # v 3.3.2
import matplotlib.patches as patches
from matplotlib.legend_handler import HandlerTuple
from matplotlib.colors import ListedColormap
import seaborn as sns
def my_tb_smooth(scalars: list[float], weight: float) -> list[float]:  # Weight between 0 and 1
    """

    ref: https://stackoverflow.com/questions/42011419/is-it-possible-to-call-tensorboard-smooth-function-manually

    :param scalars:
    :param weight:
    :return:
    """
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed: list = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return smoothed


colors = sns.color_palette("crest", n_colors=5).as_hex()
col=5
plt.clf()
sns.set_style('whitegrid')
fig, axs = plt.subplots(1, col,  figsize=(5*col,5.5))


df1=pd.read_csv('split_label_exps/split2/cifar10/client_results_ds_CIFAR10_model_Conv4_ds_CIFAR10_seed_32_n_cli_2_None_ds_split_disjoint_classes_ds_alpha_None_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_True_le_1_s_False_rand_top_True.csv')
df2=pd.read_csv('split_label_exps/split2/cifar10/run-FedDyn_cifar10_Conv4_n_cli_2_rule_split_label_rule_arg_0.3_SGD_S200_F1.000000_Lr0.100000_1_1.000000_B50_E5_W0.001000_a0.010000_seed0_lrdecay0.998000_Accuracy_test_Current cloud-tag-Accuracy_test.csv')
df3=pd.read_csv('split_label_exps/split2/cifar10/run-FedDC_0.01cifar10_Conv4_n_cli_2_rule_split_label_rule_arg_0.3_SGD_S200_F1.000000_Lr0.100000_1_1.000000_B50_E5_W0.001000_a0.010000_seed0_lrdecay0.998000_Accuracy_test_Current cloud-tag-Accuracy_test.csv')
df4=pd.read_csv('split_label_exps/split2/cifar10/run-FedAvg_cifar10_Conv4_n_cli_2_rule_split_label_rule_arg_0.6_SGD_S200_F1.000000_Lr0.100000_1_1.000000_B50_E5_W0.001000_lrdecay0.998000_seed0_Accuracy_test_Sel clients-tag-Accuracy_test.csv')

df2['smoothed']=my_tb_smooth(df2['Value'], 0.9)
df3['smoothed']=my_tb_smooth(df3['Value'], 0.9)

sns.lineplot(df2.smoothed[:1950]*100, label='FedDyn',ax=axs[0], linestyle='-.', color='C0',linewidth=1.5, legend=False)
sns.lineplot(df3.smoothed[:1950]*100, label='FedDC',ax=axs[0], linestyle='-.', color='C1',linewidth=1.5, legend=False)
sns.lineplot(df4.Value[:1950]*100, label='FedAvg',ax=axs[0], linestyle='-.',color='C3',linewidth=1.5,  legend=False)


p1=my_tb_smooth(df1[df1['client_list']==0]['test_accuracy_list'].values[:1950], 0.6)
p2=my_tb_smooth(df1[df1['client_list']==1]['test_accuracy_list'].values[:1950], 0.6)
sns.lineplot(p1 ,ax=axs[0],color=colors[4], linestyle='-',label='Peer Model',linewidth=1.5,  legend=False)
sns.lineplot(p2 ,ax=axs[0],color=colors[1], linestyle='-',label='Peer Model',linewidth=1.5,  legend=False)

axs[0].set_title(label='CIFAR10 2 Peers\n5 Labels Per Peer', fontdict = {'fontsize' : 26})



















df1=pd.read_csv('split_label_exps/split5/client_results_ds_MNIST_model_MLP_n_cli_5_ds_split_disjoint_classes_ds_alpha_0.3_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_True_le_1_s_False.csv')
df2=pd.read_csv( 'split_label_exps/split5/run-FedDyn_mnist_2NN_n_cli_5_rule_split_label5_rule_arg_0.3_SGD_S100_F1.000000_Lr0.100000_1_1.000000_B50_E1_W0.001000_a0.010000_seed0_lrdecay0.998000_Accuracy_test_Current cloud-tag-Accuracy_test.csv')
df3=pd.read_csv( 'split_label_exps/split5/run-FedDC_0.1mnist_2NN_n_cli_5_rule_split_label5_rule_arg_0.3_SGD_S100_F1.000000_Lr0.100000_1_1.000000_B50_E1_W0.001000_a0.100000_seed0_lrdecay0.998000_Accuracy_test_Current cloud-tag-Accuracy_test.csv')
df4=pd.read_csv( 'split_label_exps/split5/run-FedAvg_mnist_2NN_n_cli_5_rule_split_label5_rule_arg_0.3_SGD_S100_F1.000000_Lr0.100000_1_1.000000_B50_E1_W0.001000_lrdecay0.998000_seed0_Accuracy_test_Sel clients-tag-Accuracy_test.csv')

df2['smoothed']=my_tb_smooth(df2['Value'], 0.75)
df3['smoothed']=my_tb_smooth(df3['Value'], 0.75)

#df4=pd.read_csv( 'split_label_exps/split5/')




sns.set_style('whitegrid')

sns.lineplot(df2.smoothed[:140]*100, label='FedDyn',ax=axs[2], linestyle='-.', color='C0',linewidth=2.5, legend=False)
sns.lineplot(df3.smoothed[:140]*100, label='FedDC',ax=axs[2], linestyle='-.', color='C1',linewidth=2.5, legend=False)
sns.lineplot(df4.Value[:140]*100, label='FedAvg',ax=axs[2], linestyle='-.',color='C3',linewidth=2.5,  legend=False)




sns.lineplot(df1[df1['client_list']==1]['test_accuracy_list'].values[:140] ,ax=axs[2],color=colors[0], linestyle='-',label='Peer Model',linewidth=2,  legend=False)
sns.lineplot(df1[df1['client_list']==2]['test_accuracy_list'].values[:140] ,ax=axs[2],color=colors[1], linestyle='-',label='Peer Model',linewidth=2,  legend=False)
sns.lineplot(df1[df1['client_list']==3]['test_accuracy_list'].values[:140] ,ax=axs[2],color=colors[2], linestyle='-',label='Peer Model',linewidth=2,  legend=False)
sns.lineplot(df1[df1['client_list']==4]['test_accuracy_list'].values[:140] ,ax=axs[2], color=colors[3],linestyle='-',label='Peer Model',linewidth=2,  legend=False)
sns.lineplot(df1[df1['client_list']==5]['test_accuracy_list'].values[:140] ,ax=axs[2],color=colors[4], linestyle='-',label='Peer Model',linewidth=2,  legend=False)



axs[2].set_title(label='MNIST 5 Peers\n2 Labels Per Peer', fontdict = {'fontsize' : 26})
#axs[0].axhline(98, linestyle=':', linewidth=2, color='c',label='Target Accuracy' )





df1=pd.read_csv('split_label_exps/split73/client_results_ds_CIFAR10_model_Conv4_ds_CIFAR10_seed_1_n_cli_2_73_ds_split_disjoint_classes_ds_alpha_None_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_False_le_1_s_False_rand_top_False.csv')
df2=pd.read_csv('split_label_exps/split73/run-FedDyn_cifar10_Conv4_n_cli_2_rule_split_label73_rule_arg_0.3_SGD_S200_F1.000000_Lr0.100000_1_1.000000_B50_E5_W0.001000_a0.010000_seed0_lrdecay0.998000_Accuracy_test_Current cloud-tag-Accuracy_test.csv')
df3=pd.read_csv('split_label_exps/split73/run-FedDC_0.01cifar10_Conv4_n_cli_2_rule_split_label73_rule_arg_0.3_SGD_S200_F1.000000_Lr0.100000_1_1.000000_B50_E5_W0.001000_a0.010000_seed0_lrdecay0.998000_Accuracy_test_Current cloud-tag-Accuracy_test.csv')
df4=pd.read_csv('split_label_exps/split73/run-FedAvg_cifar10_Conv4_n_cli_2_rule_split_label73_rule_arg_0.3_SGD_S200_F1.000000_Lr0.100000_1_1.000000_B50_E5_W0.001000_lrdecay0.998000_seed0_Accuracy_test_Sel clients-tag-Accuracy_test.csv')

df2['smoothed']=my_tb_smooth(df2['Value'], 0.9)
df3['smoothed']=my_tb_smooth(df3['Value'], 0.9)

sns.lineplot(df2.smoothed[:1000]*100, label='FedDyn',ax=axs[1], linestyle='-.', color='C0',linewidth=1.5, legend=False)
sns.lineplot(df3.smoothed[:1000]*100, label='FedDC',ax=axs[1], linestyle='-.', color='C1',linewidth=1.5, legend=False)
sns.lineplot(df4.Value[:1000]*100, label='FedAvg',ax=axs[1], linestyle='-.',color='C3',linewidth=1.5,  legend=False)
sns.lineplot(df1[df1['client_list']==1]['test_accuracy_list'].values[:1000] ,ax=axs[1],color=colors[1], linestyle='-',label='Peer Model',linewidth=1.5,  legend=False)
sns.lineplot(df1[df1['client_list']==2]['test_accuracy_list'].values[:1000] ,ax=axs[1],color=colors[4], linestyle='-',label='Peer Model',linewidth=1.5,  legend=False)

axs[1].set_title(label='CIFAR10 2 Peers\n7/3 Label Split', fontdict = {'fontsize' : 26})






df1=pd.read_csv('split_label_exps/split2/fashion/client_results_ds_Fashion_MNIST_model_MLP_n_cli_2_ds_split_disjoint_classes_ds_alpha_0.3_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_True_le_1_s_False.csv')
df2=pd.read_csv('split_label_exps/split2/fashion/run-FedDyn_fashionmnist_2NN_n_cli_2_rule_split_label_rule_arg_0.3_SGD_S100_F1.000000_Lr0.100000_1_1.000000_B50_E1_W0.001000_a0.010000_seed0_lrdecay0.998000_Accuracy_test_Current cloud-tag-Accuracy_test.csv')
df3=pd.read_csv('split_label_exps/split2/fashion/run-FedDC_0.1fashionmnist_2NN_n_cli_2_rule_split_label_rule_arg_0.3_SGD_S100_F1.000000_Lr0.100000_1_1.000000_B50_E1_W0.001000_a0.100000_seed0_lrdecay0.998000_Accuracy_test_Current cloud-tag-Accuracy_test.csv')
df4=pd.read_csv('split_label_exps/split2/fashion/run-FedAvg_fashionmnist_2NN_n_cli_2_rule_split_label_rule_arg_0.3_SGD_S100_F1.000000_Lr0.100000_1_1.000000_B50_E1_W0.001000_lrdecay0.998000_seed0_Accuracy_test_Sel clients-tag-Accuracy_test.csv')

df2['smoothed']=my_tb_smooth(df2['Value'], 0.9)
df3['smoothed']=my_tb_smooth(df3['Value'], 0.9)

sns.lineplot(df2.smoothed[:750]*100, label='FedDyn',ax=axs[3], linestyle='-.', color='C0',linewidth=2.5, legend=False)
sns.lineplot(df3.smoothed[:750]*100, label='FedDC',ax=axs[3], linestyle='-.', color='C1',linewidth=2.5, legend=False)
sns.lineplot(df4.Value[:750]*100, label='FedAvg',ax=axs[3], linestyle='-.',color='C3',linewidth=2.5,  legend=False)
sns.lineplot(df1[df1['client_list']==1]['test_accuracy_list'].values[:750] ,ax=axs[3],color=colors[1], linestyle='-',linewidth=2.5,  legend=False)
sns.lineplot(df1[df1['client_list']==2]['test_accuracy_list'].values[:750] ,ax=axs[3],color=colors[4], linestyle='-',linewidth=2.5,  legend=False)
axs[3].set_title(label='Fashion MNIST 2 Peers\n5 Labels Per Peer', fontdict = {'fontsize' : 26})















df1=pd.read_csv('split_label_exps/split2/cifar100/client_results_ds_CIFAR100_model_Conv4_Cifar100_ds_CIFAR100_seed_32_n_cli_2_None_ds_split_disjoint_classes_ds_alpha_None_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_True_le_1_s_False_rand_top_True.csv')
df2=pd.read_csv('split_label_exps/split2/cifar100/run-FedDyn_cifar100_Conv4_n_cli_2_rule_split_label_rule_arg_0.3_SGD_S200_F0.150000_Lr0.100000_1_1.000000_B50_E5_W0.001000_a0.010000_seed0_lrdecay0.998000_Accuracy_test_Sel clients-tag-Accuracy_test.csv')
df3=pd.read_csv('split_label_exps/split2/cifar100/run-FedDC_0.01cifar100_Conv4_n_cli_2_rule_split_label_rule_arg_0.3_SGD_S200_F0.150000_Lr0.100000_1_1.000000_B50_E5_W0.001000_a0.010000_seed0_lrdecay0.998000_Accuracy_test_Sel clients-tag-Accuracy_test.csv')
df4=pd.read_csv('split_label_exps/split2/cifar100/run-FedAvg_cifar100_Conv4_n_cli_2_rule_split_label_rule_arg_0.3_SGD_S200_F0.150000_Lr0.100000_1_1.000000_B50_E5_W0.001000_lrdecay0.998000_seed0_Accuracy_test_Sel clients-tag-Accuracy_test.csv')

df2['smoothed']=my_tb_smooth(df2['Value'], 0.85)
df3['smoothed']=my_tb_smooth(df3['Value'], 0.85)
df4['smoothed']=my_tb_smooth(df4['Value'], 0.85)

sns.lineplot(df2.smoothed[:1500]*100, label='FedDyn',ax=axs[4], linestyle='-.', color='C0',linewidth=1.5, legend=False)
sns.lineplot(df3.smoothed[:1500]*100, label='FedDC',ax=axs[4], linestyle='-.', color='C1',linewidth=1.5, legend=False)
sns.lineplot(df4.smoothed[:1500]*100, label='FedAvg',ax=axs[4], linestyle='-.',color='C3',linewidth=1.5,  legend=False)
sns.lineplot(my_tb_smooth(df1[df1['client_list']==1]['test_accuracy_list'].values[:1500],0.6) ,ax=axs[4],color=colors[0], linestyle='-',linewidth=2,  legend=False)
sns.lineplot(my_tb_smooth(df1[df1['client_list']==0]['test_accuracy_list'].values[:1500],0.6) ,ax=axs[4],color=colors[3], linestyle='-',linewidth=2,  legend=False)


axs[4].set_title(label='CIFAR100 2 Peers\n5 Labels Per Peer', fontdict = {'fontsize' : 26})








for ax in axs:
    ax.set_xlabel("Communication Round", fontdict = {'fontsize' : 18})
    #ax.set_ylim([0, 2])
    #ax.get_legend().remove()
#axs[0].set_ylabel("Test Accuracy (%)", fontdict = {'fontsize' : 20})

axs[0].set_ylabel("Test Accuracy (%)", fontdict = {'fontsize' : 25})
axs[1].set_ylabel("", fontdict = {'fontsize' : 1})
axs[2].set_ylabel("", fontdict = {'fontsize' : 1})
axs[3].set_ylabel("", fontdict = {'fontsize' : 1})
axs[4].set_ylabel("", fontdict = {'fontsize' : 1})
#axs[0].set_xticks(fontdict = {'fontsize' : 25})
axs[0].tick_params(axis='both', which='major', labelsize=15)
axs[1].tick_params(axis='both', which='major', labelsize=15)
axs[2].tick_params(axis='both', which='major', labelsize=15)
axs[3].tick_params(axis='both', which='major', labelsize=15)
axs[4].tick_params(axis='both', which='major', labelsize=15)

ncolors = 8
cmaps_names = ['Peer Models', ]
cmaps = [sns.color_palette("crest", n_colors=8, as_cmap=True ), sns.color_palette("flare", n_colors=8, as_cmap=True )]
cmaps_gradients = [cmap(np.linspace(0, 1, ncolors)) for cmap in cmaps]
cmaps_dict = dict(zip(cmaps_names, cmaps_gradients))

# Create a list of lists of patches representing the gradient of each colormap
patches_cmaps_gradients = []
for cmap_name, cmap_colors in cmaps_dict.items():
    cmap_gradient = [patches.Patch(facecolor=c, edgecolor=c, label=cmap_name)
                     for c in cmap_colors]
    patches_cmaps_gradients.append(cmap_gradient)


#fig.legend(handles, labels, loc='lower right', )


handles, labels =ax.get_legend_handles_labels()

patches_cmaps_gradients.extend(handles)
cmaps_names.extend(labels)
print(cmaps_names)
print(patches_cmaps_gradients)
#
# Create custom legend (with a large fontsize to better illustrate the result)
#plt.legend(handles=patches_cmaps_gradients, labels=cmaps_names,
#           fontsize=16,  handler_map={list: HandlerTuple(ndivide=None, pad=0)},
#           loc='lower right')

#handles, labels = axs[0].get_legend_handles_labels()
#handles=[handles[3], handles[2], handles[1], handles[0],  ]
#labels=[labels[3], labels[2], labels[1], labels[0],   ]
#fig.legend(handles=patches_cmaps_gradients, labels=cmaps_names, loc='lower center', ncol=4,bbox_to_anchor=(.5, -.015), prop={'size': 19})
fig.legend(handles=patches_cmaps_gradients, labels=cmaps_names, loc='lower center', ncol=4,bbox_to_anchor=(.5, -.035),
           prop={'size': 25},handler_map={list: HandlerTuple(ndivide=None, pad=0)},)








axs[2].axes.set_xlim(0, 140)
axs[2].axes.set_ylim(50, 100)
axs[1].axes.set_xlim(0, 1000)
axs[1].axes.set_ylim(40, 90)
axs[0].axes.set_xlim(0, 1950)
axs[0].axes.set_ylim(40, 90)
axs[4].axes.set_xlim(0, 1000)
axs[4].axes.set_ylim(20, 55)
axs[3].axes.set_xlim(0, 750)
axs[3].axes.set_ylim(50, 90)


plt.tight_layout(rect=[0,0.1,1,1], )

plt.savefig('segregated.pdf')










