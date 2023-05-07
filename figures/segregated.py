import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

df1=pd.read_csv('split_label_exps/split5/client_results_ds_MNIST_model_MLP_n_cli_5_ds_split_disjoint_classes_ds_alpha_0.3_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_True_le_1_s_False.csv')
#df2=pd.read_csv( 'split_label_exps/split5/run-FedDyn_mnist_2NN_n_cli_5_rule_split_label5_rule_arg_0.3_SGD_S100_F1.000000_Lr0.100000_1_1.000000_B50_E1_W0.001000_a0.010000_seed0_lrdecay0.998000_Accuracy_test_Sel clients-tag-Loss_test.csv')
#df3=pd.read_csv( 'split_label_exps/split5/run-FedDC_0.1mnist_2NN_n_cli_5_rule_split_label5_rule_arg_0.3_SGD_S100_F1.000000_Lr0.100000_1_1.000000_B50_E1_W0.001000_a0.100000_seed0_lrdecay0.998000_Accuracy_test_Sel clients-tag-Loss_test.csv')

df2=pd.read_csv( 'split_label_exps/split5/run-FedDyn_mnist_2NN_n_cli_5_rule_split_label5_rule_arg_0.3_SGD_S100_F1.000000_Lr0.100000_1_1.000000_B50_E1_W0.001000_a0.010000_seed0_lrdecay0.998000_Accuracy_test_Current cloud-tag-Accuracy_test.csv')
df3=pd.read_csv( 'split_label_exps/split5/run-FedDC_0.1mnist_2NN_n_cli_5_rule_split_label5_rule_arg_0.3_SGD_S100_F1.000000_Lr0.100000_1_1.000000_B50_E1_W0.001000_a0.100000_seed0_lrdecay0.998000_Accuracy_test_Current cloud-tag-Accuracy_test.csv')
df4=pd.read_csv( 'split_label_exps/split5/run-FedAvg_mnist_2NN_n_cli_5_rule_split_label5_rule_arg_0.3_SGD_S100_F1.000000_Lr0.100000_1_1.000000_B50_E1_W0.001000_lrdecay0.998000_seed0_Accuracy_test_Sel clients-tag-Accuracy_test.csv')

df2['smoothed']=my_tb_smooth(df2['Value'], 0.75)
df3['smoothed']=my_tb_smooth(df3['Value'], 0.75)

#df4=pd.read_csv( 'split_label_exps/split5/')

col=4
plt.clf()
sns.set_style('whitegrid')
fig, axs = plt.subplots(1, col,  figsize=(5*col,5.5))


sns.set_style('whitegrid')

sns.lineplot(df2.smoothed[:140]*100, label='FedDyn',ax=axs[2], linestyle='-.', color='C0',linewidth=1.5, legend=False)
sns.lineplot(df3.smoothed[:140]*100, label='FedDC',ax=axs[2], linestyle='-.', color='C1',linewidth=1.5, legend=False)
sns.lineplot(df4.Value[:140]*100, label='FedAvg',ax=axs[2], linestyle='-.',color='C3',linewidth=1.5,  legend=False)

sns.lineplot(df1[df1['client_list']==1]['test_accuracy_list'].values[:140] ,ax=axs[2],color='C2', linestyle='-',label='Peer Model',linewidth=2,  legend=False)
sns.lineplot(df1[df1['client_list']==2]['test_accuracy_list'].values[:140] ,ax=axs[2],color='C2', linestyle='-',label='Peer Model',linewidth=2,  legend=False)
sns.lineplot(df1[df1['client_list']==3]['test_accuracy_list'].values[:140] ,ax=axs[2],color='C2', linestyle='-',label='Peer Model',linewidth=2,  legend=False)
sns.lineplot(df1[df1['client_list']==4]['test_accuracy_list'].values[:140] ,ax=axs[2], color='C2',linestyle='-',label='Peer Model',linewidth=2,  legend=False)
sns.lineplot(df1[df1['client_list']==5]['test_accuracy_list'].values[:140] ,ax=axs[2],color='C2', linestyle='-',label='Peer Model',linewidth=2,  legend=False)



axs[2].set_title(label='MNIST 5 Peers\n2 Classes Per Peer', fontdict = {'fontsize' : 22})
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
sns.lineplot(df1[df1['client_list']==1]['test_accuracy_list'].values[:1000] ,ax=axs[1],color='C2', linestyle='-',label='Peer Model',linewidth=1.5,  legend=False)
sns.lineplot(df1[df1['client_list']==2]['test_accuracy_list'].values[:1000] ,ax=axs[1],color='C2', linestyle='-',label='Peer Model',linewidth=1.5,  legend=False)

axs[1].set_title(label='CIFAR10 2 Peers\n7/3 Class Split', fontdict = {'fontsize' : 22})






df1=pd.read_csv('split_label_exps/split2/fashion/client_results_ds_Fashion_MNIST_model_MLP_n_cli_2_ds_split_disjoint_classes_ds_alpha_0.3_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_True_le_1_s_False.csv')
df2=pd.read_csv('split_label_exps/split2/fashion/run-FedDyn_fashionmnist_2NN_n_cli_2_rule_split_label_rule_arg_0.3_SGD_S100_F1.000000_Lr0.100000_1_1.000000_B50_E1_W0.001000_a0.010000_seed0_lrdecay0.998000_Accuracy_test_Current cloud-tag-Accuracy_test.csv')
df3=pd.read_csv('split_label_exps/split2/fashion/run-FedDC_0.1fashionmnist_2NN_n_cli_2_rule_split_label_rule_arg_0.3_SGD_S100_F1.000000_Lr0.100000_1_1.000000_B50_E1_W0.001000_a0.100000_seed0_lrdecay0.998000_Accuracy_test_Current cloud-tag-Accuracy_test.csv')
df4=pd.read_csv('split_label_exps/split2/fashion/run-FedAvg_fashionmnist_2NN_n_cli_2_rule_split_label_rule_arg_0.3_SGD_S100_F1.000000_Lr0.100000_1_1.000000_B50_E1_W0.001000_lrdecay0.998000_seed0_Accuracy_test_Sel clients-tag-Accuracy_test.csv')

df2['smoothed']=my_tb_smooth(df2['Value'], 0.9)
df3['smoothed']=my_tb_smooth(df3['Value'], 0.9)

sns.lineplot(df2.smoothed[:750]*100, label='FedDyn',ax=axs[0], linestyle='-.', color='C0',linewidth=2.5, legend=False)
sns.lineplot(df3.smoothed[:750]*100, label='FedDC',ax=axs[0], linestyle='-.', color='C1',linewidth=2.5, legend=False)
sns.lineplot(df4.Value[:750]*100, label='FedAvg',ax=axs[0], linestyle='-.',color='C3',linewidth=2.5,  legend=False)
sns.lineplot(df1[df1['client_list']==1]['test_accuracy_list'].values[:750] ,ax=axs[0],color='C2', linestyle='-',label='Peer Model',linewidth=2.5,  legend=False)
sns.lineplot(df1[df1['client_list']==2]['test_accuracy_list'].values[:750] ,ax=axs[0],color='C2', linestyle='-',label='Peer Model',linewidth=2.5,  legend=False)
axs[0].set_title(label='Fashion MNIST 2 Peers\n5 Classes Per Peer', fontdict = {'fontsize' : 22})







df1=pd.read_csv('split_label_exps/split3/c10/client_results_ds_CIFAR10_model_Conv4_n_cli_3_ds_split_disjoint_classes_ds_alpha_None_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_True_le_5_s_False_rand_top_False.csv')
df2=pd.read_csv('split_label_exps/split3/c10/run-FedDyn_cifar10_Conv4_n_cli_3_rule_split_label3_rule_arg_0.3_SGD_S200_F1.000000_Lr0.100000_1_1.000000_B50_E5_W0.001000_a0.010000_seed0_lrdecay0.998000_Accuracy_test_Current cloud-tag-Accuracy_test.csv')
df3=pd.read_csv('split_label_exps/split3/c10/run-FedDC_0.01cifar10_Conv4_n_cli_3_rule_split_label3_rule_arg_0.3_SGD_S200_F1.000000_Lr0.100000_1_1.000000_B50_E5_W0.001000_a0.010000_seed0_lrdecay0.998000_Accuracy_test_Current cloud-tag-Accuracy_test.csv')
#df4=pd.read_csv('split_label_exps/split3/c10/')

df2['smoothed']=my_tb_smooth(df2['Value'], 0.95)
df3['smoothed']=my_tb_smooth(df3['Value'], 0.95)


#df3['smoothed']=my_tb_smooth(df3['Value'], 0.5)

sns.lineplot(df2.smoothed[:1500]*100, label='FedDyn',ax=axs[3], linestyle='-.', color='C0',linewidth=1.5, legend=False)
sns.lineplot(df3.smoothed[:1500]*100, label='FedDC',ax=axs[3], linestyle='-.', color='C1',linewidth=1.5, legend=False)
#sns.lineplot(df4.Value[:1500], label='FedAvg',ax=axs[3], linestyle='-.',color='C3',linewidth=1.5,  legend=False)
sns.lineplot(my_tb_smooth(df1[df1['client_list']==1]['test_accuracy_list'].values[:1500],0.9) ,ax=axs[3],color='C2', linestyle='-',label='Peer Model',linewidth=2,  legend=False)
sns.lineplot(my_tb_smooth(df1[df1['client_list']==2]['test_accuracy_list'].values[:1500],0.9) ,ax=axs[3],color='C2', linestyle='-',label='Peer Model',linewidth=2,  legend=False)
sns.lineplot(my_tb_smooth(df1[df1['client_list']==3]['test_accuracy_list'].values[:1500],0.9) ,ax=axs[3],color='C2', linestyle='-',label='Peer Model',linewidth=2,  legend=False)


axs[3].set_title(label='CIFAR10 3 Peers\nSTILL RUNNING', fontdict = {'fontsize' : 22})








for ax in axs:
    ax.set_xlabel("Communication Round", fontdict = {'fontsize' : 14})
    #ax.set_ylim([0, 2])
    #ax.get_legend().remove()
axs[0].set_ylabel("Test Accuracy (%)", fontdict = {'fontsize' : 20})

#plt.xticks([.2,.5, .5, .6,.8,.9])
#plt.xticks([20, 40,50,60,80, 90])
#plt.xlim([19.9, 90.1])
handles, labels = axs[0].get_legend_handles_labels()
handles=[handles[3], handles[2], handles[1], handles[0],  ]
labels=[labels[3], labels[2], labels[1], labels[0],   ]
fig.legend(handles, labels, loc='lower center', ncol=4,bbox_to_anchor=(.5, -.015), prop={'size': 19})

axs[2].axes.set_xlim(0, 140)
axs[2].axes.set_ylim(50, 100)
axs[1].axes.set_xlim(0, 1000)
axs[1].axes.set_ylim(40, 90)
axs[0].axes.set_xlim(0, 750)
axs[0].axes.set_ylim(55, 90)
axs[3].axes.set_xlim(0, 1500)
axs[3].axes.set_ylim(40, 80)

plt.tight_layout(rect=[0,0.075,1,1], )
#print(labels)

plt.savefig('segregated.png')










