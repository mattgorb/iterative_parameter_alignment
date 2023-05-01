import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df1=pd.read_csv('client_results/client_results_ds_MNIST_model_MLP_n_cli_5_ds_split_disjoint_classes_ds_alpha_0.3_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_True_le_1_s_False.csv')
df2=pd.read_csv( 'split_label_exps/split5/run-FedDyn_mnist_2NN_n_cli_5_rule_split_label5_rule_arg_0.3_SGD_S100_F1.000000_Lr0.100000_1_1.000000_B50_E1_W0.001000_a0.010000_seed0_lrdecay0.998000_Accuracy_test_Sel clients-tag-Accuracy_test.csv')
df3=pd.read_csv( 'split_label_exps/split5/run-FedDC_0.1mnist_2NN_n_cli_5_rule_split_label5_rule_arg_0.3_SGD_S100_F1.000000_Lr0.100000_1_1.000000_B50_E1_W0.001000_a0.100000_seed0_lrdecay0.998000_Accuracy_test_Sel clients-tag-Accuracy_test.csv')
df4=pd.read_csv( 'split_label_exps/split5/run-FedAvg_mnist_2NN_n_cli_5_rule_split_label5_rule_arg_0.3_SGD_S100_F1.000000_Lr0.100000_1_1.000000_B50_E1_W0.001000_lrdecay0.998000_seed0_Accuracy_test_Sel clients-tag-Accuracy_test.csv')




col=3
plt.clf()
sns.set_style('whitegrid')
fig, axs = plt.subplots(1, col, sharex=True, figsize=(5*col,5.25))


sns.set_style('whitegrid')

sns.lineplot(df2.Value[:150]*100, label='FedDyn',ax=axs[0], linestyle='-.', color='C0',linewidth=1.5, legend=False)
sns.lineplot(df3.Value[:150]*100, label='FedDC',ax=axs[0], linestyle='-.', color='C1',linewidth=1.5, legend=False)
sns.lineplot(df4.Value[:150]*100, label='FedAvg',ax=axs[0], linestyle='-.',color='C3',linewidth=1.5,  legend=False)

sns.lineplot(df1[df1['client_list']==1]['test_accuracy_list'].values[:150] ,ax=axs[0],color='C2', linestyle='-',label='Peer Model',linewidth=1.5,  legend=False)
sns.lineplot(df1[df1['client_list']==2]['test_accuracy_list'].values[:150] ,ax=axs[0],color='C2', linestyle='-',label='Peer Model',linewidth=1.5,  legend=False)
sns.lineplot(df1[df1['client_list']==3]['test_accuracy_list'].values[:150] ,ax=axs[0],color='C2', linestyle='-',label='Peer Model',linewidth=1.5,  legend=False)
sns.lineplot(df1[df1['client_list']==4]['test_accuracy_list'].values[:150] ,ax=axs[0], color='C2',linestyle='-',label='Peer Model',linewidth=1.5,  legend=False)
sns.lineplot(df1[df1['client_list']==5]['test_accuracy_list'].values[:150] ,ax=axs[0],color='C2', linestyle='-',label='Peer Model',linewidth=1.5,  legend=False)

#axs[0].errorbar([20,40,50,60,80,90], [81.05,81.6,81.68,81.12,79.06,70.22] ,[0.2,0.22,.9,.3,.65,.66], label='asdf',linewidth =2,  )

axs[0].set_title(label='MNIST, 5 Peers', fontdict = {'fontsize' : 22})
axs[0].axhline(98, linestyle=':', linewidth=2, color='c',label='Target Accuracy' )





for ax in axs:
    ax.set_xlabel("Communication Round", fontdict = {'fontsize' : 14})
    #ax.get_legend().remove()
axs[0].set_ylabel("Test Accuracy", fontdict = {'fontsize' : 14})

#plt.xticks([.2,.5, .5, .6,.8,.9])
#plt.xticks([20, 40,50,60,80, 90])
#plt.xlim([19.9, 90.1])
handles, labels = axs[0].get_legend_handles_labels()
print(labels)
handles=[handles[0],  handles[1], handles[3], handles[2], handles[-1]]
labels=[labels[0], labels[1], labels[3],  labels[2], labels[-1]]
fig.legend(handles, labels, loc='lower left', ncol=5,bbox_to_anchor=(.04, .001), prop={'size': 14})



plt.tight_layout(rect=[0,.1,1,1])
#print(labels)

plt.savefig('segregated.png')