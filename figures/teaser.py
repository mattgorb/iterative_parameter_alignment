import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df1=pd.read_csv('split_label_exps/split2/')
df2=pd.read_csv( 'split_label_exps/split2/')
df2=pd.read_csv( 'split_label_exps/split2/')
df2=pd.read_csv( 'split_label_exps/split2/')



plt.clf()
sns.set_style('whitegrid')


sns.lineplot(df2.Value[:140]*100, label='FedDyn',ax=axs[0], linestyle='-.', color='C0',linewidth=1.5, legend=False)
sns.lineplot(df3.Value[:140]*100, label='FedDC',ax=axs[0], linestyle='-.', color='C1',linewidth=1.5, legend=False)
sns.lineplot(df4.Value[:140]*100, label='FedAvg',ax=axs[0], linestyle='-.',color='C3',linewidth=1.5,  legend=False)

sns.lineplot(df1[df1['client_list']==1]['test_accuracy_list'].values[:140] ,ax=axs[0],color='C2', linestyle='-',label='Peer Model',linewidth=1.5,  legend=False)
sns.lineplot(df1[df1['client_list']==2]['test_accuracy_list'].values[:140] ,ax=axs[0],color='C2', linestyle='-',label='Peer Model',linewidth=1.5,  legend=False)
sns.lineplot(df1[df1['client_list']==3]['test_accuracy_list'].values[:140] ,ax=axs[0],color='C2', linestyle='-',label='Peer Model',linewidth=1.5,  legend=False)
sns.lineplot(df1[df1['client_list']==4]['test_accuracy_list'].values[:140] ,ax=axs[0], color='C2',linestyle='-',label='Peer Model',linewidth=1.5,  legend=False)
sns.lineplot(df1[df1['client_list']==5]['test_accuracy_list'].values[:140] ,ax=axs[0],color='C2', linestyle='-',label='Peer Model',linewidth=1.5,  legend=False)
