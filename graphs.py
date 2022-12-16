from matplotlib import pyplot as plt
import pandas as pd


df=pd.read_csv('norms/weight_diff_double.csv')
set_mi=list(set(df['merge_iter'].values))

x=list(df.groupby('merge_iter')['epoch'].aggregate(['max']).values)[:15]

plt.clf()
plt.plot([i for i in range(len(df['weight_diff_layer1'].values))], df['weight_diff_layer1'], '.-')

#plt.plot([i for i in range(len(df['weight_diff_layer2'].values))], df['weight_diff_layer2'], '.-')

#plt.vlines(x, ymin=min(df['weight_diff_layer1'].values), ymax=max(df['weight_diff_layer1'].values), linestyles='dashed', color='blue')
plt.show()