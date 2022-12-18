from matplotlib import pyplot as plt
import pandas as pd


df=pd.read_csv('norms/weight_diff_double.csv')
df=df.head(200)
plt.clf()
#plt.plot([i for i in range(len(df['weight_diff_layer1'].values))], df['weight_diff_layer1'], '.-')
plt.plot([i for i in range(len(df['weight_diff_layer2'].values))], df['weight_diff_layer2'], '.-')
x=set(df['epoch'].values)
x = [i for i in x if str(i) != 'nan']
print(x)
plt.vlines(x, ymin=min(df['weight_diff_layer1'].values), ymax=max(df['weight_diff_layer1'].values),)
plt.show()


'''
df=pd.read_csv('norms/norms_double.csv')
df=df[250:]
#df=df.head(100)
plt.clf()
#plt.plot([i for i in range(len(df['model1_fc2'].values))], df['model1_fc2'], '.-')
#plt.plot([i for i in range(len(df['model2_fc2'].values))], df['model2_fc2'], '.-')
#plt.plot([i for i in range(len(df['model1_wa2'].values))], df['model1_wa2'], '.-')
#plt.plot([i for i in range(len(df['model2_wa2'].values))], df['model2_wa2'], '.-')

plt.plot([i for i in range(len(df['model1_fc2'].values))], df['model1_fc2']-df['model2_fc2'], '.-')


x=set(df[df['train_epoch'].notnull()]['number'].values)
x = [i for i in x if str(i) != 'nan']
print(x)
#plt.vlines(x, ymin=min(df['model1_fc2'].values), ymax=max(df['model1_fc2'].values),)
plt.show()
'''