from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
'''
df=pd.read_csv('norms/weight_diff_double.csv')
#df=df.head(200)
plt.clf()
#plt.plot([i for i in range(len(df['weight_diff_layer1'].values))], df['weight_diff_layer1'], '.-')
plt.plot([i for i in range(len(df['weight_diff_layer2'].values))], df['weight_diff_layer2'], '.-')
#x=set(df['epoch'].values)
#x = [i for i in x if str(i) != 'nan']
#print(x)
#plt.vlines(x, ymin=min(df['weight_diff_layer2'].values), ymax=max(df['weight_diff_layer2'].values),)
plt.show()
'''

'''
df=pd.read_csv('norms/norms_double.csv')
#df=df[0:60]
df=df[6*100:6*100+60]
#df=df.head(100)
plt.clf()
plt.plot([i for i in range(len(df['model1_fc2'].values))], df['model1_fc2'], '.-')
plt.plot([i for i in range(len(df['model2_fc2'].values))], df['model2_fc2'], '.-')
plt.plot([i for i in range(len(df['model1_wa2'].values))], df['model1_wa2'], '.-')
plt.plot([i for i in range(len(df['model2_wa2'].values))], df['model2_wa2'], '.-')

#plt.plot([i for i in range(len(df['model1_fc2'].values))], df['model1_fc2']-df['model2_fc2'], '.-')
#plt.xticks(plt.xticks()[::6])
#p
#x=set(df[df['train_epoch'].notnull()]['number'].values)
#x = [i for i in x if str(i) != 'nan']
#print(x)
plt.vlines([i*6 for i in range(11)], ymin=min(df['model1_fc2'].values), ymax=max(df['model1_fc2'].values),)

plt.show()
'''



''''''
df=pd.read_csv('norms/weight_diff_single.csv')
#df=df.head(200)
plt.clf()
#plt.plot([i for i in range(len(df['weight_diff_layer1'].values))], df['weight_diff_layer1'], '.-')
plt.plot([i for i in range(len(df['weight_diff_layer2'].values))], df['weight_diff_layer2'], '.-')
#x=set(df['epoch'].values)
#x = [i for i in x if str(i) != 'nan']
#print(x)
plt.xlabel('epoch')
plt.ylabel('weight diff ')
#plt.vlines(x, ymin=min(df['weight_diff_layer1'].values), ymax=max(df['weight_diff_layer1'].values),)
plt.show()



'''
df=pd.read_csv('norms/norms_single.csv')
#df=df[100:150]
df=df[0:151]
#df=df.head(100)
print(df.shape)
plt.clf()
plt.plot([i for i in range(len(df['model1_fc1'].values))], df['model1_fc1'], '.-', label='model 1')
plt.plot([i for i in range(len(df['model2_fc1'].values))], df['model2_fc1'], '.-', label='model 2')
plt.plot([i for i in range(len(df['model2_wa1'].values))], df['model2_wa1'], '.-', label='model 2 weight_align')

plt.vlines([i*6 for i in range(26)], ymin=min(df['model1_fc1'].values), ymax=max(df['model1_fc1'].values),
           colors='gray', ls=':', lw=1,)

z=np.arange(0, 151, 6)
#my_ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#my_ticks=[x+16 for x in my_ticks]
my_ticks=np.arange(0, 26, 1)
plt.xticks(z,my_ticks)
plt.xlabel('epoch')
plt.ylabel('layer 1 norm')
plt.title('layer 1 norm, single merge')
plt.legend()
plt.savefig('images/single_norms_layer1_epoch25.png')
#plt.show()
'''
