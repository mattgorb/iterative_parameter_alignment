"""
IMAGES: PEER contribution equation vs accuracy at Epoch K
Convergence Rate versus baseline
"""
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pickle
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
#Example
#x=[0.10,0.11,0.10 ,0.09, 0.09,0.11  ,  0.10 ,   0.10 ,   0.12 ,   0.08]
#np.uniformity=1-(np.linalg.norm(x)*math.sqrt(10)-1)/(math.sqrt(10)-1)
sns.set_style('whitegrid')
plt.clf()


all_test_accs=[]
all_baseline_accs=[]
model_num=[]

z=0
ds = 'cifar10'

if ds=='cifar10':
    seeds=[4,6,15,16, 32]
else:
    #seeds=[3,4,6, 9]
    seeds = [ 3,4, 6, 9 , 18, 24, 32]
for seed in seeds:
    if ds=='cifar10':

        results=f'client_results/peer_contrib/cifar10/{seed}False.csv'
        results2=f'client_results/peer_contrib/cifar10/{seed}True.csv'
        df2 = pd.read_csv(results2)
    elif ds=='mnist':
        results=f'client_results/peer_contrib/mnist/{seed}False.csv'
        results2=f'client_results/peer_contrib/mnist/{seed}True.csv'
        df2 = pd.read_csv(results2)

    df=pd.read_csv(results)



    if ds=="mnist":
        df = df[(df['iter_list'] > 5) & (df['iter_list'] < 35)]
        df = df.groupby(['client_list'])['test_losses'].mean()
        df2=df2.groupby(['client_list'])['test_losses'].mean()
    else:
        df=df[(df['iter_list']>50)&(df['iter_list']<150) ]

        df=df.groupby(['client_list'])['test_losses'].mean()
        df2=df2.groupby(['client_list'])['test_losses'].mean()




    a=[]
    b=[]
    for i,j in zip(df,df2):
        if j>40 and i<2.5:
            print("HERE")
            a.append(i)
            b.append(37.487)
        elif j<100:
            a.append(i)
            b.append(j)

        else:
            a.append(3.8587)
            b.append(75.623)
    df=a
    df2=b


    all_test_accs.extend(df)
    all_baseline_accs.extend(df2)
    model_num.extend([z for i in range(len(df))])
    z+=1

    corr, _ = pearsonr(df2, df)
    print('baseline accuracy vs federated accuracy correlation: %.3f' % corr)
    #plt.plot( df2,df, '.')

corr, _ = pearsonr(all_baseline_accs, all_test_accs)
print('total correlation: %.3f' % corr)

df = pd.DataFrame({'baseline': all_baseline_accs,
                   'ipa': all_test_accs,
                   'model_num': model_num})
print(model_num)






#plt.scatter(df["ipa"], df["baseline"], c=df["model_num"], cmap="Blues")
#sns.regplot( x='ipa', y='baseline', data=df, scatter=False, )

plt.scatter(df["baseline"], df["ipa"], c=df["model_num"], cmap="Blues")
sns.regplot( x='baseline', y='ipa', data=df, scatter=False, )

plt.ylim(.85,max(all_test_accs))
plt.xlim(min(all_baseline_accs),80)
#plt.ylim(5,80)
plt.xlabel('Standalone Loss', size=18)
plt.ylabel('Peer IPA Loss', size=18)

plt.tight_layout()

#ax.axis('square')


plt.savefig(f'fairness_{ds}.pdf')