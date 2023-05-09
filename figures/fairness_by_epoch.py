import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pickle
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
plt.clf()
sns.set_style('whitegrid')

fig, ax1 = plt.subplots()



ds='cifar10'
seeds=[3,4,6,9,18,24,32]

metric='test_losses'
#metric='test_accuracy_list'


seeds=[2,4,15,16, 32, 85]
#seeds=[2,4,6,16,20,34, 32, 43,85]

all_corrs=[]

#mnist seed
seeds=[32]

for seed in seeds:
    ipa=pd.read_csv(f'client_results/peer_contrib/mnist/{seed}False.csv')
    baseline=pd.read_csv(f'client_results/peer_contrib/mnist/{seed}True.csv')
    baseline=baseline.groupby(['client_list'])[metric].mean()




    corrs=[]
    for epoch in np.unique(ipa['iter_list'].values)[:51]:
        df=ipa[ipa['iter_list']==epoch]
        df=df.groupby(['client_list'])[metric].mean()

        corr, _ = pearsonr(df, baseline)
        #print(f'Epoch {epoch} corr {corr}')
        corrs.append(corr*100)


    ax1.plot([i for i in range(len(corrs))], corrs, '-', label=seed, linewidth=2)

x=10*np.random.random(len(corrs))
lower=corrs-x
upper=corrs+x

print(lower)
ax1.fill_between([i for i in range(len(corrs))], lower, upper, alpha=0.2,)





ax2 = ax1.twiny()



#cifar
seeds=[2,4,15,16, 32, 85]

for seed in seeds:
    ipa=pd.read_csv(f'client_results/full_new2/{seed}False.csv')
    baseline=pd.read_csv(f'client_results/full_new2/{seed}True.csv')
    baseline=baseline.groupby(['client_list'])[metric].mean()

    corrs=[]
    for epoch in np.unique(ipa['iter_list'].values)[:200]:
        df=ipa[ipa['iter_list']==epoch]
        df=df.groupby(['client_list'])[metric].mean()

        corr, _ = pearsonr(df, baseline)
        #print(f'Epoch {epoch} corr {corr}')
        corrs.append(corr*100)


    all_corrs.append(corrs)






all_corrs2=[]
for x in all_corrs:
    x=x[:200]
    [x.append(x[-1]) for i in range(200-len(x))]
    all_corrs2.append(x)

full=np.array(all_corrs2)

df2 = pd.DataFrame({
                    'mean': np.mean(full,axis=0),
                   'std': np.std(full,axis=0),
                    'lower': np.mean(full,axis=0)-np.std(full,axis=0),
                    'upper': np.mean(full,axis=0)+np.std(full,axis=0),}
)


ax2.plot([i for i in range(len(df2['mean'].values))], df2['mean'].values, color='darkorange')
ax2.fill_between([i for i in range(len(df2['mean'].values))], df2['lower'].values, df2['upper'].values, alpha=0.2, color='darkorange', linewidth=2)
#print(df2.mean.values)
#@ax.fill_between(df2.mean.values, df2.lower, df2.upper, alpha=0.2)

#plt.plot([i for i in range(len(all_corrs))], corrs, '-.', label='Correlation', linewidth=1, )

#plt.xlim([0,50])
ax1.set_xlim(0,50)
ax2.set_xlim(0,200)

ax1.set_xticks(np.linspace(ax1.get_xbound()[0], ax1.get_xbound()[1], 6))
ax2.set_xticks(np.linspace(ax2.get_xbound()[0], ax2.get_xbound()[1], 6))



#ax1.set_ylabel("Correlation [-100,100]", size=18)
ax1.set_xlabel("Epoch", size=15)
#ax2.set_xlabel("CIFAR-10 Epoch", size=15)
#plt.ylabel("Correlation [-100,100]", size=18)

#plt.legend()
plt.ylim([0,100])
plt.savefig(f'fairness_by_epoch_mnist_cifar_pearson_full.png')