
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pickle
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

sns.set_style('whitegrid')



ds='cifar10'
seeds=[3,4,6,9,18,24,32]

metric='test_losses'
#metric='test_accuracy_list'


seeds=[32]




for seed in seeds:
    #ipa=pd.read_csv(f'client_results/peer_contrib/{ds}/{seed}False.csv')
    #baseline=pd.read_csv(f'client_results/peer_contrib/{ds}/{seed}True.csv')
    ipa=pd.read_csv(f'client_results/full_new/{ds}/{seed}False.csv')
    baseline=pd.read_csv(f'client_results/full_new/{ds}/{seed}True.csv')

    baseline=baseline.groupby(['client_list'])[metric].mean()

    corrs=[]
    for epoch in np.unique(ipa['iter_list'].values):
        df=ipa[ipa['iter_list']==epoch]
        df=df.groupby(['client_list'])[metric].mean()

        corr, _ = pearsonr(df, baseline)
        print(f'Epoch {epoch} corr {corr}')
        corrs.append(corr*100)

    #total = ipa[(ipa['iter_list'] > 4) & (ipa['iter_list'] < 11)]
    #total=total.groupby(['client_list'])[metric].mean()
    #corr, _ = pearsonr(total, baseline)

    #print(f'Total: {corr}\n\n\n')




    with sns.color_palette("crest", n_colors=20):
        plt.clf()
        fig, ax1 = plt.subplots()

        for peer in list(np.unique(ipa['client_list'].values)):
            ax1.plot([i for i in range(len(ipa[ipa['client_list'] == peer]['test_accuracy_list'].values, ))],
                     ipa[ipa['client_list'] == peer]['test_accuracy_list'].values,linewidth=1.25, )

    ax1.plot([i for i in range(len(corrs))], corrs, '-', label='Correlation', linewidth=2.5, color='tab:orange')


plt.ylim([35,max(max(corrs), 95)])
#ax2.ylim([40,max(max(corrs), 95)])
plt.xlim([0, 98])
plt.legend(loc='lower right')


ax1.set_xlabel('Communication Round')
ax1.set_ylabel('Test Accuracy (%)')
ax2 = ax1.twinx()
ax2.set_ylabel('Correlation [-100,100]')

plt.tight_layout()

plt.savefig(f'fairness_by_epoch_{ds}_pearson_{seed}.png')



#plt.savefig(f'fairness_by_epoch_{ds}_pearson_full.png')