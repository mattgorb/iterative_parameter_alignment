
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

import numpy as np                                    # v 1.19.2
import matplotlib.pyplot as plt                       # v 3.3.2
import matplotlib.patches as patches
from matplotlib.legend_handler import HandlerTuple

ds='cifar10'res
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
        #ax2 = ax1.twinx()
        for peer in list(np.unique(ipa['client_list'].values)):
            ax1.plot([i for i in range(len(ipa[ipa['client_list'] == peer]['test_accuracy_list'].values, ))],
                     ipa[ipa['client_list'] == peer]['test_accuracy_list'].values,linewidth=1.25, )

    ax1.plot([i for i in range(len(corrs))], corrs, '-', label='Model Correlation ', linewidth=2.5, color='tab:orange')









ncolors = 100
cmaps_names = ['Peer Models']
cmaps = [sns.color_palette("crest", n_colors=8, as_cmap=True )]
cmaps_gradients = [cmap(np.linspace(0, 1, ncolors)) for cmap in cmaps]
cmaps_dict = dict(zip(cmaps_names, cmaps_gradients))

# Create a list of lists of patches representing the gradient of each colormap
patches_cmaps_gradients = []
for cmap_name, cmap_colors in cmaps_dict.items():
    cmap_gradient = [patches.Patch(facecolor=c, edgecolor=c, label=cmap_name)
                     for c in cmap_colors]
    patches_cmaps_gradients.append(cmap_gradient)


handles, labels = ax1.get_legend_handles_labels()
print(handles)
print(labels)

patches_cmaps_gradients.append(handles[0])
cmaps_names.append(labels[0])

# Create custom legend (with a large fontsize to better illustrate the result)
plt.legend(handles=patches_cmaps_gradients, labels=cmaps_names, fontsize=16,
           handler_map={list: HandlerTuple(ndivide=None, pad=0)})

#plt.legend(loc='lower right')



ax1.set_xlabel('Communication Round', size=18)
ax1.set_ylabel('Test Accuracy/Correlation', size=18)
#ax2.set_ylabel('Correlation [-100,100]')
#ax2.set_ylabel('Correlation [-100,100]')


#ax1.set_yticks(np.linspace(ax1.get_ybound()[0], ax1.get_ybound()[1], 5))
#ax2.set_yticks(np.linspace(ax2.get_ybound()[0], ax2.get_ybound()[1], 5))


plt.ylim([35,max(max(corrs), 95)])
plt.xlim([0, 98])

plt.tight_layout()

plt.savefig(f'fairness_by_epoch_{ds}_pearson_{seed}.png')



#plt.savefig(f'fairness_by_epoch_{ds}_pearson_full.png')