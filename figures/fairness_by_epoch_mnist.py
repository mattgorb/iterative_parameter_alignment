import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pickle
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

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

import seaborn as sns
sns.set_style("whitegrid")


ds='mnist'

#baseline=pd.read_csv(f'client_results/peer_contrib/benchmarks/client_results_ds_MNIST_model_MLP_ds_MNIST_seed_1_n_cli_20_None_ds_split_classimbalance_ds_alpha_0.3_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_False_le_25_s_True_rand_top_False.csv')
ipa=pd.read_csv(f'client_results/peer_contrib/benchmarks/client_results_ds_MNIST_model_MLP_ds_MNIST_seed_1_n_cli_20_None_ds_split_classimbalance_ds_alpha_0.3_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_False_le_5_s_False_rand_top_False.csv')

#baseline=pd.read_csv(f'client_results/peer_contrib/benchmarks/client_results_ds_CIFAR10_model_Conv4_ds_CIFAR10_seed_1_n_cli_10_None_ds_split_classimbalance_ds_alpha_0.3_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_False_le_50_s_True_rand_top_True.csv')
#ipa=pd.read_csv(f'client_results/peer_contrib/benchmarks/client_results_ds_CIFAR10_model_Conv4_ds_CIFAR10_seed_1_n_cli_10_None_ds_split_classimbalance_ds_alpha_0.3_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_False_le_5_s_False_rand_top_True.csv')

with sns.color_palette("crest", n_colors=10):
    corrs=[]
    for client in np.unique(ipa['client_list'].values):
        df=ipa[ipa['client_list']==client]
        #df=df.groupby(['client_list'])[metric].mean()
        print(df['test_accuracy_list'].values)
        #sys.exit()
        #plt.scatter(t, y, c='Blues', marker='.')
        #plt.plot([i for i in range(len(df['test_accuracy_list'].values))][5:30], df['test_accuracy_list'].values[5:30], '-.', cmap="Blues")



        #plt.plot(np.random.rand(5, 10))
        #plt.scatter([i for i in range(len(df['test_accuracy_list'].values))][6:30], df['test_accuracy_list'].values[6:30],)#  c=df["model_num"], cmap="Blues", marker='.')
        #with sns.color_palette("Blues", n_colors=20):
        plt.plot([i for i in range(len(df['test_accuracy_list'].values))], df['test_accuracy_list'].values, '-')


plt.title('CIFAR-10, Convergence Rates of Peers', size=18)

plt.xlim([5,30])
plt.ylabel("Test Accuracy (%)" ,size=16)
plt.xlabel("Communication Round",size=16)












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


plt.ylim([75,95])
# Create custom legend (with a large fontsize to better illustrate the result)
plt.legend(handles=patches_cmaps_gradients, labels=cmaps_names, fontsize=16,
           handler_map={list: HandlerTuple(ndivide=None, pad=0)}, loc='lower right')






plt.tight_layout()

plt.savefig(f'fairness_by_epoch_{ds}_basic.pdf')