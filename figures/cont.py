"""
IMAGES: PEER contribution equation vs accuracy at Epoch K
Convergence Rate versus baseline
"""

import matplotlib.pyplot as plt
import math
import pickle
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
#Example
#x=[0.10,0.11,0.10 ,0.09, 0.09,0.11  ,  0.10 ,   0.10 ,   0.12 ,   0.08]
#np.uniformity=1-(np.linalg.norm(x)*math.sqrt(10)-1)/(math.sqrt(10)-1)

plt.clf()

ds = 'cifar10'
alpha=0.25

all_test_accs=[]
all_baseline_accs=[]

if ds=='cifar10':
    seeds=[4,6,15,16, 32]
else:
    seeds = [ 3,4, 6, 9 , 18, 24, 32]
    #seeds = [3,]
for seed in seeds:
    if ds=='cifar10':

        results=f'client_results/peer_contrib/cifar10/{seed}False.csv'
        standalone=f'client_results/peer_contrib/cifar10/{seed}True.csv'
        df_standalone = pd.read_csv(standalone)
    elif ds=='mnist':

        results=f'client_results/peer_contrib/mnist/{seed}False.csv'
        standalone=f'client_results/peer_contrib/mnist/{seed}True.csv'
        df_standalone = pd.read_csv(standalone)

    df=pd.read_csv(results)

    df_standalone = df_standalone.groupby(['client_list'])['test_losses'].mean()

    corrs=[]

    for iter in np.unique(df['iter_list'].values):


        df_temp=df[df['iter_list']==iter].sort_values(by=['client_list'])

        corr, _ = pearsonr(df_temp['test_losses'].values, df_standalone.values)
        corrs.append(corr)
    print(corrs)
    plt.plot( [i for i in range(len(corrs))], corrs, '-.')

    '''for client in np.unique(df['client_list'].values):
        df_temp = df[df['client_list'] == client].sort_values(by=['iter_list'])
        print(client)
        plt.plot([i for i in range(len(df_temp['iter_list'].values))], df_temp['test_accuracy_list'].values, '.')'''

#plt.ylim([95,100])
plt.savefig(f'fairness_by_epochs_{ds}.png')