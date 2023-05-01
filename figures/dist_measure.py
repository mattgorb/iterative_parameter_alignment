import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import pdist,squareform
import numpy as np
from matplotlib.colors import ListedColormap
import scipy
import math

'''measure='both_correct_pred'
measure='both_incorrect_pred'
measure='scores_hamming'
'''

measure='p1_weight_distance'
measure='p2_weight_distance'

measures=['both_correct_pred','both_incorrect_pred','scores_hamming','p1_weight_distance','p2_weight_distance',]
#measures=['scores_hamming','p1_weight_distance','p2_weight_distance',]
for measure in measures:
    dir=np.load(f'heatmap/fashion/model_MLP_ds_Fashion_MNIST_n_cli_10_ds_split_dirichlet_ds_alpha_0.3_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_True_le_1_s_False_rand_top_False_{measure}_iter_200.npy')

    iid=np.load(f'heatmap/fashion/model_MLP_ds_Fashion_MNIST_n_cli_10_ds_split_iid_ds_alpha_0.3_align_se_waf_1_delta_None_init_type_kaiming_normal_same_init_True_le_1_s_False_rand_top_False_{measure}_iter_100.npy')

    if measure in ['both_correct_pred', 'both_incorrect_pred']:
        np.fill_diagonal(dir, 0)
        np.fill_diagonal(iid, 0)


    print('\n\n')
    print(measure)
    print("IID")
    flattened_id_matrix=squareform(iid)
    #print("Length")
    #print(len(flattened_id_matrix))
    print('Average')
    print(np.mean(flattened_id_matrix))
    print('Standard Error')
    print(np.std(flattened_id_matrix)/math.sqrt(len(flattened_id_matrix)))

    print("DIR")
    flattened_id_matrix=squareform(dir)
    #print("Length")
    #print(len(flattened_id_matrix))
    print('Average')
    print(np.mean(flattened_id_matrix))
    print('Standard Error')
    print(np.std(flattened_id_matrix)/math.sqrt(len(flattened_id_matrix)))

'''



'''