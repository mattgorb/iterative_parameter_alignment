import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import pdist,squareform
import numpy as np
from matplotlib.colors import ListedColormap
import scipy

'''bottom=np.load('/Users/matthewgorbett/PycharmProjects/iterative_weight_alignment/figures/heatmap/'
          'model_Conv4_n_cli_10_ds_split_dirichlet_ds_alpha_0.3_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_False_le_5_s_False_scores_hamming_iter_190.npy')

top=np.load('/Users/matthewgorbett/PycharmProjects/iterative_weight_alignment/figures/heatmap/'
          'model_Conv4_n_cli_10_ds_split_iid_ds_alpha_0.3_align_se_waf_1_delta_None_init_type_kaiming_normal_same_init_False_le_5_s_False_scores_hamming_iter_200.npy')
'''
bottom=np.load('heatmap/'
          'model_Conv4_n_cli_10_ds_split_dirichlet_ds_alpha_0.3_align_ae_waf_1_delta_None_init_type_kaiming_normal_same_init_False_le_5_s_False_scores_hamming_iter_180.npy')

top=np.load('heatmap/'
          'model_Conv4_n_cli_10_ds_split_iid_ds_alpha_0.3_align_se_waf_1_delta_None_init_type_kaiming_normal_same_init_False_le_5_s_False_scores_hamming_iter_50.npy')


print(bottom)
print(top)

#result=np.array(np.tril(bottom)+np.triu(top))
#result=bottom

f, ax = plt.subplots(figsize=(12, 12))
ax.set_aspect("equal")
cmap=sns.color_palette('crest',)
#cmap=sns.color_palette('Blues',)

cmap=sns.blend_palette(cmap, n_colors=25)

#cmap=sns.light_palette("seagreen", as_cmap=True)
cmap=sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=False, as_cmap=True)
#cmap=sns.color_palette("light:b", as_cmap=True)

#print(len(cmap))
cmap="YlGn"
#cmap='Greens'


cmap=sns.color_palette('Blues',)
cmap=sns.blend_palette(cmap, n_colors=16)

'''
IID
Prediction Hamming Distances
[[0, 1583, 1614, 1855, 1871, 1318, 1837, 1569, 1385, 1600],
[1583, 0, 950, 1415, 1496, 1491, 1371, 1222, 1186, 1377], 
[1614, 950, 0, 1372, 1508, 1455, 1384, 1210, 1220, 1345], 
[1855, 1415, 1372, 0, 1453, 1741, 1378, 1163, 1401, 1407],
[1871, 1496, 1508, 1453, 0, 1796, 1549, 1419, 1462, 1581],
[1318, 1491, 1455, 1741, 1796, 0, 1668, 1485, 1288, 1516],
[1837, 1371, 1384, 1378, 1549, 1668, 0, 1267, 1302, 1319], 
[1569, 1222, 1210, 1163, 1419, 1485, 1267, 0, 1130, 1246],
[1385, 1186, 1220, 1401, 1462, 1288, 1302, 1130, 0, 1083],
[1600, 1377, 1345, 1407, 1581, 1516, 1319, 1246, 1083, 0]]
'''

'''
HETERO
Prediction Hamming Distances
[[0, 1583, 1614, 1855, 1871, 1318, 1837, 1569, 1385, 1600],
[1583, 0, 950, 1415, 1496, 1491, 1371, 1222, 1186, 1377], 
[1614, 950, 0, 1372, 1508, 1455, 1384, 1210, 1220, 1345], 
[1855, 1415, 1372, 0, 1453, 1741, 1378, 1163, 1401, 1407],
[1871, 1496, 1508, 1453, 0, 1796, 1549, 1419, 1462, 1581],
[1318, 1491, 1455, 1741, 1796, 0, 1668, 1485, 1288, 1516],
[1837, 1371, 1384, 1378, 1549, 1668, 0, 1267, 1302, 1319], 
[1569, 1222, 1210, 1163, 1419, 1485, 1267, 0, 1130, 1246],
[1385, 1186, 1220, 1401, 1462, 1288, 1302, 1130, 0, 1083],
[1600, 1377, 1345, 1407, 1581, 1516, 1319, 1246, 1083, 0]]
'''

#cmap_args={'use_gridspec':True,"shrink": .72, "location":"bottom",'pad':0.07,'anchor':(0.15, 1.0)}

print(np.triu(top)+np.tril(bottom))
#sys.exit()

A=sns.heatmap(np.triu(top)+np.tril(bottom),  cmap=cmap,linewidths=1,square=False, cbar_kws={
 'use_gridspec':True,"shrink": .6,"orientation": "vertical" }, )
A.set_yticklabels(A.get_yticks(), size = 15)

#sns.heatmap(np.tril(bottom),   cmap=cmap,mask=np.triu(top),linewidths=1, square=False,cbar=False)
#sns.heatmap(np.triu(top),  cmap=cmap,mask=np.tril(bottom),linewidths=1,square=False, cbar_kws={
 #'use_gridspec':True,"shrink": .6,"orientation": "vertical" }, )


ax.figure.axes[-1].set_ylabel('Mismatching Predictions',labelpad=15)
ax.figure.axes[-1].yaxis.label.set_size(30)
#.set_ylabel('Mean Absolute Error', rotation=270, fontsize = 15, labelpad=15)
ax.figure.axes[-1].tick_params(labelsize=25)

ax.set_title('IID', fontsize=40)
ax.set_xlabel('Peer Model', fontsize=30)
ax.set_ylabel('Dir(0.3)', fontsize=40)
plt.tight_layout()
plt.savefig('/Users/matthewgorbett/PycharmProjects/iterative_weight_alignment/figures/heatmap1.pdf', bbox_inches='tight')