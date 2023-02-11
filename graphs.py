from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


'''df=pd.read_csv('~/Downloads/weight_alignment_csvs/1layer_model_stats_ae_model1.csv')
df2=pd.read_csv('~/Downloads/weight_alignment_csvs/1layer_model_stats_ae_model2.csv')

df3=pd.read_csv('~/Downloads/weight_alignment_csvs/1layer_model_stats_se_model1.csv')
df4=pd.read_csv('~/Downloads/weight_alignment_csvs/1layer_model_stats_se_model2.csv')

plt.clf()
plt.plot([i for i in range(df.shape[0])], df['model1_trainer.test_accuracy_list'].values, '.', label='Model 1 AE')
plt.plot([i for i in range(df2.shape[0])], df2['model2_trainer.test_accuracy_list'].values, '.', label='Model 2 AE')
plt.plot([i for i in range(df3.shape[0])], df3['model1_trainer.test_accuracy_list'].values, '.', label='Model 1 SE')
plt.plot([i for i in range(df4.shape[0])], df4['model2_trainer.test_accuracy_list'].values, '.', label='Model 2 SE')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('1 Layer Linear NN')
plt.savefig('figures/1layer_test_accuracy.pdf')'''



df=pd.read_csv('~/Downloads/weight_alignment_csvs/1layer_weight_diff_ae_model1.csv')
df2=pd.read_csv('~/Downloads/weight_alignment_csvs/1layer_weight_diff_ae_model2.csv')

df=df[df['merge_iter'].isin([2,3,4,5,6,7])]
df2=df2[df2['merge_iter'].isin([2,3,4,5,6,7])]

plt.clf()
plt.plot([i for i in range(df.shape[0])], df['model1_weight_align_loss_list'].values, '.', label='Model 1 AE')
plt.plot([i for i in range(df2.shape[0])], df2['model1_weight_align_loss_list'].values, '.', label='Model 2 AE')
plt.xlabel('Epoch')
plt.ylabel('Absolute Error, Weight-Weight_Align')
plt.legend()
plt.savefig('figures/1layer_weight_diff_ae.pdf')



df3=pd.read_csv('~/Downloads/weight_alignment_csvs/1layer_weight_diff_se_model1.csv')
df4=pd.read_csv('~/Downloads/weight_alignment_csvs/1layer_weight_diff_se_model2.csv')
df3=df3[df3['merge_iter'].isin([2,3,4,5,6,7])]
df4=df4[df4['merge_iter'].isin([2,3,4,5,6,7])]
plt.clf()
plt.plot([i for i in range(df3.shape[0])], df3['model1_weight_align_loss_list'].values, '.', label='Model 1 SE')
plt.plot([i for i in range(df4.shape[0])], df4['model1_weight_align_loss_list'].values, '.', label='Model 2 SE')
plt.xlabel('Epoch')
plt.ylabel('Squared Error, Weight-Weight_Align')
plt.legend()

plt.savefig('figures/1layer_weight_diff_se.pdf')