from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


'''
df=pd.read_csv('~/Downloads/weight_alignment_csvs/1layer_model_stats_ae.csv')
df2=pd.read_csv('~/Downloads/weight_alignment_csvs/1layer_model_stats_ae.csv')

df3=pd.read_csv('~/Downloads/weight_alignment_csvs/1layer_model_stats_se.csv')
df4=pd.read_csv('~/Downloads/weight_alignment_csvs/1layer_model_stats_se.csv')

plt.clf()
plt.plot([i for i in range(df.shape[0])][:1300], df['model1_trainer.test_accuracy_list'].values[:1300], '.', label='Model 1 AE')
plt.plot([i for i in range(df2.shape[0])][:1300], df2['model2_trainer.test_accuracy_list'].values[:1300], '.', label='Model 2 AE')
plt.plot([i for i in range(df3.shape[0])], df3['model1_trainer.test_accuracy_list'].values, '.', label='Model 1 SE')
plt.plot([i for i in range(df4.shape[0])], df4['model2_trainer.test_accuracy_list'].values, '.', label='Model 2 SE')

plt.axhline(92.8, color='gray', label='Baseline')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('1 Layer Linear NN')
plt.savefig('figures/1layer_test_accuracy.pdf')
'''




'''
df=pd.read_csv('~/Downloads/weight_alignment_csvs/1layer_weight_diff_ae_model1.csv')
df=df[df['merge_iter'].isin([500,501,502,503,504,505])]
epochs=[list(df['merge_iter'].values).index(i) for i in [500,501,502,503,504,505]]
plt.clf()
plt.plot([i for i in range(df.shape[0])], df['model1_weight_align_ae_loss_list'].values, '.', label='Model 1 AE')
plt.xticks([])
i=2
for epoch in epochs:
    plt.axvline(epoch, color='gray')
    plt.text(epoch, 0, i,rotation=90)
    i+=1
plt.xlabel('Epoch')
plt.ylabel('Absolute Error, Weight-Weight_Align')
plt.title('Epochs 500 to 505')
plt.legend()
plt.savefig('figures/1layer_weight_diff_model1_ae_500_505.pdf')



df=pd.read_csv('~/Downloads/weight_alignment_csvs/1layer_weight_diff_ae_model2.csv')
df=df[df['merge_iter'].isin([500,501,502,503,504,505])]
epochs=[list(df['merge_iter'].values).index(i) for i in [500,501,502,503,504,505]]
plt.clf()
plt.plot([i for i in range(df.shape[0])], df['model2_weight_align_ae_loss_list'].values, '.', label='Model 2 AE')
plt.xticks([])
i=2
for epoch in epochs:
    plt.axvline(epoch, color='gray')
    plt.text(epoch, 0, i,rotation=90)
    i+=1
plt.xlabel('Epoch')
plt.ylabel('Absolute Error, Weight-Weight_Align')
plt.legend()
plt.title('Epochs 500 to 505')
plt.savefig('figures/1layer_weight_diff_model2_ae_500_505.pdf')




df=pd.read_csv('~/Downloads/weight_alignment_csvs/1layer_weight_diff_se_model1.csv')
df=df[df['merge_iter'].isin([500,501,502,503,504,505])]
epochs=[list(df['merge_iter'].values).index(i) for i in [500,501,502,503,504,505]]
plt.clf()
plt.plot([i for i in range(df.shape[0])], df['model1_weight_align_se_loss_list'].values, '.', label='Model 1 SE')
plt.xticks([])
i=2
for epoch in epochs:
    plt.axvline(epoch, color='gray')
    x_bounds = plt.ylim()[0]
    plt.text(epoch, x_bounds-.075, i,)
    i+=1
plt.xlabel('Epoch')
plt.ylabel('Squared Error, (Weight-Weight_Align)**2')
plt.legend()
plt.title('Epochs 500 to 505')
plt.savefig('figures/1layer_weight_diff_model1_se_500_505.pdf')



df=pd.read_csv('~/Downloads/weight_alignment_csvs/1layer_weight_diff_se_model2.csv')
df=df[df['merge_iter'].isin([500,501,502,503,504,505])]
epochs=[list(df['merge_iter'].values).index(i) for i in [500,501,502,503,504,505]]
plt.clf()
plt.plot([i for i in range(df.shape[0])], df['model2_weight_align_se_loss_list'].values, '.', label='Model 2 SE')


plt.xticks([])
i=2
for epoch in epochs:
    plt.axvline(epoch, color='gray')
    x_bounds = plt.ylim()[0]
    plt.text(epoch, x_bounds-.075, i,)
    i+=1
plt.xlabel('Epoch')
plt.ylabel('Squared Error, (Weight-Weight_Align)**2')

plt.title('Epochs 500 to 505')

plt.legend()
plt.savefig('figures/1layer_weight_diff_model2_se_500_505.pdf')

'''



'''
df=pd.read_csv('~/Downloads/weight_alignment_csvs/mlp_detach_model_stats_ae.csv')
df2=pd.read_csv('~/Downloads/weight_alignment_csvs/mlp_detach_model_stats_ae.csv')

df3=pd.read_csv('~/Downloads/weight_alignment_csvs/mlp_detach_model_stats_se.csv')
df4=pd.read_csv('~/Downloads/weight_alignment_csvs/mlp_detach_model_stats_se.csv')

plt.clf()
plt.plot([i for i in range(df.shape[0])], df['model1_trainer.test_accuracy_list'].values, '.', label='Model 1 AE')
plt.plot([i for i in range(df2.shape[0])], df2['model2_trainer.test_accuracy_list'].values, '.', label='Model 2 AE')
plt.plot([i for i in range(df3.shape[0])], df3['model1_trainer.test_accuracy_list'].values, '.', label='Model 1 SE')
plt.plot([i for i in range(df4.shape[0])], df4['model2_trainer.test_accuracy_list'].values, '.', label='Model 2 SE')

plt.axhline(98.3, color='gray', label='Baseline')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('MLP NN')
plt.savefig('figures/mlp_test_accuracy.pdf')

'''

'''
#epoch_list=[500,501,502,503,504,505]
#title='Epoch 500 to 505'
#fig_name='_500_505'
epoch_list=[2,3,4,5,6,7]
title='Epoch 2 to 7'
fig_name='_2_7'

df=pd.read_csv('~/Downloads/weight_alignment_csvs/mlp_detach_weight_diff_ae_model1.csv')
df=df[df['merge_iter'].isin(epoch_list)]
epochs=[list(df['merge_iter'].values).index(i) for i in epoch_list]
plt.clf()
plt.plot([i for i in range(df.shape[0])], df['model1_weight_align_ae_loss_list'].values, '.', label='Model 1 AE')
plt.xticks([])
i=2
for epoch in epochs:
    plt.axvline(epoch, color='gray')
    plt.text(epoch, 0, i,rotation=90)
    i+=1
plt.xlabel('Epoch')
plt.ylabel('Absolute Error, Weight-Weight_Align')
plt.title(title)
plt.legend()
plt.savefig(f'figures/mlp_weight_diff_model1_ae{fig_name}.pdf')



df=pd.read_csv('~/Downloads/weight_alignment_csvs/mlp_detach_weight_diff_ae_model2.csv')
df=df[df['merge_iter'].isin(epoch_list)]
epochs=[list(df['merge_iter'].values).index(i) for i in epoch_list]
plt.clf()
plt.plot([i for i in range(df.shape[0])], df['model2_weight_align_ae_loss_list'].values, '.', label='Model 2 AE')
plt.xticks([])
i=2
for epoch in epochs:
    plt.axvline(epoch, color='gray')
    plt.text(epoch, 0, i,rotation=90)
    i+=1
plt.xlabel('Epoch')
plt.ylabel('Absolute Error, Weight-Weight_Align')
plt.legend()
plt.title(title)
plt.savefig(f'figures/mlp_weight_diff_model2_ae{fig_name}.pdf')




df=pd.read_csv('~/Downloads/weight_alignment_csvs/mlp_detach_weight_diff_se_model1.csv')
df=df[df['merge_iter'].isin(epoch_list)]
epochs=[list(df['merge_iter'].values).index(i) for i in epoch_list]
plt.clf()
plt.plot([i for i in range(df.shape[0])], df['model1_weight_align_se_loss_list'].values, '.', label='Model 1 SE')
plt.xticks([])
i=2
for epoch in epochs:
    plt.axvline(epoch, color='gray')
    #x_bounds = plt.ylim()[0]
    #plt.text(epoch, x_bounds-.075, i,)
    i+=1
plt.xlabel('Epoch')
plt.ylabel('Squared Error, (Weight-Weight_Align)**2')
plt.legend()
plt.title(title)
plt.savefig(f'figures/mlp_weight_diff_model1_se{fig_name}.pdf')



df=pd.read_csv('~/Downloads/weight_alignment_csvs/mlp_detach_weight_diff_se_model2.csv')
df=df[df['merge_iter'].isin(epoch_list)]
epochs=[list(df['merge_iter'].values).index(i) for i in epoch_list]
plt.clf()
plt.plot([i for i in range(df.shape[0])], df['model2_weight_align_se_loss_list'].values, '.', label='Model 2 SE')


plt.xticks([])
i=2
for epoch in epochs:
    plt.axvline(epoch, color='gray')
    #x_bounds = plt.ylim()[0]
    #plt.text(epoch, x_bounds-.075, i,)
    i+=1
plt.xlabel('Epoch')
plt.ylabel('Squared Error, (Weight-Weight_Align)**2')

plt.title(title)

plt.legend()
plt.savefig(f'figures/mlp_weight_diff_model2_se{fig_name}.pdf')

'''




























model_type='mlp_detach'



df=pd.read_csv(f'~/Downloads/weight_alignment_csvs/{model_type}_model_stats_ae_homogenous.csv')
df2=pd.read_csv(f'~/Downloads/weight_alignment_csvs/{model_type}_model_stats_ae_homogenous.csv')

df3=pd.read_csv(f'~/Downloads/weight_alignment_csvs/{model_type}_model_stats_se_homogenous.csv')
df4=pd.read_csv(f'~/Downloads/weight_alignment_csvs/{model_type}_model_stats_se_homogenous.csv')

plt.clf()
plt.plot([i for i in range(df.shape[0])], df['model1_trainer.test_accuracy_list'].values, '.', label='Model 1 AE')
plt.plot([i for i in range(df2.shape[0])], df2['model2_trainer.test_accuracy_list'].values, '.', label='Model 2 AE')
plt.plot([i for i in range(df3.shape[0])], df3['model1_trainer.test_accuracy_list'].values, '.', label='Model 1 SE')
plt.plot([i for i in range(df4.shape[0])], df4['model2_trainer.test_accuracy_list'].values, '.', label='Model 2 SE')

plt.axhline(98.3, color='gray', label='Baseline')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('MLP NN')
plt.savefig(f'figures/{model_type}_test_accuracy_homogenous.pdf')


#epoch_list=[500,501,502,503,504,505]
#title='Epoch 500 to 505'
#fig_name='_500_505'
epoch_list=[2,3,4,5,6,7]
title='Epoch 2 to 7'
fig_name='_2_7'

df=pd.read_csv(f'~/Downloads/weight_alignment_csvs/{model_type}_weight_diff_ae_model1_homogenous.csv')
df=df[df['merge_iter'].isin(epoch_list)]
epochs=[list(df['merge_iter'].values).index(i) for i in epoch_list]
plt.clf()
plt.plot([i for i in range(df.shape[0])], df['model1_weight_align_ae_loss_list'].values, '.', label='Model 1 AE')
plt.xticks([])
i=2
for epoch in epochs:
    plt.axvline(epoch, color='gray')
    plt.text(epoch, 0, i,rotation=90)
    i+=1
plt.xlabel('Epoch')
plt.ylabel('Absolute Error, Weight-Weight_Align')
plt.title(title)
plt.legend()
plt.savefig(f'figures/{model_type}_weight_diff_model1_ae{fig_name}_homogenous.pdf')


df=pd.read_csv(f'~/Downloads/weight_alignment_csvs/{model_type}_weight_diff_se_model1_homogenous.csv')
df=df[df['merge_iter'].isin(epoch_list)]
epochs=[list(df['merge_iter'].values).index(i) for i in epoch_list]
plt.clf()
plt.plot([i for i in range(df.shape[0])], df['model1_weight_align_se_loss_list'].values, '.', label='Model 1 SE')
plt.xticks([])
i=2
for epoch in epochs:
    plt.axvline(epoch, color='gray')
    plt.text(epoch, 0, i,rotation=90)
    i+=1
plt.xlabel('Epoch')
plt.ylabel('Squared Error, (Weight-Weight_Align)^2')
plt.title(title)
plt.legend()
plt.savefig(f'figures/{model_type}_weight_diff_model1_se{fig_name}_homogenous.pdf')


