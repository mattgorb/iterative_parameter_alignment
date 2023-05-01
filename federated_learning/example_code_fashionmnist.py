from utils_general import *
from utils_methods import *
from utils_weight_alignment import *

# Dataset initialization
data_path = '/s/luffy/b/nobackup/mgorb/'  # The folder to save Data & Model

########
# For 'CIFAR100' experiments
#     - Change the dataset argument from CIFAR10 to CIFAR100.
########
# For 'mnist' experiments
#     - Change the dataset argument from CIFAR10 to mnist.
########
# For 'emnist' experiments
#     - Download emnist dataset from (https://www.nist.gov/itl/products-and-services/emnist-dataset) as matlab format and unzip it in data_path + "Data/Raw/" folder.
#     - Change the dataset argument from CIFAR10 to emnist.
########
# For Shakespeare experiments
# First generate dataset using LEAF Framework and set storage_path to the data folder
# storage_path = 'LEAF/shakespeare/data/'
#     - In IID use

# name = 'shakepeare'
# data_obj = ShakespeareObjectCrop(storage_path, name, crop_amount = 2000)

#      - In non-IID use
# name = 'shakepeare_nonIID'
# data_obj = ShakespeareObjectCrop_noniid(storage_path, name, crop_amount = 2000)
#########


# Generate IID or Dirichlet distribution
# IID
# n_client = 20
# data_obj = DatasetObject(dataset='mnist', n_client=n_client, seed=23, rule='iid', unbalanced_sgm=0, data_path=data_path)
epoch = 1
# rule='split_label'


'''
rule = 'split_label'
rule_arg = 0.3
n_client = 2
optim = 'SGD'
'''


'''
rule='Drichlet'
rule_arg=0.6
n_client = 20
optim='SGD'
'''

'''
rule='iid'
rule_arg=0.3
n_client = 20
optim='SGD'
'''

'''
rule='Drichlet'
rule_arg=0.3
n_client = 20
optim='SGD'
'''


'''

rule='split_label3'
rule_arg=0.3
n_client = 5
optim='SGD'
'''


# IF Adam, import utils_general_adam in utils_methods file
'''
rule='Drichlet'
rule_arg=0.3
n_client = 20
optim='SGD'
'''

''''''
rule='split_label3'
rule_arg=0.3
n_client = 2
optim='SGD'


'''
rule='iid'
rule_arg=0.3
n_client = 20
optim='SGD'
'''

'''
rule='iid'
rule_arg=0.3
n_client = 2
optim='SGD'
'''

# Dirichlet (0.6)
# data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=20, unbalanced_sgm=0, rule='Drichlet', rule_arg=0.6, data_path=data_path)
data_obj = DatasetObject(dataset='Fashion_MNIST', n_client=n_client, seed=20, unbalanced_sgm=0, rule=rule, rule_arg=rule_arg,
                         data_path=data_path)

model_name = 'fashionmnist_2NN'  # Model type

###
# Common hyperparameters
if rule == 'split_label' or rule == 'split_label5':
    com_amount = 3000
else:
    com_amount = 1000

save_period = 100
weight_decay = 1e-3
batch_size = 50
act_prob = 1
# act_prob = 0.15
suffix = f'{model_name}_n_cli_{n_client}_rule_{rule}_rule_arg_{rule_arg}_{optim}'
lr_decay_per_round = 0.998

# Model function
model_func = lambda: client_model(model_name)
init_model = model_func()

# Initalise the model for all methods with a random seed or load it from a saved initial model
torch.manual_seed(37)

print('%sModel/%s/%s_init_mdl.pt' % (data_path, data_obj.name, model_name))


init_model = model_func()
if not os.path.exists('%sModel/%s/%s_init_mdl.pt' % (data_path, data_obj.name, model_name)):
    if not os.path.exists('%sModel/%s/' % (data_path, data_obj.name)):
        print("Create a new directory")
        os.mkdir('%sModel/%s/' % (data_path, data_obj.name))
    torch.save(init_model.state_dict(), '%sModel/%s/%s_init_mdl.pt' % (data_path, data_obj.name, model_name))
else:
    # Load model
    init_model.load_state_dict(torch.load('%sModel/%s/%s_init_mdl.pt' % (data_path, data_obj.name, model_name)))

# # ####
print('FedDC')

alpha_coef = 0.1
learning_rate = 0.1
print_per = 1

n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / n_client
n_iter_per_epoch = np.ceil(n_data_per_client / batch_size)
n_minibatch = (epoch * n_iter_per_epoch).astype(np.int64)

[avg_ins_mdls, avg_cld_mdls, avg_all_mdls, trn_sel_clt_perf, tst_sel_clt_perf, trn_cur_cld_perf, tst_cur_cld_perf,
 trn_all_clt_perf, tst_all_clt_perf] = train_FedDC(data_obj=data_obj, act_prob=act_prob, n_minibatch=n_minibatch,
                                                   learning_rate=learning_rate, batch_size=batch_size, epoch=epoch,
                                                   com_amount=com_amount, print_per=print_per,
                                                   weight_decay=weight_decay,
                                                   model_func=model_func, init_model=init_model, alpha_coef=alpha_coef,
                                                   sch_step=1, sch_gamma=1, save_period=save_period, suffix=suffix,
                                                   trial=True,
                                                   data_path=data_path, lr_decay_per_round=lr_decay_per_round)
## ####


# baselines
print('FedDyn')

alpha_coef = 1e-2
learning_rate = 0.1
print_per = 1

[avg_ins_mdls, avg_cld_mdls, avg_all_mdls, trn_sel_clt_perf, tst_sel_clt_perf, trn_cur_cld_perf, tst_cur_cld_perf,
 trn_all_clt_perf, tst_all_clt_perf] = train_FedDyn(data_obj=data_obj, act_prob=act_prob,
                                                    learning_rate=learning_rate, batch_size=batch_size, epoch=epoch,
                                                    com_amount=com_amount, print_per=print_per,
                                                    weight_decay=weight_decay,
                                                    model_func=model_func, init_model=init_model, alpha_coef=alpha_coef,
                                                    sch_step=1, sch_gamma=1, save_period=save_period, suffix=suffix,
                                                    trial=True,
                                                    data_path=data_path, lr_decay_per_round=lr_decay_per_round)

# ###
print('SCAFFOLD')

n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / n_client
n_iter_per_epoch = np.ceil(n_data_per_client / batch_size)

n_minibatch = (epoch * n_iter_per_epoch).astype(np.int64)
learning_rate = 0.1
print_per = 1

[fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all] = train_SCAFFOLD(data_obj=data_obj,
                                                                                                      act_prob=act_prob,
                                                                                                      learning_rate=learning_rate,
                                                                                                      batch_size=batch_size,
                                                                                                      n_minibatch=n_minibatch,
                                                                                                      com_amount=com_amount,
                                                                                                      print_per=n_minibatch // 2,
                                                                                                      weight_decay=weight_decay,
                                                                                                      model_func=model_func,
                                                                                                      init_model=init_model,
                                                                                                      sch_step=1,
                                                                                                      sch_gamma=1,
                                                                                                      save_period=save_period,
                                                                                                      suffix=suffix,
                                                                                                      trial=True,
                                                                                                      data_path=data_path,
                                                                                                      lr_decay_per_round=lr_decay_per_round)

####
print('FedAvg')

learning_rate = 0.1
print_per = 1

[fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all] = train_FedAvg(data_obj=data_obj,
                                                                                                    act_prob=act_prob,
                                                                                                    learning_rate=learning_rate,
                                                                                                    batch_size=batch_size,
                                                                                                    epoch=epoch,
                                                                                                    com_amount=com_amount,
                                                                                                    print_per=print_per,
                                                                                                    weight_decay=weight_decay,
                                                                                                    model_func=model_func,
                                                                                                    init_model=init_model,
                                                                                                    sch_step=1,
                                                                                                    sch_gamma=1,
                                                                                                    save_period=save_period,
                                                                                                    suffix=suffix,
                                                                                                    trial=True,
                                                                                                    data_path=data_path,
                                                                                                    lr_decay_per_round=lr_decay_per_round)

####
print('FedProx')

learning_rate = 0.1
print_per = 1
mu = 1e-4

[fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all] = train_FedProx(data_obj=data_obj,
                                                                                                     act_prob=act_prob,
                                                                                                     learning_rate=learning_rate,
                                                                                                     batch_size=batch_size,
                                                                                                     epoch=epoch,
                                                                                                     com_amount=com_amount,
                                                                                                     print_per=print_per,
                                                                                                     weight_decay=weight_decay,
                                                                                                     model_func=model_func,
                                                                                                     init_model=init_model,
                                                                                                     sch_step=1,
                                                                                                     sch_gamma=1,
                                                                                                     save_period=save_period,
                                                                                                     mu=mu,
                                                                                                     suffix=suffix,
                                                                                                     trial=True,
                                                                                                     data_path=data_path,
                                                                                                     lr_decay_per_round=lr_decay_per_round)
exit(0)

# Plot results
plt.figure(figsize=(6, 5))
plt.plot(np.arange(com_amount) + 1, tst_all_clt_perf[:com_amount, 1], label='FedDyn')
plt.ylabel('Test Accuracy', fontsize=16)
plt.xlabel('Communication Rounds', fontsize=16)
plt.legend(fontsize=16, loc='lower right', bbox_to_anchor=(1.015, -0.02))
plt.grid()
plt.xlim([0, com_amount + 2])
plt.title(data_obj.name, fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('%s.pdf' % data_obj.name, dpi=1000, bbox_inches='tight')
# plt.show()
