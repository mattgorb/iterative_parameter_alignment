from __future__ import print_function
from utils.trainer import Trainer
import torch.optim as optim
from utils.model_utils import model_selector
from utils.align_util import set_weight_align_param
import numpy as np
import torch
import pandas as pd
from tensorboardX import SummaryWriter
import os
import shutil

class Merge_Iterator:
    def __init__(self, args, train_loaders, test_loader,train_weight_list, device, weight_dir):

        self.args = args
        self.device = device
        self.weight_dir = weight_dir
        self.num_clients=self.args.num_clients
        self.train_loaders=train_loaders
        self.test_loader = test_loader
        self.train_weight_list=train_weight_list

        #for logging
        self.client_list=[]
        self.iter_list=[]
        self.train_losses=[]
        self.train_ce_losses=[]
        self.test_losses=[]
        self.test_accuracy_list=[]
        self.best_test_accuracy=[]
        self.average_test_accuracy=[]

        self.model_cnf_str= f'model_{self.args.model}_n_cli_{self.args.num_clients}_ds_split_{self.args.dataset_split}_ds_alpha_{self.args.dirichlet_alpha}' \
                             f'_align_{self.args.align_loss}_waf_{self.args.weight_align_factor}_delta_{self.args.delta}_init_type_{self.args.weight_init}' \
                             f'_same_init_{self.args.same_initialization}_le_{self.args.local_epochs}_e_{self.args.ensemble}_s_{self.args.single}'

        self.tensorboard_dir=f'{self.args.base_dir}Runs/{self.args.dataset}/self.model_cnf_str'

        if os.path.exists(self.tensorboard_dir):
            shutil.rmtree(self.tensorboard_dir)

        self.writer = SummaryWriter(self.tensorboard_dir)

    def ensemble(self):
        self.model.eval()
        correct1 = 0
        correct2 = 0

        with torch.no_grad():

            for data, target in self.model_trainers[0].test_loader:
                data, target = data.to(self.device), target.to(self.device)

                for idx,trainer in enumerate(self.model_trainers):
                    model=trainer.model
                    model.eval()
                    output, sd = self.model(data, )
                    if idx==0:
                        out_all=output.unsqueeze(dim=2)
                        out_max=output
                    else:
                        out_all=torch.cat([out_all, output.unsqueeze(dim=2)])
                        out_max = torch.cat([out_max, output], dim=1)


                avg_pred_ensemble=torch.mean(out_all, dim=2)
                avg_pred_ensemble = avg_pred_ensemble.argmax(dim=1, keepdim=True)

                out_max=out_max.view(out_max.size(0), -1)
                top_pred_ensemble = torch.remainder(out_max.argmax(1), output.size(1))


                correct1 += avg_pred_ensemble.eq(target.view_as(avg_pred_ensemble)).sum().item()


                correct2 += top_pred_ensemble.eq(target.view_as(top_pred_ensemble)).sum().item()

        print(f"Ensemeble test set results: \n\tAveraged across clients: {100. * correct1 / len(self.test_loader.dataset)}"
              f"\n\tTop prediction across clients: {100. * correct2 / len(self.test_loader.dataset)}")



    def run(self):
        merge_iterations = self.args.merge_iter
        #intra_merge_iterations=[10 for i in range(2)]+[5 for i in range(2)]+[2 for i in range(10)]+[1 for i in range(10000)]


        if self.args.same_initialization:
            self.models = [model_selector(self.args) for i in range(self.num_clients)]
        else:
            self.models=[]
            for _ in range(self.num_clients):
                self.models.append(model_selector(self.args))
                self.args.weight_seed+=1
                print(f'Setting weight seed to {self.args.weight_seed}')



        '''
        self.models = [torch.nn.DataParallel(
            model_selector(self.args),
            device_ids=[7, 0, 1, 2, 3, 4, 5, 6]).to(self.device)
            for i in range(self.num_clients)]
        '''


        model_parameters = filter(lambda p: p.requires_grad, self.models[0].parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'Model parameters: {params}')




        #run this for IID datasets
        if self.args.single_model:
            self.models = model_selector(self.args)
            self.model_trainers=[Trainer(self.args, [self.train_loaders[i],
                                     self.test_loader], self.model, self.device,
                                   f'model_single_{self.args.dataset}_'+self.model_cnf_str, )
                        for i in range(self.num_clients)]

        else:
            self.model_trainers = [Trainer(self.args, [self.train_loaders[i], self.test_loader], self.models[i], self.device,
                                           f'model_{i}_{self.args.dataset}_' + self.model_cnf_str, ) for i in range(self.num_clients)]

            set_weight_align_param(self.models, self.args, self.train_weight_list)

        for trainer in self.model_trainers:
            trainer.optimizer = optim.Adam(trainer.model.parameters(), lr=self.args.lr)

        for iter in range(1,merge_iterations+1):
            client=0
            for trainer in self.model_trainers:
                trainer.fit()

                print(f'Model {trainer.model_name} Train loss: {trainer.train_loss}, '
                      f'Train CE loss: {trainer.train_loss_ce}, '
                      f'Test loss: {trainer.test_loss},  '
                      f'Test accuracy: {trainer.test_acc}')

                client+=1

                self.client_to_tensorboard(iter, client, trainer)

            self.ensemble()

            self.log_results(iter)

    def log_results(self,iter):
            print(f'Summary, Merge Iteration: {iter}')
            avg_acc=0
            avg_loss=0
            test_accs=[]
            for i in range(len(self.model_trainers)):
                trainer=self.model_trainers[i]
                print(f'\tModel {i} Train loss: {trainer.train_loss}, '
                      f'Train CE loss: {trainer.train_loss_ce}, '
                      f'Test loss: {trainer.test_loss},  '
                      f'Test accuracy: {trainer.test_acc}')
                avg_acc+=trainer.test_acc
                avg_loss+=trainer.test_loss
                test_accs.append(trainer.test_acc)

            self.best_test_accuracy.append(max(test_accs))
            self.average_test_accuracy.append(avg_loss/len(self.model_trainers))

            print(f'\tAverages: Test loss: {avg_loss/len(self.model_trainers)},Test accuracy: {avg_acc/len(self.model_trainers)}')
            print(f'\tBest Test accuracy: {max(test_accs)}')

            self.results_to_csv()

            max_acc=max(test_accs)
            avg_acc/=len(self.model_trainers)
            avg_loss/=len(self.model_trainers)

            self.writer.add_scalars('Accuracy/test', {  'max_client_test_accuracy': max_acc,
                                        'avg_client_test_accuracy': avg_acc ,
                                        'avg_client_test_loss': avg_loss}, iter)







    def client_to_tensorboard(self,iter, client , trainer):
        self.client_list.append(client)
        self.iter_list.append(iter)
        self.train_losses.append(trainer.train_loss)
        self.train_ce_losses.append(trainer.train_loss_ce)
        self.test_losses.append(trainer.test_loss)
        self.test_accuracy_list.append(trainer.test_acc)

        self.writer.add_scalars('ClientPerformance/test',
                           {
                               'client_num':client,
                               'test_loss':trainer.test_loss,
                               'test_acc':trainer.test_acc
                           }, iter)


    def results_to_csv(self):

        df = pd.DataFrame({'client_list': self.client_list,
                           'iter_list': self.iter_list,
                           'train_losses': self.train_losses,
                           'train_ce_losses': self.train_ce_losses,
                           'test_losses': self.test_losses,
                           'test_accuracy_list': self.test_accuracy_list, })
        df.to_csv(
            f'{self.args.base_dir}/weight_alignment_csvs/client_results_ds_{self.args.dataset}_model_{self.args.model}_cli'
            f'_{self.args.num_clients}_split_{self.args.dataset_split}_dir_alph_{self.args.dirichlet_alpha}_align'
            f'_{self.args.align_loss}_waf_{self.args.weight_align_factor}_{self.args.weight_init}'
                                    f'_same_init_{self.args.same_initialization}.csv')

        df = pd.DataFrame({'best_test_accuracy': self.best_test_accuracy,
                           'average_test_accuracy': self.average_test_accuracy, })
        df.to_csv(
            f'{self.args.base_dir}/weight_alignment_csvs/overall_results_ds_{self.args.dataset}_model_{self.args.model}_cli'
            f'_{self.args.num_clients}_split_{self.args.dataset_split}_dir_alph_{self.args.dirichlet_alpha}_align'
            f'_{self.args.align_loss}_waf_{self.args.weight_align_factor}_{self.args.weight_init}'
                                    f'_same_init_{self.args.same_initialization}.csv')