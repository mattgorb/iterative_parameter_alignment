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
from models.layers import LinearMerge, ConvMerge
import scipy.spatial

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
                             f'_same_init_{self.args.same_initialization}_le_{self.args.local_epochs}_s_{self.args.single_model}'

        self.tensorboard_dir=f'{self.args.base_dir}Runs/{self.args.dataset}/{self.model_cnf_str}'



        if os.path.exists(self.tensorboard_dir):
            shutil.rmtree(self.tensorboard_dir)

        self.writer = SummaryWriter(self.tensorboard_dir)

    def ensemble(self):
        correct1 = 0
        correct2 = 0

        with torch.no_grad():

            for data, target in self.model_trainers[0].test_loader:
                data, target = data.to(self.device), target.to(self.device)

                for idx,trainer in enumerate(self.model_trainers):
                    model=trainer.model
                    model.eval()
                    output, sd = model(data, )
                    if idx==0:
                        out_all=output.unsqueeze(dim=2)
                        out_max=output
                    else:
                        out_all=torch.cat([out_all, output.unsqueeze(dim=2)], dim=2)
                        out_max = torch.cat([out_max, output], dim=1)

                avg_pred_ensemble=torch.mean(out_all, dim=2)
                avg_pred_ensemble = avg_pred_ensemble.argmax(dim=1, keepdim=True)

                out_max=out_max.view(out_max.size(0), -1)
                top_pred_ensemble = torch.remainder(out_max.argmax(1), output.size(1))

                correct1 += avg_pred_ensemble.eq(target.view_as(avg_pred_ensemble)).sum().item()
                correct2 += top_pred_ensemble.eq(target.view_as(top_pred_ensemble)).sum().item()

        print(f"Ensemeble test set results: \n\tAveraged across clients: {100. * correct1 / len(self.test_loader.dataset)}"
              f"\n\tTop prediction across clients: {100. * correct2 / len(self.test_loader.dataset)}")

    def comparison_statistics(self, iteration):
        '''
        1. Distance between each models parameters
        2. Distance between each models outputs on test set
        '''

        dist_matrix_p1=[]
        dist_matrix_p2=[]
        for idx, trainer in enumerate(self.model_trainers):
            model1 = trainer.model
            model1.load_state_dict(torch.load(trainer.save_path)['model_state_dict'])
            model1.eval()

            model1.eval()
            dist_matrix2_p1=[]
            dist_matrix2_p2 = []
            for idx2, trainer2 in enumerate(self.model_trainers):
                model2 = trainer2.model
                model2.load_state_dict(torch.load(trainer2.save_path)['model_state_dict'])
                model2.eval()


                model1_param_list=torch.Tensor().to(self.args.device)
                model2_param_list=torch.Tensor().to(self.args.device)
                for model1_mods, model2_mods, in zip(model1.named_modules(), model2.named_modules()):
                    n1, m1 = model1_mods
                    n2, m2 = model2_mods
                    if not type(m1) == LinearMerge and not type(m1) == ConvMerge:
                        continue
                    if hasattr(m1, "weight"):
                        model1_param_list=torch.cat([model1_param_list, torch.flatten(m1.weight)])
                        model2_param_list=torch.cat([model2_param_list, torch.flatten(m2.weight)])
                    if hasattr(m1, "bias"):
                        model1_param_list=torch.cat([model1_param_list, torch.flatten(m1.bias)])
                        model2_param_list=torch.cat([model2_param_list, torch.flatten(m2.bias)])


                assert(model1_param_list.size()==model2_param_list.size())

                #Distance metrics
                dist_matrix2_p1.append(torch.cdist(torch.unsqueeze(model1_param_list, dim=0),torch.unsqueeze(model2_param_list, dim=0), p=1).item())
                dist_matrix2_p2.append(torch.cdist(torch.unsqueeze(model1_param_list, dim=0),torch.unsqueeze(model2_param_list, dim=0), p=2).item())


                del model1_param_list
                del model2_param_list

            dist_matrix_p1.append(dist_matrix2_p1)
            dist_matrix_p2.append(dist_matrix2_p2)


        print('Parameter Distances')
        print(dist_matrix_p1)
        print(dist_matrix_p2)

        np.save(f'{self.args.base_dir}weight_alignment_similarity/{self.model_cnf_str}_p1_weight_distance_iter_{iteration}.npy',  dist_matrix_p1)

        np.save(f'{self.args.base_dir}weight_alignment_similarity/{self.model_cnf_str}_p2_weight_distance_iter_{iteration}.npy', dist_matrix_p2)


        model_scores = {}
        model_scores_hamming = {}

        for idx, trainer in enumerate(self.model_trainers):
            model = trainer.model

            model.load_state_dict(torch.load(trainer.save_path)['model_state_dict'])

            model.eval()
            print(trainer.save_path)

            scores = torch.Tensor().to(self.args.device)
            preds = torch.LongTensor().to(self.args.device)

            correct = 0
            with torch.no_grad():
                for data, labels in self.model_trainers[0].test_loader:
                    data = data.to(self.args.device)
                    outputs,_ = model(data)
                    _, predicted = torch.max(outputs, 1)

                    scores=torch.cat([scores, outputs], dim=0)
                    preds=torch.cat([preds, predicted], dim=0)

                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            print(f'{idx}: 100. * correct / len(self.test_loader.dataset)')


            model_scores[idx] = scores
            model_scores_hamming[idx]= preds


        sys.exit()
        distance_p1=[]
        distance_p2=[]
        for key, value in model_scores.items():
            distance2_p1 = []
            distance2_p2 = []
            for key2, value2 in model_scores.items():
                distance2_p1.append(torch.cdist(torch.unsqueeze(torch.flatten(value), dim=0),
                                                torch.unsqueeze(torch.flatten(value2), dim=0), p=1).item())
                distance2_p2.append(torch.cdist(torch.unsqueeze(torch.flatten(value), dim=0),
                                                torch.unsqueeze(torch.flatten(value2), dim=0), p=2).item())


            distance_p1.append(distance2_p1)
            distance_p2.append(distance2_p2)

        np.save(f'{self.args.base_dir}weight_alignment_similarity/{self.model_cnf_str}_scores_p1_iter_{iteration}.npy', distance_p1)
        np.save(f'{self.args.base_dir}weight_alignment_similarity/{self.model_cnf_str}_scores_p2_iter_{iteration}.npy', distance_p2)

        print('Prediction Distances')
        print(distance_p1)
        print(distance_p2)

        distance_p1=[]
        for key, value in model_scores_hamming.items():
            distance2_p1 = []
            for key2, value2 in model_scores_hamming.items():
                #distance2_p1.append(torch.cdist(torch.unsqueeze(value.double(), dim=0),
                                                #torch.unsqueeze(value2.double(), dim=0), p=1).item())
                hamming_bool=(value.double()!=value2.double())
                distance2_p1.append(torch.sum(hamming_bool).item())

            distance_p1.append(distance2_p1)
        np.save(f'{self.args.base_dir}weight_alignment_similarity/{self.model_cnf_str}_scores_hamming_iter_{iteration}.npy', distance_p1)

        print('Prediction Hamming Distances')
        print(distance_p1)

        print(hamming_bool.size())
        print(value[:100])
        print(value2[:100])
        sys.exit()

    def run(self):
        merge_iterations = self.args.merge_iter

        if self.args.same_initialization:
            self.models = [model_selector(self.args) for i in range(self.num_clients)]
        else:
            self.models=[]
            for _ in range(self.num_clients):
                self.models.append(model_selector(self.args))
                self.args.weight_seed+=1

                print(f'Setting weight seed to {self.args.weight_seed}')
                for idx,model1 in enumerate(self.models):
                    for idx2, model2 in enumerate(self.models):
                        if idx==idx2:
                            continue
                        if model1.fc1.weight[0][0]==model2.fc1.weight[0][0]:
                            print('initial weights are the same')
                            sys.exit(0)
                        else:
                            print('models have different weights')
        '''
        self.models = [torch.nn.DataParallel(
            model_selector(self.args),
            device_ids=[7, 0, 1, 2, 3, 4, 5, 6]).to(self.device)
            for i in range(self.num_clients)]
        '''


        model_parameters = filter(lambda p: p.requires_grad, self.models[0].parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'Model parameters: {params}')





        if self.args.single_model:
            print("Running single model")
            # run this for IID datasets
            if self.args.dataset_split=="iid":
                print("Running single model shared across clients for IID data")
                self.model = model_selector(self.args)
                self.model_trainers=[Trainer(self.args, [self.train_loaders[i],
                                         self.test_loader], self.model, self.device,
                                       f'model_single_{self.args.dataset}_'+self.model_cnf_str, )
                            for i in range(self.num_clients)]
            else:
                print("Running single model on each peer with parameter sharing")
                self.model_trainers = [
                    Trainer(self.args, [self.train_loaders[i], self.test_loader], self.models[i], self.device,
                            f'model_{i}_baseline_{self.args.dataset}_' + self.model_cnf_str, ) for i in range(self.num_clients)]

        else:

            self.model_trainers = [Trainer(self.args, [self.train_loaders[i], self.test_loader], self.models[i], self.device,
                                           f'model_{i}_{self.args.dataset}_' + self.model_cnf_str, ) for i in range(self.num_clients)]

            set_weight_align_param(self.models, self.args, self.train_weight_list)

        for trainer in self.model_trainers:
            trainer.optimizer = optim.Adam(trainer.model.parameters(), lr=self.args.lr)

        for iter in range(merge_iterations+1):
            client=0
            for trainer in self.model_trainers:
                trainer.fit()

                print(f'Model {trainer.model_name} Train loss: {trainer.train_loss}, '
                      f'Train CE loss: {trainer.train_loss_ce}, '
                      f'Test loss: {trainer.test_loss},  '
                      f'Test accuracy: {trainer.test_acc}')

                client+=1

                self.client_to_tensorboard(iter, client, trainer)

            if iter%10 ==0:
                self.ensemble()
                self.comparison_statistics(iter)

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