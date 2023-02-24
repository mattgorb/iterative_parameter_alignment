from __future__ import print_function
from utils.trainer import Trainer
import torch.optim as optim
from utils.model_utils import model_selector
from utils.align_util import set_weight_align_param
import numpy as np
import torch


class Merge_Iterator:
    def __init__(self, args, train_loaders, test_loader, device, weight_dir):

        self.args = args
        self.device = device
        self.weight_dir = weight_dir
        self.num_clients=self.args.num_clients
        self.train_loaders=train_loaders
        self.test_loader = test_loader

    def run(self):
        merge_iterations = self.args.merge_iter
        #intra_merge_iterations=[10 for i in range(2)]+[5 for i in range(2)]+[2 for i in range(10)]+[1 for i in range(10000)]

        self.models=[model_selector(self.args) for i in range(self.num_clients)]
        '''self.models = [torch.nn.DataParallel(
            model_selector(self.args),
            device_ids=[7, 0, 1, 2, 3, 4, 5, 6]).to(self.device)
            for i in range(self.num_clients)]'''


        model_parameters = filter(lambda p: p.requires_grad, self.models[0].parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'Model parameters: {params}')


        self.model_trainers=[Trainer(self.args, [self.train_loaders[i],
                                     self.test_loader], self.models[i], self.device,
                                   f'model_{i}_{self.args.model}_{self.args.dataset}_n_clients{self.args.num_clients}_{self.args.align_loss}')
                        for i in range(self.num_clients)]


        set_weight_align_param(self.models, self.args)

        for trainer in self.model_trainers:
            trainer.optimizer = optim.Adam(trainer.model.parameters(), lr=self.args.lr)

        for iter in range(1,merge_iterations+1):

            for trainer in self.model_trainers:
                trainer.fit()
                self.args.delta=1.2
                print(f'Model {trainer.model_name} Train loss: {trainer.train_loss}, '
                      f'Train CE loss: {trainer.train_loss_ce}, '
                      f'Test loss: {trainer.test_loss},  '
                      f'Test accuracy: {trainer.test_acc}')


            print(f'Summary, Merge Iteration: {iter}')
            avg_acc=0
            avg_loss=0
            for i in range(len(self.model_trainers)):
                trainer=self.model_trainers[i]
                print(f'\tModel {i} Train loss: {trainer.train_loss}, '
                      f'Train CE loss: {trainer.train_loss_ce}, '
                      f'Test loss: {trainer.test_loss},  '
                      f'Test accuracy: {trainer.test_acc}')
                avg_acc+=trainer.test_acc
                avg_loss+=trainer.test_loss
            print(f'\tAverages: Test loss: {avg_loss/len(self.model_trainers)},Test accuracy: {avg_acc/len(self.model_trainers)}')

