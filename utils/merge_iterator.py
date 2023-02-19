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

        #self.models=[model_selector(self.args) for i in range(self.num_clients)]
        '''self.models = [torch.nn.DataParallel(
            model_selector(self.args),
            device_ids=[7, 0, 1, 2, 3, 4, 5, 6])
            for i in range(self.num_clients)]'''

        self.models = [torch.nn.DistributedDataParallel(
            model_selector(self.args),
            device_ids=[7, 0, 1, 2, 3, 4, 5, 6])
            for i in range(self.num_clients)]
        #torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6])

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

        for iter in range(merge_iterations):
            for trainer in self.model_trainers:
                trainer.fit()
                print(f'Model {trainer.model_name} Train loss: {trainer.train_loss}, '
                      f'Train CE loss: {trainer.train_loss_ce}, '
                      f'Test loss: {trainer.test_loss},  '
                      f'Test accuracy: {trainer.test_acc}')


            print(f'Summary, Merge Iteration: {iter}')
            for i in range(len(self.model_trainers)):
                trainer=self.model_trainers[i]
                print(f'\tModel {i} Train loss: {trainer.train_loss}, '
                      f'Train CE loss: {trainer.train_loss_ce}, '
                      f'Test loss: {trainer.test_loss},  '
                      f'Test accuracy: {trainer.test_acc}')

