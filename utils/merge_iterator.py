from __future__ import print_function
from utils.trainer import Trainer
import torch.optim as optim
from utils.model_utils import model_selector
from utils.align_util import set_weight_align_param


class Merge_Iterator:
    def __init__(self, args, train_loaders, test_loader, device, weight_dir):

        self.args = args
        self.device = device
        self.weight_dir = weight_dir
        #self.train_loader1 = datasets[0]
        #self.train_loader2 = datasets[1]
        self.num_clients=self.args.num_clients
        self.train_loaders=train_loaders
        self.test_loader = test_loader

    def run(self):
        merge_iterations = self.args.merge_iter
        #intra_merge_iterations=[10 for i in range(2)]+[5 for i in range(2)]+[2 for i in range(10)]+[1 for i in range(10000)]

        self.models=[model_selector(self.args) for i in range(self.num_clients)]


        self.model_trainers=[Trainer(self.args, [self.train_loaders[i],
                                     self.test_loader], self.models[i], self.device,
                                  f'{self.weight_dir}model{i}_0.pt', f'model{i}_{self.args.model}')
                        for i in range(self.num_clients)]


        for iter in range(merge_iterations):
            for trainer in self.model_trainers:
                trainer.fit()
                print(f'Model {trainer.model_name} Train loss: {trainer.train_loss}, '
                      f'Train CE loss: {trainer.train_loss_ce}, '
                      f'Test loss: {trainer.test_loss},  '
                      f'Test accuracy: {trainer.test_acc}')

            if iter==0:
                set_weight_align_param(self.models, self.args)
                for trainer in self.model_trainers:
                    trainer.optimizer=optim.Adam(trainer.model.parameters(), lr=self.args.lr)

            print(f'Summary, Merge Iteration: {iter}')
            for i in range(len(self.model_trainers)):
                trainer=self.model_trainers[i]
                print(f'\tModel {i} Train loss: {trainer.train_loss}, '
                      f'Train CE loss: {trainer.train_loss_ce}, '
                      f'Test loss: {trainer.test_loss},  '
                      f'Test accuracy: {trainer.test_acc}')

