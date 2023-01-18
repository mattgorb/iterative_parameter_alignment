from __future__ import print_function
from utils.trainer import Trainer
import torch.optim as optim
from utils.model_utils import model_selector
from utils.align_util import set_weight_align_param


class Merge_Iterator:
    def __init__(self, args, datasets, device, weight_dir):

        self.args = args
        self.device = device
        self.weight_dir = weight_dir
        self.train_loader1 = datasets[0]
        self.train_loader2 = datasets[1]
        self.test_dataset = datasets[2]

    def train_single(self, model, save_path, train_dataset, model_name):
        '''
        ****** We need to initialize a new optimizer during each iteration.
        Not sure why, but this is the only way it works.
        '''
        trainer = Trainer(self.args, [train_dataset, self.test_dataset], model, self.device, save_path, model_name)
        trainer.fit()
        return trainer

    def run(self):
        merge_iterations = self.args.merge_iter
        #intra_merge_iterations=[10 for i in range(2)]+[5 for i in range(2)]+[2 for i in range(10)]+[1 for i in range(10000)]

        model1 = model_selector(self.args)
        model2 = model_selector(self.args)

        model1_trainer = Trainer(self.args, [self.train_loader1, self.test_dataset], model1, self.device,
                                 f'{self.weight_dir}model1_0.pt', 'model1_double')
        model2_trainer = Trainer(self.args, [self.train_loader2, self.test_dataset], model2, self.device,
                                 f'{self.weight_dir}model2_0.pt', 'model2_double')

        '''
        AdaDelta works with re-initialization (because of the adadptive state)
        SGD works with one initialization, but requires tuning the weight_align_factor and learning rate.
        model1_trainer.optimizer = optim.SGD(model1.parameters(), lr=self.args.lr)
        model2_trainer.optimizer = optim.SGD(model2.parameters(), lr=self.args.lr)
        '''
        lr_schedule = [0.001 for i in range(3000)] + \
                      [0.0005 for i in range(2000)] + \
                      [0.0001 for i in range(3000)] + \
                      [0.00005 for i in range(2000)] + \
                      [0.000025 for i in range(3000)] + \
                      [0.00001 for i in range(2000)] + \
                      [0.000001 for i in range(5000)]

        for iter in range(merge_iterations):
            #model1_trainer.optimizer = optim.Adam(model1.parameters(), lr=lr_schedule[iter])
            #model2_trainer.optimizer = optim.Adam(model2.parameters(), lr=lr_schedule[iter])


            #print(f'Inter Merge Iterations: {intra_merge_iterations[iter]}')
            for iter2 in range(1):
            #for iter2 in range(intra_merge_iterations[iter]):
                model1_trainer.fit()
                model2_trainer.fit()

            if iter==0:
                set_weight_align_param(model1, model2, self.args)
                model1_trainer.optimizer=optim.Adam(model1.parameters(), lr=lr_schedule[iter])
                model2_trainer.optimizer=optim.Adam(model2.parameters(), lr=lr_schedule[iter])
            print(f'Merge Iteration: {iter} \n'
                  f'\tModel 1 Train loss: {model1_trainer.train_loss}, Train CE loss: {model1_trainer.train_loss_ce}, Test loss: {model1_trainer.test_loss},  Test accuracy: {model1_trainer.test_acc}\n'
                  f'\tModel 2 Train loss: {model2_trainer.train_loss}, Train CE loss: {model2_trainer.train_loss_ce}, Test loss: {model2_trainer.test_loss},  Test accuracy: {model2_trainer.test_acc}')

