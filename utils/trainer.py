from __future__ import print_function
import argparse
import torch
import torch.nn as nn

import torch.optim as optim
import pandas as pd

class Trainer:
    def __init__(self, args, datasets, model, device, model_name,):
        self.args = args
        self.model = model
        self.train_loader, self.test_loader = datasets[0], datasets[1]

        if self.args.optimizer=='Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=.1)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=1)

        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.device = device

        self.merge_iter=1

        self.weight_dir = f'{self.args.base_dir}iwa_weights/'

        self.model_name = model_name
        self.save_path=f'{self.weight_dir}{self.model_name}.pt'
        self.model_cnf_str= f'model_{self.args.model}_ds_{self.args.dataset}_seed_{self.args.seed}_n_cli_{self.args.num_clients}_{self.args.uneven}_ds_split_{self.args.dataset_split}_ds_alpha_{self.args.dirichlet_alpha}' \
                             f'_align_{self.args.align_loss}_waf_{self.args.weight_align_factor}_delta_{self.args.delta}_init_type_{self.args.weight_init}' \
                             f'_same_init_{self.args.same_initialization}_le_{self.args.local_epochs}_s_{self.args.single_model}_rand_top_{self.args.random_topology}'
        self.test_accs=[]
        self.epoch=[]
        self.align_losses=[]


    def fit(self, log_output=True):
        print(f'Model {self.model_name}, merge iteration: {self.merge_iter}')
        if self.merge_iter>1:
            #adam opt
            checkpoint = torch.load(self.save_path)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


        for epoch in range(1, self.args.local_epochs + 1):
            self.train()

            test_loss, test_acc = self.test()
            self.test_loss = test_loss
            self.test_acc = test_acc

            self.test_accs.append(test_acc)


            if log_output:
                print( f'Local Epoch: {epoch}, Train loss: {self.train_loss}, Test loss: {self.test_loss}, Test Acc: {self.test_acc}')
            if self.args.single_model:
                df = pd.DataFrame({'test_accs': self.test_accs,})
                df.to_csv(
                    f'{self.args.base_dir}/weight_alignment_csvs/single_{self.model_name}_{self.model_cnf_str[:25]}.csv')
            if self.args.record_align_losses:
                df = pd.DataFrame({'align_losses': self.align_losses,})
                df.to_csv(
                    f'{self.args.base_dir}/weight_alignment_csvs/single_align_losses_{self.model_name}_{self.model_cnf_str[:15]}.csv')

        torch.save({
            'epoch': self.merge_iter,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, self.save_path)

        self.optimizer.zero_grad()
        del self.optimizer
        with torch.cuda.device(f"cuda:{self.args.gpu}"):
            torch.cuda.empty_cache()

        self.merge_iter+=1




    def model_loss(self):
        return self.best_loss

    def train(self, ):
        self.model.train()
        train_loss_ce=0
        train_loss=0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output, weight_align = self.model(data)

            loss = self.criterion(output, target) + self.args.weight_align_factor * weight_align
            loss.backward()


            #torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=1)  # Clip gradients to prevent exploding

            self.optimizer.step()

            train_loss += loss.item()
            train_loss_ce += self.criterion(output, target).item()

            if self.args.record_align_losses:
                self.align_losses.append(weight_align.item())

        if self.args.optimizer!='Adam':
            self.scheduler.step()

        self.train_loss_ce=train_loss_ce/len(self.train_loader.dataset)
        self.train_loss= train_loss / len(self.train_loader.dataset)

    def test(self, ):
        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, sd = self.model(data, )
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.test_loader.dataset)
        return test_loss, 100. * correct / len(self.test_loader.dataset)

