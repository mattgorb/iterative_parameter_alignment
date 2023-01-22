from __future__ import print_function
import argparse
import torch
import torch.nn as nn

import torch.optim as optim


class Trainer:
    def __init__(self, args, datasets, model, device, save_path, model_name):
        self.args = args
        self.model = model
        self.train_loader, self.test_loader = datasets[0], datasets[1]
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.device = device
        self.save_path = save_path
        self.model_name = model_name

        self.fc1_norm_list = []
        self.fc2_norm_list = []
        self.wa1_norm_list = []
        self.wa2_norm_list = []
        self.train_iter_list=[]
        self.train_iter=0

    def fit(self, log_output=False):
        self.train_iter+=1
        for epoch in range(1, self.args.epochs + 1):
            self.train()
            test_loss, test_acc = self.test()
            self.test_loss = test_loss
            self.test_acc = test_acc

            #if epoch_loss < self.train_loss:
                #torch.save(self.model.state_dict(), self.save_path)
            if log_output:
                print( f'Epoch: {epoch}, Train loss: {self.train_loss}, Test loss: {self.test_loss}, Test Acc: {self.test_acc}')

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
            '''
            weight_align_factor=250 works for this particular combination, summing both CrossEntropyLoss and weight alignment
            For model w/o weight alignment paramter, second part of loss is 0  
            '''
            loss = self.criterion(output, target) + self.args.weight_align_factor * weight_align
            train_loss += loss
            train_loss_ce += self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

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
