from __future__ import print_function
from matplotlib import pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import random
from torch.optim.lr_scheduler import CosineAnnealingLR

import pandas as pd
import numpy as np

import torch


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def linear_init(in_dim, out_dim, bias=False, args=None, ):
    layer = LinearMerge(in_dim, out_dim, bias=bias)
    layer.init(args)
    return layer


class LinearMerge(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_align = None
        #self.loss=torch.nn.HuberLoss(reduction='sum')
        self.delta=nn.Parameter(torch.zeros(1), requires_grad=True)


    def init(self, args):
        self.args = args
        set_seed(self.args.weight_seed)
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)

        weights_diff_ae = torch.tensor(0)
        #weights_diff_se = torch.tensor(0)
        #print(self.delta)
        if self.weight_align is not None:
            #weights_diff_ae = torch.sum((self.weight - self.weight_align).abs())
            weights_diff_ae = torch.sum((self.weight - self.weight_align).abs())
            #weights_diff_ae = torch.sum((self.weight - self.weight_align).abs())
            weights_diff_se = torch.sum(torch.sqrt(torch.square(self.weight - self.weight_align)))


            #if weights_diff_se!=weights_diff_ae:
            #sys.exit()

        #align_loss=F.sigmoid(self.delta)*weights_diff_ae+(1-F.sigmoid(self.delta))*weights_diff_se

        return out, weights_diff_ae, weights_diff_se

class Net(nn.Module):
    def __init__(self, args, weight_merge=False):
        super(Net, self).__init__()
        self.args = args
        self.weight_merge = weight_merge
        if self.weight_merge:
            self.fc1 = linear_init(28 * 28, 1024, bias=False, args=self.args, )
            self.fc2 = linear_init(1024, 10, bias=False, args=self.args, )
        else:
            self.fc1 = nn.Linear(28 * 28, 1024, bias=False)
            self.fc2 = nn.Linear(1024, 10, bias=False)

    def forward(self, x,):
        if self.weight_merge:
            x, wa1_ae, wa1_se = self.fc1(x.view(-1, 28 * 28))
            x = F.relu(x)
            x, wa2_ae, wa2_se = self.fc2(x)
            score_diff_ae = wa1_ae + wa2_ae
            score_diff_se = wa1_se + wa2_se
            return x, score_diff_ae, score_diff_se
        else:
            x = self.fc1(x.view(-1, 28 * 28))
            x = F.relu(x)
            x = self.fc2(x)
            return x, torch.tensor(0), torch.tensor(0)


def get_datasets(args):
    # not using normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    if args.baseline:
        dataset1 = datasets.MNIST(f'{args.base_dir}data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(f'{args.base_dir}data', train=False, transform=transform)
        train_loader = DataLoader(dataset1, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        return train_loader, test_loader
    else:
        dataset1 = datasets.MNIST(f'{args.base_dir}data', train=True, download=True, transform=transform)
        dataset2 = datasets.MNIST(f'{args.base_dir}data', train=True, transform=transform)
        # split dataset in half by labels
        labels = torch.unique(dataset1.targets)
        ds1_labels = labels[:len(labels) // 2]
        ds2_labels = labels[len(labels) // 2:]
        print(f'ds1_labels: {ds1_labels}')
        print(f'ds2_labels: {ds2_labels}')
        ds1_indices = [idx for idx, target in enumerate(dataset1.targets) if target in ds1_labels]
        ds2_indices = [idx for idx, target in enumerate(dataset1.targets) if target in ds2_labels]

        dataset1.data, dataset1.targets = dataset1.data[ds1_indices], dataset1.targets[ds1_indices]
        dataset2.data, dataset2.targets = dataset2.data[ds2_indices], dataset2.targets[ds2_indices]
        assert (set(ds1_indices).isdisjoint(ds2_indices))
        test_dataset = datasets.MNIST(f'{args.base_dir}data', train=False, transform=transform)
        train_loader1 = DataLoader(dataset1, batch_size=args.batch_size, shuffle=True)
        train_loader2 = DataLoader(dataset2, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        return train_loader1, train_loader2, test_loader


class Trainer:
    def __init__(self, args, datasets, model, device, save_path, model_name):
        self.args = args
        self.model = model
        self.train_loader, self.test_loader = datasets[0], datasets[1]


        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        #self.optimizer = optim.SGD(self.model.parameters(), lr=0.1,  weight_decay=1e-3)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=1)

        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.device = device
        self.save_path = save_path
        self.model_name = model_name

        #Results lists
        self.weight_align_ae_loss_list=[]
        self.weight_align_se_loss_list=[]

        self.test_accuracy_list=[]
        self.train_loss_list=[]
        self.test_loss_list=[]
        self.batch_epoch_list=[]
        self.epoch_list=[]


    def fit(self, log_output=False):
        self.train_loss = 1e6

        for epoch in range(1, self.args.epochs + 1):
            epoch_loss = self.train()
            self.train_loss = epoch_loss

            test_loss, test_acc = self.test()
            self.test_loss = test_loss
            self.test_acc = test_acc

            #self.scheduler.step()

            if log_output:
                print(f'Epoch: {epoch}, Train loss: {self.train_loss}, Test loss: {self.test_loss}, Test Acc: {self.test_acc}')

            if self.args.baseline:
                self.test_accuracy_list.append(self.test_acc)
                self.train_loss_list.append(self.train_loss)
                self.test_loss_list.append(self.test_loss)
                self.epoch_list.append(epoch)

        if not self.args.baseline:
            self.test_accuracy_list.append(self.test_acc)
            self.train_loss_list.append(self.train_loss)
            self.test_loss_list.append(self.test_loss)
            self.epoch_list.append(self.merge_iter)


    def model_loss(self):
        return self.best_loss

    def train(self, ):
        self.model.train()
        train_loss = 0
        train_align_loss=0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output, weight_align_ae, weight_align_se = self.model(data)
            if self.args.align_loss=='ae' or self.args.baseline==True:
                weight_align_loss=weight_align_ae
            elif self.args.align_loss == 'se':
                weight_align_loss=weight_align_se
            else:
                print('Set align loss')
                sys.exit()
            #loss = self.criterion(output, target) + self.criterion(output, target)*self.args.weight_align_factor * weight_align_loss
            loss = self.criterion(output, target)+self.args.weight_align_factor * weight_align_loss


            train_loss += loss.item()
            train_align_loss += weight_align_loss.item()

            loss.backward()

            #torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)

            self.optimizer.step()

        self.train_align_loss=train_align_loss/len(self.train_loader.dataset)
        train_loss /= len(self.train_loader.dataset)
        return train_loss

    def test(self, ):
        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, _, _ = self.model(data, )
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.test_loader.dataset)
        return test_loss, 100. * correct / len(self.test_loader.dataset)


def set_weight_align_param(model1, model2, args):
    for model1_mods, model2_mods, in zip(model1.named_modules(), model2.named_modules(), ):
        n1, m1 = model1_mods
        n2, m2 = model2_mods
        if not type(m2) == LinearMerge:
            continue
        if hasattr(m1, "weight"):
            '''
            m1.weight gets updated to m2.weight_align because it is not detached.  and vice versa
            This is a simple way to "share" the weights between models. 
            Alternatively we could set m1.weight=m2.weight_align after merge model is done training.  
            '''

            # We only want to merge one models weights in this file
            #m2.weight_align = nn.Parameter(m1.weight, requires_grad=True)
            #m1.weight_align = nn.Parameter(m2.weight, requires_grad=True)
            m2.weight_align = m1.weight
            m1.weight_align = m2.weight

class Merge_Iterator:
    def __init__(self, args, datasets, device, weight_dir):
        self.args = args
        self.device = device
        self.weight_dir = weight_dir
        self.train_loader1 = datasets[0]
        self.train_loader2 = datasets[1]
        self.test_dataset = datasets[2]


    def run(self):
        merge_iterations = self.args.merge_iter

        model1 = Net(self.args, weight_merge=True).to(self.device)
        model2 = Net(self.args, weight_merge=True).to(self.device)

        self.model1_trainer = Trainer(self.args, [self.train_loader1, self.test_dataset], model1, self.device,
                                 f'{self.weight_dir}model1_0.pt', 'model1_double')
        self.model2_trainer = Trainer(self.args, [self.train_loader2, self.test_dataset], model2, self.device,
                                 f'{self.weight_dir}model2_0.pt', 'model2_double')



        set_weight_align_param(model1, model2, self.args)
        self.model1_trainer.optimizer = optim.Adam(model1.parameters(), lr=self.args.lr)
        self.model2_trainer.optimizer = optim.Adam(model2.parameters(), lr=self.args.lr)


        for iter in range(merge_iterations):
            self.model1_trainer.merge_iter=iter
            self.model2_trainer.merge_iter=iter


            self.model1_trainer.fit()
            self.model2_trainer.fit()


            print(f'm1 d: {self.model1_trainer.model.fc1.delta}')
            print(f'm1 d: {self.model1_trainer.model.fc2.delta}')
            print(f'm2 d: {self.model2_trainer.model.fc1.delta}')
            print(f'm2 d: {self.model2_trainer.model.fc2.delta}')

            print(f'Merge Iteration: {iter} \n'
                  f'\tModel 1 Train loss: {self.model1_trainer.train_loss},Train align loss: {self.model1_trainer.train_align_loss}, Test loss: {self.model1_trainer.test_loss},  Test accuracy: {self.model1_trainer.test_acc}\n'
                  f'\tModel 2 Train loss: {self.model2_trainer.train_loss},Train align loss: {self.model1_trainer.train_align_loss},Test loss: {self.model2_trainer.test_loss},  Test accuracy: {self.model2_trainer.test_acc}')



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Weight Align')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train')
    parser.add_argument('--merge_iter', type=int, default=2500,
                        help='number of iterations to merge')
    parser.add_argument('--weight_align_factor', type=float, default=25, )
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7,
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--weight_seed', type=int, default=1, )
    parser.add_argument('--gpu', type=int, default=6, )
    parser.add_argument('--align_loss', type=str, default=None)
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--baseline', type=bool, default=False, help='train base model')
    parser.add_argument('--set_weight_from_weight_align', type=bool, default=True, )
    parser.add_argument('--graphs', type=bool, default=False, help='add norm graphs during training')
    parser.add_argument('--base_dir', type=str, default="/s/luffy/b/nobackup/mgorb/",
                        help='Directory for data and weights')
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    weight_dir = f'{args.base_dir}iwa_weights/'
    if args.baseline:
        print("Running Baseline")
        train_loader1, test_dataset = get_datasets(args)
        model = Net(args, weight_merge=False).to(device)
        save_path = f'{weight_dir}mnist_baseline.pt'
        trainer = Trainer(args, [train_loader1, test_dataset], model, device, save_path, 'model_baseline')
        trainer.fit(log_output=True)


    else:
        train_loader1, train_loader2, test_dataset = get_datasets(args)
        merge_iterator = Merge_Iterator(args, [train_loader1, train_loader2, test_dataset], device, weight_dir)
        merge_iterator.run()


if __name__ == '__main__':
    main()