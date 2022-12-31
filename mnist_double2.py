from __future__ import print_function
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
from matplotlib import pyplot as plt
import pandas as pd

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

    def init(self, args):
        self.args = args
        set_seed(self.args.weight_seed)
        # this isn't default initialization.  not sure if necessary, need to test.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")
        # models do NOT need to be initialized the same, however they appeared to converge slightly faster with same init
        # self.args.weight_seed+=1

    def forward(self, x):
        x = F.linear(x, self.weight, self.bias)
        weights_diff = torch.tensor(0)
        if self.weight_align is not None:
            # using absolute error here.
            weights_diff = torch.sum((self.weight - self.weight_align).abs())
            # MSE loss -- not able to get as good results using this loss fn.
            # weights_diff=torch.mean((self.weight-self.weight_align)**2)
        return x, weights_diff


class Net(nn.Module):
    def __init__(self, args, weight_merge=False):
        super(Net, self).__init__()
        self.args = args
        self.weight_merge = weight_merge
        # bias False for now, have not tested adding bias to the loss fn.
        if self.weight_merge:
            self.fc1 = linear_init(28 * 28, 1024, bias=False, args=self.args, )
            self.fc2 = linear_init(1024, 10, bias=False, args=self.args, )
        else:
            self.fc1 = nn.Linear(28 * 28, 1024, bias=False)
            self.fc2 = nn.Linear(1024, 10, bias=False)

    def forward(self, x, ):
        if self.weight_merge:
            x, wa1 = self.fc1(x.view(-1, 28 * 28))
            x = F.relu(x)
            x, wa2 = self.fc2(x)
            score_diff = wa1 + wa2
            return x, score_diff
        else:
            x = self.fc1(x.view(-1, 28 * 28))
            x = F.relu(x)
            x = self.fc2(x)
            return x, torch.tensor(0)


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
        '''
        #use this code for p/1-p split.  need to test
        #p=0.8
        ds1_indices=ds1_indices[:int(len(ds1_indices)*p)]+ds2_indices[int(len(ds2_indices)*p):]
        ds2_indices=ds1_indices[int(len(ds1_indices)*p):]+ds2_indices[:int(len(ds2_indices)*p)]
        '''
        '''
        #use this code to split dataset down middle. need to test
        dataset1.data, dataset1.targets = dataset1.data[:int(len(dataset1.targets)/2)], dataset1.targets[:int(len(dataset1.targets)/2)]
        dataset2.data, dataset2.targets = dataset2.data[int(len(dataset1.targets)/2):], dataset2.targets[int(len(dataset1.targets)/2):]
        '''
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
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.device = device
        self.save_path = save_path
        self.model_name = model_name
        self.fc1_norm_list = []
        self.fc2_norm_list = []
        self.wa1_norm_list = []
        self.wa2_norm_list = []

    def fit(self, log_output=False):
        self.train_loss = 1e6
        for epoch in range(1, self.args.epochs + 1):
            epoch_loss = self.train()
            self.train_loss = epoch_loss
            test_loss, test_acc = self.test()
            self.test_loss = test_loss
            self.test_acc = test_acc

            #if epoch_loss < self.train_loss:
                #torch.save(self.model.state_dict(), self.save_path)
            if log_output:
                print(
                    f'Epoch: {epoch}, Train loss: {self.train_loss}, Test loss: {self.test_loss}, Test Acc: {self.test_acc}')

    def model_loss(self):
        return self.best_loss

    def train(self, ):
        self.model.train()
        train_loss = 0

        if self.args.graphs:
            if self.model.fc1.weight is not None:
                self.fc1_norm_list.append(torch.norm(self.model.fc1.weight, p=1).detach().cpu().item())
                self.fc2_norm_list.append(torch.norm(self.model.fc2.weight, p=1).detach().cpu().item())

            if hasattr(self.model.fc1, 'weight_align'):
                if self.model.fc1.weight_align is not None:
                    self.wa1_norm_list.append(torch.norm(self.model.fc1.weight_align, p=1).detach().cpu().item())
                    self.wa2_norm_list.append(torch.norm(self.model.fc2.weight_align, p=1).detach().cpu().item())

                else:
                    self.wa1_norm_list.append(None)
                    self.wa2_norm_list.append(None)


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
            loss.backward()
            self.optimizer.step()

            if self.args.graphs:
                if self.model.fc1.weight is not None:
                    if batch_idx in [10,25,50,75]:
                        self.fc1_norm_list.append(torch.norm(self.model.fc1.weight, p=1).detach().cpu().item())
                        self.fc2_norm_list.append(torch.norm(self.model.fc2.weight, p=1).detach().cpu().item())

                if hasattr(self.model.fc1, 'weight_align'):

                    if self.model.fc1.weight_align is not None:
                        if batch_idx in [10,25,50,75]:
                            self.wa1_norm_list.append(torch.norm(self.model.fc1.weight_align, p=1).detach().cpu().item())
                            self.wa2_norm_list.append(torch.norm(self.model.fc2.weight_align, p=1).detach().cpu().item())
                    else:
                        if batch_idx in [10, 25, 50, 75]:
                            self.wa1_norm_list.append(None)
                            self.wa2_norm_list.append(None)

        if self.args.graphs:
            if self.model.fc1.weight is not None:
                    self.fc1_norm_list.append(torch.norm(self.model.fc1.weight, p=1).detach().cpu().item())
                    self.fc2_norm_list.append(torch.norm(self.model.fc2.weight, p=1).detach().cpu().item())

            if hasattr(self.model.fc1, 'weight_align'):
                if self.model.fc1.weight_align is not None:
                        self.wa1_norm_list.append(torch.norm(self.model.fc1.weight_align, p=1).detach().cpu().item())
                        self.wa2_norm_list.append(torch.norm(self.model.fc2.weight_align, p=1).detach().cpu().item())
                else:
                        self.wa1_norm_list.append(None)
                        self.wa2_norm_list.append(None)



        train_loss /= len(self.train_loader.dataset)
        return train_loss

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
            # m1.weight_align=nn.Parameter(m2.weight, requires_grad=True)
            # if args.detach():
            #m2.weight_align = nn.Parameter(m1.weight.clone().detach(), requires_grad=True)
            m2.weight_align = nn.Parameter(m1.weight, requires_grad=True)
            m1.weight_align = nn.Parameter(m2.weight, requires_grad=True)

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
        intra_merge_iterations=[25 for i in range(2)]+[15 for i in range(2)]+[10 for i in range(2)]+[5 for i in range(4)]+[2 for i in range(10)]+[1 for i in range(1000)]

        model1 = Net(self.args, weight_merge=True).to(self.device)
        model2 = Net(self.args, weight_merge=True).to(self.device)
        model1_trainer = Trainer(self.args, [self.train_loader1, self.test_dataset], model1, self.device,
                                 f'{self.weight_dir}model1_0.pt', 'model1_double')
        model2_trainer = Trainer(self.args, [self.train_loader2, self.test_dataset], model2, self.device,
                                 f'{self.weight_dir}model2_0.pt', 'model2_double')

        wd1=[]
        wd2=[]


        #model1_trainer.optimizer = optim.SGD(model1.parameters(),lr=self.args.lr )
        #model2_trainer.optimizer = optim.SGD(model2.parameters(), lr=self.args.lr)
        for iter in range(merge_iterations):
            #model1_trainer=self.train_single(model1, f'{self.weight_dir}model1_{iter}.pt', self.train_loader1,'model1_single')
            #model2_trainer = self.train_single(model2, f'{self.weight_dir}model2_{iter}.pt', self.train_loader2, 'model2_single')

            #model1_trainer.optimizer=optim.Adam(model1.parameters(), lr=self.args.lr)
            #model2_trainer.optimizer=optim.Adam(model2.parameters(), lr=self.args.lr)
            model1_trainer.optimizer = optim.Adadelta(model1.parameters(), )
            model2_trainer.optimizer = optim.Adadelta(model2.parameters(), )

            #print(f'Inter Merge Iterations: {intra_merge_iterations[iter]}')
            #for iter2 in range(intra_merge_iterations[iter]):
            for iter2 in range(1):
                model1_trainer.fit()
                model2_trainer.fit()
                if iter>0:
                    wd1.append(torch.sum((model1_trainer.model.fc1.weight-model2_trainer.model.fc1.weight).abs()).detach().cpu().item())
                    wd2.append(torch.sum((model1_trainer.model.fc2.weight-model2_trainer.model.fc2.weight).abs()).detach().cpu().item())
            #print(wd1)
            #print(wd2)

            if iter==0:
                set_weight_align_param(model1, model2, self.args)



            print(f'Merge Iteration: {iter} \n'
                  f'\tModel 1 Train loss: {model1_trainer.train_loss}, Test loss: {model1_trainer.test_loss},  Test accuracy: {model1_trainer.test_acc}\n'
                  f'\tModel 2 Train loss: {model2_trainer.train_loss}, Test loss: {model2_trainer.test_loss},  Test accuracy: {model2_trainer.test_acc}')


            df=pd.DataFrame({'model1_fc1':model1_trainer.fc1_norm_list,
                             'model1_fc2':model1_trainer.fc2_norm_list,
                             'model2_fc1': model2_trainer.fc1_norm_list,
                             'model2_fc2':model2_trainer.fc2_norm_list,
                             'model2_wa1':model2_trainer.wa1_norm_list,
                             'model2_wa2':model2_trainer.wa2_norm_list,
                             'model1_wa1': model1_trainer.wa1_norm_list,
                             'model1_wa2': model1_trainer.wa2_norm_list
                             })
            df.to_csv('norms/norms_double.csv')

            df=pd.DataFrame({'weight_diff_layer1':wd1,
                             'weight_diff_layer2':wd2})
            df.to_csv('norms/weight_diff_double.csv')

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train')
    parser.add_argument('--merge_iter', type=int, default=20000,
                        help='number of iterations to merge')
    parser.add_argument('--weight_align_factor', type=int, default=250, )
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7,
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--weight_seed', type=int, default=1, )
    parser.add_argument('--gpu', type=int, default=1, )
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--baseline', type=bool, default=False, help='train base model')
    parser.add_argument('--graphs', type=bool, default=False, help='add norm graphs during training')
    parser.add_argument('--base_dir', type=str, default="/s/luffy/b/nobackup/mgorb/",
                        help='Directory for data and weights')
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    weight_dir = f'{args.base_dir}iwa_weights/'
    if args.baseline:
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