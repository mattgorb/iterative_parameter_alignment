from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import  DataLoader
import math
import random
from torch.optim.lr_scheduler import CosineAnnealingLR


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def linear_init(in_dim, out_dim, bias=False, args=None,):
    layer=LinearMerge(in_dim,out_dim,bias=bias)
    layer.init(args)
    return layer

class LinearMerge(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_align = None

    def init(self,args):
        self.args=args
        set_seed(self.args.weight_seed)
        #this isn't default initialization.  not sure if necessary, need to test.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

    #not sure if reinitializing weights helps convergence.  need to test.
    def reset_weights(self,):
        self.args.weight_seed+=1
        set_seed(self.args.weight_seed)
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        x = F.linear(x, self.weight, self.bias)
        weights_diff=None
        if self.weight_align is not None:
            #using absolute error here.  Need to test using MSE loss
            weights_diff=torch.sum((self.weight-self.weight_align).abs())
        return x, weights_diff


class Net(nn.Module):
    def __init__(self,args, weight_merge=False):
        super(Net, self).__init__()
        self.args=args
        self.weight_merge=weight_merge

        #bias False for now, have not tested adding bias to the loss fn.
        if self.weight_merge:
            self.fc1 = linear_init(28*28, 1024, bias=False, args=self.args, )
            self.fc2 = linear_init(1024, 10, bias=False, args=self.args, )
        else:
            self.fc1 = nn.Linear(28*28, 1024, bias=False)
            self.fc2 = nn.Linear(1024, 10, bias=False)
    def forward(self, x, ):
        if self.weight_merge:
            x,sd1 = self.fc1(x.view(-1, 28*28))
            x = F.relu(x)
            x,sd2= self.fc2(x)
            if sd1 is not None:
                score_diff=sd1+sd2
            else:
                score_diff=torch.tensor(0)
            return x, score_diff
        else:
            x = self.fc1(x.view(-1, 28*28))
            x = F.relu(x)
            x= self.fc2(x)

            return x, torch.tensor(0)


def get_datasets(args):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
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

        #split dataset in half by labels
        labels=torch.unique(dataset1.targets)
        ds1_labels=labels[:len(labels)//2]
        ds2_labels=labels[len(labels)//2:]
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
        assert(set(ds1_indices).isdisjoint(ds2_indices))


        test_dataset = datasets.MNIST(f'{args.base_dir}data', train=False,  transform=transform)
        train_loader1 = DataLoader(dataset1,batch_size=args.batch_size, shuffle=True)
        train_loader2 = DataLoader(dataset2,batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset,batch_size=args.batch_size, shuffle=False)

        return train_loader1, train_loader2, test_loader

class Trainer:
    def __init__(self, args,datasets, model, device, save_path, epochs):
        self.args = args
        self.model = model
        self.train_loader, self.test_loader=datasets[0],datasets[1]
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion=nn.CrossEntropyLoss(reduction='mean')
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.device=device
        self.save_path=save_path
        self.epochs=epochs

    def fit(self):
        self.best_loss=1e6
        for epoch in range(1, self.epochs + 1):
            print(f'Epoch {epoch}')
            train_loss = self.train()
            if train_loss<self.best_loss:
                self.best_loss=train_loss
                print(f'Saving model with train loss {train_loss}')
                torch.save(self.model.state_dict(), self.save_path)
                self.test()
            self.scheduler.step()

    def model_loss(self):
        return self.best_loss

    def train(self,):
        self.model.train()
        train_loss=0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output, sd = self.model(data)

            '''
            5000 works for this particular combination, summing both CrossEntropyLoss and weight alignment
            Taking mean of CE and sum of weight alignment might make this hyperparameter unnecessary.
            For model w/o weight alignment paramter, second part of loss is 0
            When 5000 isn't used, the model converges too quickly towards second models dataset.  
            '''
            loss = self.criterion(output, target)+2*sd#5000*sd
            train_loss+=loss
            loss.backward()
            self.optimizer.step()
        train_loss /= len(self.train_loader.dataset)
        return train_loss

    def test(self,):
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

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))


def set_weight_align_param(model1, model2,):
    print("=> Generating MLC mask")
    for model1_mods, model2_mods, in zip(model1.named_modules(), model2.named_modules(),):
        n1,m1=model1_mods
        n2,m2=model2_mods
        if not type(m2)==LinearMerge:
            continue
        if hasattr(m1, "weight"):
            '''
            m1.weight gets updated to m2.weight_align because it is not detached.  
            This is a simple way to "share" the weights between models. 
            Alternatively we could set m1.weight=m2.weight_align after merge model is done training.  
            '''
            m2.weight_align = nn.Parameter(m1.weight, requires_grad=True)
            #m2.reset_weights()

class Merge_Iterator:
    def __init__(self, args,datasets, device,weight_dir):
        self.args=args
        self.device=device
        self.weight_dir=weight_dir
        self.train_loader1 = datasets[0]
        self.train_loader2 = datasets[1]
        self.test_dataset = datasets[2]

    def train_single(self, model,save_path, train_dataset,epochs ):
        #not sure if we need to initialize a new training optimizer etc. during each iteration.  need to test.
        trainer = Trainer(self.args, [train_dataset, self.test_dataset], model, self.device, save_path, epochs)
        trainer.fit()
        return trainer

    def run(self):
        merge_iterations=20
        for iter in range(merge_iterations):

            if iter==0:
                reference_model = Net(self.args, weight_merge=False ).to(self.device)
                merge_model = Net(self.args, weight_merge=True).to(self.device)

            print(f"Merge Iterator: {iter}, training model 1")
            self.train_single(reference_model, f'{self.weight_dir}reference_model_{iter}.pt', self.train_loader1, self.args.ref_epochs)
            set_weight_align_param(reference_model, merge_model,)
            print(f"Merge Iterator: {iter}, training model 2")
            self.train_single(merge_model, f'{self.weight_dir}model_2_{iter}.pt' ,self.train_loader2,self.args.merge_epochs)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--ref_epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--merge_epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--weight_seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--gpu', type=int, default=1, metavar='S',
                        help='which gpu to use')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--baseline', type=bool,default=False,help='train base model')
    parser.add_argument('--base_dir', type=str,default="/s/luffy/b/nobackup/mgorb/",help='Directory for data and weights')
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    weight_dir=f'{args.base_dir}mlc_weights/'
    if args.baseline:
        train_loader1, test_dataset = get_datasets(args)
        model = Net(args, sparse=False).to(device)
        save_path=f'{weight_dir}mnist_baseline.pt'
        trainer=Trainer(args,[train_loader1, test_dataset], model, device, save_path)
        trainer.fit()
    else:
        train_loader1, train_loader2, test_dataset=get_datasets(args)
        merge_iterator=Merge_Iterator(args,[train_loader1,train_loader2,test_dataset], device,weight_dir)
        merge_iterator.run()

if __name__ == '__main__':

    main()