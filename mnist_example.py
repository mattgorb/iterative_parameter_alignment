from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import  DataLoader
import torch.autograd as autograd
import math
import random

class GetSubnetSTE(autograd.Function):
    @staticmethod
    def forward(ctx, scores,):
        # Get the subnetwork by sorting the scores and using the top k%
        return (scores>0).float()

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def linear_init(in_dim, out_dim, bias=None, args=None,):
    layer=SubnetLinear(in_dim,out_dim, bias)
    layer.init(args)
    return layer

# Not learning weights, finding subnet
class SubnetLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))

    def init(self,args):
        self.args=args
        set_seed(self.args.weight_seed)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        set_seed(self.args.score_seed)
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def forward(self, x):
        subnet = GetSubnetSTE.apply(self.scores, )
        w = self.weight * subnet
        x= F.linear(x, w, self.bias)
        return x


class Net(nn.Module):
    def __init__(self,args, sparse=False):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return x

'''class Net(nn.Module):
    def __init__(self,args, sparse=False):
        super(Net, self).__init__()
        self.args=args
        if sparse:
            self.fc1 = linear_init(28*28, 128, bias=None, args=self.args, )
            self.fc2 = linear_init(128, 128, bias=None, args=self.args, )
            self.fc3 = linear_init(128, 10, bias=None, args=self.args, )
        else:
            self.fc1 = nn.Linear(28*28, 128)
            self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        x = self.fc1(x.view(x.size(0),-1))
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x'''

def freeze_model_weights(model):
    print("=> Freezing model weights")
    for n, m in model.named_modules():
        if not type(m)==SubnetLinear:
            continue
        if hasattr(m, "weight") and m.weight is not None:
            print(f"==> No gradient to {n}.weight")
            m.weight.requires_grad = False
            if m.weight.grad is not None:
                print(f"==> Setting gradient of {n}.weight to None")
                m.weight.grad = None

            if hasattr(m, "bias") and m.bias is not None:
                print(f"==> No gradient to {n}.bias")
                m.bias.requires_grad = False

                if m.bias.grad is not None:
                    print(f"==> Setting gradient of {n}.bias to None")
                    m.bias.grad = None

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

    dataset1 = datasets.MNIST(f'{args.base_dir}data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST(f'{args.base_dir}data', train=True, transform=transform)

    #split dataset in half
    labels=torch.unique(dataset1.targets)
    ds1_labels=labels[:len(labels)//2]
    ds2_labels=labels[len(labels)//2:]
    print(f'ds1_labels: {ds1_labels}')
    print(f'ds2_labels: {ds2_labels}')
    ds1_indices = [idx for idx, target in enumerate(dataset1.targets) if target in ds1_labels]
    ds2_indices = [idx for idx, target in enumerate(dataset1.targets) if target in ds2_labels]
    dataset1.data, dataset1.targets = dataset1.data[ds1_indices], dataset1.targets[ds1_indices]
    dataset2.data, dataset2.targets = dataset2.data[ds2_indices], dataset2.targets[ds2_indices]
    assert(ds1_indices.isdisjoint(ds2_indices))

    test_dataset = datasets.MNIST(f'{args.base_dir}data', train=False,  transform=transform)
    train_loader1 = DataLoader(dataset1,batch_size=args.batch_size, shuffle=True)
    train_loader2 = DataLoader(dataset2,batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=args.batch_size, shuffle=False)

    return train_loader1, train_loader2, test_loader

class Trainer:
    def __init__(self, args,datasets, model, device, save_path):
        self.args = args
        self.model = model
        self.train_loader, self.test_loader=datasets[0],datasets[1]
        self.optimizer= optim.Adam(self.model.parameters(), lr=args.lr)
        self.criterion=nn.CrossEntropyLoss(reduction='sum')
        self.device=device
        self.save_path=save_path

    def fit(self):
        self.best_loss=1e6
        print('Fitting model...')
        for epoch in range(1, self.args.epochs + 1):
            print(f'Epoch {epoch}')
            train_loss = self.train()
            self.test()
            if train_loss<self.best_loss:
                self.best_loss=train_loss
                print(f'Saving model with train loss {train_loss}')
                torch.save(self.model.state_dict(), self.save_path)

    def model_loss(self):
        return self.best_loss

    def train(self,):
        self.model.train()
        train_loss=0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)

            #loss = self.criterion(output, target)
            loss = F.nll_loss(output, target)
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
                output = self.model(data)
                test_loss += self.criterion(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

class MLC_Iterator:
    def __init__(self, args,datasets, device,):
        self.args=args
        self.device=device
        self.train_loader1 = datasets[0]
        self.train_loader2 = datasets[1]
        self.test_dataset = datasets[2]

    def train_single(self, model,model_name, mlc_iter,train_dataset):
        freeze_model_weights(model)
        save_path = f'{self.args.weight_dir}{model_name}_{mlc_iter}.pt'
        trainer = Trainer(self.args, [train_dataset, self.test_dataset], model, self.device, save_path)
        trainer.fit()
        return trainer.best_loss

    def run(self):
        mlc_iterations=20
        epsilon=1e-2

        results_dict={}

        for iter in range(mlc_iterations):
            model1 = Net(self.args, sparse=True).to(self.device)
            model_1_loss=self.train_single(model1, "model_1", iter, self.train_loader1)

            model2 = Net(self.args, sparse=True).to(self.device)
            model_2_loss=self.train_single(model2, "model_2", iter, self.train_loader2)



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--weight_seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--score_seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--gpu', type=int, default=1, metavar='S',
                        help='which gpu to use')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--baseline', type=bool,default=False,help='train base model')
    parser.add_argument('--randinit_baseline', type=bool,default=False,help='train subnetwork with randomly initialized weights')
    parser.add_argument('--base_dir', type=str,default="/s/luffy/b/nobackup/mgorb/",help='Directory for data and weights')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    weight_dir=f'{args.base_dir}mlc_weights/'
    if args.baseline:
        train_loader1, test_dataset = get_datasets(args)
        if args.randinit_baseline:
            model = Net(args, sparse=True).to(device)
            save_path=f'{weight_dir}mnist_ri_subnetwork_baseline.pt'
        else:
            model = Net(args, sparse=False).to(device)
            save_path=f'{weight_dir}mnist_baseline.pt'
        freeze_model_weights(model)
        trainer=Trainer(args,[train_loader1, test_dataset], model, device, save_path)
        trainer.fit()
    else:
        train_loader1, train_loader2, test_dataset=get_datasets(args)
        mlc_iterator=MLC_Iterator(args,[train_loader1,train_loader2,test_dataset], device,)


if __name__ == '__main__':
    main()