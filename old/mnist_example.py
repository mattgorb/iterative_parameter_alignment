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
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR



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
    layer=SubnetLinear(in_dim,out_dim,bias=bias)
    layer.init(args)
    return layer

def conv_init(in_channels, out_channels, kernel_size, stride,  args=None,):
    layer=SubnetConv(in_channels, out_channels, kernel_size, stride,bias=False)
    layer.init(args)
    return layer

# Not learning weights, finding subnet
class SubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        self.mlc_mask = None

    def init(self,args):
        self.args=args
        set_seed(self.args.weight_seed)
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")
        set_seed(self.args.score_seed)
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def get_subnet(self):
        subnet = GetSubnetSTE.apply(self.scores, )
        if self.mlc_mask is not None:
            subnet=torch.where(self.mlc_mask==-1, subnet, self.mlc_mask)
        return subnet

    def forward(self, x):
        subnet = GetSubnetSTE.apply(self.scores, )
        if self.mlc_mask is not None:
            subnet=torch.where(self.mlc_mask==-1, subnet, self.mlc_mask)
        w = self.weight * subnet
        x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

# Not learning weights, finding subnet
class SubnetLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        self.mlc_mask = None
    def init(self,args):
        self.args=args
        set_seed(self.args.weight_seed)
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")
        set_seed(self.args.score_seed)
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def get_subnet(self):
        subnet = GetSubnetSTE.apply(self.scores, )
        if self.mlc_mask is not None:
            subnet=torch.where(self.mlc_mask==-1, subnet, self.mlc_mask)
        return subnet

    def forward(self, x):
        subnet = GetSubnetSTE.apply(self.scores, )
        if self.mlc_mask is not None:
            subnet=torch.where(self.mlc_mask==-1, subnet, self.mlc_mask)
        w = self.weight * subnet
        x= F.linear(x, w, self.bias)
        return x


class Net(nn.Module):
    def __init__(self,args, sparse=False):
        super(Net, self).__init__()
        self.args=args
        self.sparse=sparse
        if self.sparse:
            self.conv1 = conv_init(1,32,3,1, args=self.args, )
            self.conv2 = conv_init(32, 64, 3, 1, args=self.args, )
            self.fc1 = linear_init(9216, 128, bias=None, args=self.args, )
            self.fc2 = linear_init(128, 10, bias=None, args=self.args, )
        else:
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def assert_model_weight_equality(model1, model2, mlc_mask=False):
    print("=> Freezing model weights")
    for model1_mods, model2_mods in zip(model1.named_modules(), model2.named_modules()):
        n1,m1=model1_mods
        n2,m2=model2_mods
        if not type(m1)==SubnetLinear and not type(m1)==SubnetConv:
            continue
        if hasattr(m1, "weight") and m1.weight is not None:
            assert(torch.equal(m1.weight,m2.weight))
            if mlc_mask:
                assert(torch.equal(m1.mlc_mask, m2.mlc_mask))
            if hasattr(m1, "bias") and m1.bias is not None:
                assert(torch.equal(m1.bias,m2.bias))




def freeze_model_weights(model):
    print("=> Freezing model weights")
    for n, m in model.named_modules():
        if not type(m)==SubnetLinear and not type(m)==SubnetConv:
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
    else:
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

        #p/1-p split
        #p=0.9
        #ds1_indices=ds1_indices[:int(len(ds1_indices)*p)]+ds2_indices[int(len(ds2_indices)*p):]
        #ds2_indices=ds1_indices[int(len(ds1_indices)*p):]+ds2_indices[:int(len(ds2_indices)*p)]

        dataset1.data, dataset1.targets = dataset1.data[ds1_indices], dataset1.targets[ds1_indices]
        dataset2.data, dataset2.targets = dataset2.data[ds2_indices], dataset2.targets[ds2_indices]
        #assert(set(ds1_indices).isdisjoint(ds2_indices)) #turning off for 80/20 split

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
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        #self.optimizer = optim.SGD([p for p in self.model.parameters() if p.requires_grad],lr=0.1,  momentum=0.9, weight_decay=0.0005, )

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.epochs)

        self.criterion=nn.CrossEntropyLoss(reduction='sum')
        self.device=device
        self.save_path=save_path

    def fit(self):
        self.best_loss=1e6
        print('Fitting model...')
        for epoch in range(1, self.args.epochs + 1):
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
            output = self.model(data)
            loss = self.criterion(output, target)
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


def generate_mlc(model1, model2, model_new):
    print("=> Generating MLC mask")
    for model1_mods, model2_mods, new_model_mods in zip(model1.named_modules(), model2.named_modules(), model_new.named_modules()):
        n1,m1=model1_mods
        n2,m2=model2_mods
        n_new, m_new=new_model_mods
        if not type(m1)==SubnetLinear and not type(m1)==SubnetConv:
            continue
        if hasattr(m1, "weight") and m1.weight is not None:
            assert(torch.equal(m1.weight,m2.weight))
            m1_mask=m1.get_subnet()
            m2_mask=m2.get_subnet()
            mlc=(m1_mask.bool()==m2_mask.bool()).float()

            mlc_mask=torch.ones_like(m1.weight) * -1

            '''
            new logic: keep most important scores from each subnetwork.  
            but override these values where there is a matching linear codimension.  
            '''
            '''k=int(m1.scores.numel()*0.99)
            _, idx1 = m1.scores.abs().flatten().sort()
            _, idx2 = m2.scores.abs().flatten().sort()
            mlc_mask.flatten()[idx1[k:]]=m1.scores.flatten()[idx1[k:]]
            mlc_mask.flatten()[idx2[k:]]=m1.scores.flatten()[idx2[k:]]'''

            mlc_mask=torch.where(mlc==1, m1_mask, mlc_mask)

            m_new.mlc_mask=nn.Parameter(mlc_mask, requires_grad=False)
            #m1.mlc_mask=nn.Parameter(mlc_mask, requires_grad=False)
            #m2.mlc_mask=nn.Parameter(mlc_mask, requires_grad=False)
            print(f'Module: {n_new} matching masks: {int(torch.sum(mlc))}/{torch.numel(mlc)}, %: {int(torch.sum(mlc))/torch.numel(mlc)}')
    return model_new


class MLC_Iterator:
    def __init__(self, args,datasets, device,weight_dir):
        self.args=args
        self.device=device
        self.weight_dir=weight_dir
        self.train_loader1 = datasets[0]
        self.train_loader2 = datasets[1]
        self.test_dataset = datasets[2]

    def train_single(self, model,save_path, train_dataset):
        freeze_model_weights(model)

        trainer = Trainer(self.args, [train_dataset, self.test_dataset], model, self.device, save_path)
        trainer.fit()
        return trainer

    def run(self):
        mlc_iterations=50
        epsilon=1e-2

        results_dict={}

        for iter in range(mlc_iterations):
            if iter==0:
                model1 = Net(self.args, sparse=True).to(self.device)
                model2 = Net(self.args, sparse=True).to(self.device)
                assert_model_weight_equality(model1, model2, mlc_mask=False)
            else:
                model1=copy.deepcopy(model_new)
                model2=copy.deepcopy(model_new)
                assert_model_weight_equality(model1, model2, mlc_mask=True)
                assert_model_weight_equality(model1, results_dict[f'model_1_{iter - 1}'].model)


            print(f"MLC Iterator: {iter}, training model 1")
            model_1_trainer=self.train_single(model1, f'{self.weight_dir}model_1_{iter}.pt', self.train_loader1)
            print(f"MLC Iterator: {iter}, training model 2")
            model_2_trainer=self.train_single(model2, f'{self.weight_dir}model_2_{iter}.pt' ,self.train_loader2)
            results_dict[f'model_1_{iter}']=model_1_trainer
            results_dict[f'model_2_{iter}']=model_2_trainer

            self.args.score_seed+=1
            model_new = Net(self.args, sparse=True).to(self.device)
            model_new=generate_mlc(model1, model2, model_new)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
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
        mlc_iterator=MLC_Iterator(args,[train_loader1,train_loader2,test_dataset], device,weight_dir)
        mlc_iterator.run()

if __name__ == '__main__':
    main()