import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import random

import numpy as np

device='cuda'
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class Net(nn.Module):
    def __init__(self, args,):
        super(Net, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(28 * 28, 128, bias=True)
        self.fc2 = nn.Linear(128, 10, bias=True)
        set_seed(self.args.seed)

        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc2.weight, mode="fan_in", nonlinearity="relu")

    def fc1_out(self, x):
        x = self.fc1(x.view(-1, 28 * 28))
        x=F.relu(x)
        return x


    def forward(self, x, ):
        x = self.fc1(x.view(-1, 28 * 28))
        x=F.relu(x)
        x=self.fc2(x)
        return x

def get_datasets(args):
    # not using normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

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
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        device = device
        self.save_path = save_path
        self.model_name = model_name

    def fit(self, log_output=True):
        self.train_loss = 1e6
        for epoch in range(1, self.args.epochs + 1):
            epoch_loss = self.train()
            self.train_loss = epoch_loss
            test_loss, test_acc = self.test()
            self.test_loss = test_loss
            self.test_acc = test_acc

            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, self.save_path)

            if log_output:
                print(f'Epoch: {epoch}, Train loss: {self.train_loss}, Test loss: {self.test_loss}, Test Acc: {self.test_acc}')


    def train(self, ):
        self.model.train()
        train_loss = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(device), target.to(device)
            self.optimizer.zero_grad()
            output = self.model(data)

            loss = self.criterion(output, target)

            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        train_loss /= len(self.train_loader.dataset)
        return train_loss

    def test(self, ):
        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data, )
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.test_loader.dataset)
        return test_loss, 100. * correct / len(self.test_loader.dataset)

def test(model, device,test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, )
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_acc=(100. * correct / len(test_loader.dataset))
    print(f'Test accuracy:  {test_acc}')

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Weight Align')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs to train')

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1.0)')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--gpu', type=int, default=6, )

    parser.add_argument('--base_dir', type=str, default="/s/luffy/b/nobackup/mgorb/",
                        help='Directory for data and weights')
    args = parser.parse_args()
    set_seed(args.seed)
    #device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    weight_dir = f'{args.base_dir}adversarial_merge_weights/'



    train_loader1, train_loader2, test_dataset = get_datasets(args)


    model1 = Net(args, ).to(device)
    save_path = f'{weight_dir}mnist_model1_2.pt'
    #trainer1 = Trainer(args, [train_loader1, test_dataset], model1, device, save_path, 'mnist_model1')



    #args.seed+=1


    model2 = Net(args, ).to(device)
    save_path = f'{weight_dir}mnist_model2_2.pt'
    #trainer2 = Trainer(args, [train_loader2, test_dataset], model2, device, save_path, 'mnist_model2')

    print(model1.fc1.weight[0][:10])
    print(model2.fc1.weight[0][:10])
    sys.exit()

    model_merge = Net(args, ).to(device)
    optim_merge = optim.Adam(model_merge.parameters(), lr=args.lr)




    for i in range(5):
        print(f'Iteration, activation alignment: {i}')
        print(f"Training model 1")
        trainer1 = Trainer(args, [train_loader1, test_dataset], model1, device, save_path, 'mnist_model1')
        trainer1.fit(log_output=True)

        print(f"Training model 2")
        trainer2 = Trainer(args, [train_loader2, test_dataset], model2, device, save_path, 'mnist_model2')
        trainer2.fit(log_output=True)


        model1.eval()
        model2.eval()



        model_merge.fc1.weight=torch.nn.Parameter(torch.mean(torch.stack([model1.fc1.weight, model2.fc1.weight], dim=0), dim=0), requires_grad=False)
        model_merge.fc1.bias=torch.nn.Parameter(torch.mean(torch.stack([model1.fc1.bias, model2.fc1.bias], dim=0), dim=0), requires_grad=False)

        model_merge.fc2.weight=torch.nn.Parameter(torch.mean(torch.stack([model1.fc2.weight, model2.fc2.weight], dim=0), dim=0), requires_grad=False)
        model_merge.fc2.bias=torch.nn.Parameter(torch.mean(torch.stack([model1.fc2.bias, model2.fc2.bias], dim=0), dim=0), requires_grad=False)


        model_merge.eval()
        test(model_merge, device, test_dataset)

        model1.fc1.weight=nn.Parameter(model_merge.fc1.weight.clone().detach().to(device), requires_grad=True)
        model1.fc1.bias=nn.Parameter(model_merge.fc1.bias.clone().detach().to(device), requires_grad=True)
        model2.fc1.weight=nn.Parameter(model_merge.fc1.weight.clone().detach().to(device), requires_grad=True)
        model2.fc1.bias=nn.Parameter(model_merge.fc1.bias.clone().detach().to(device), requires_grad=True)

        model1.fc2.weight=nn.Parameter(model_merge.fc2.weight.clone().detach().to(device), requires_grad=True)
        model1.fc2.bias=nn.Parameter(model_merge.fc2.bias.clone().detach().to(device), requires_grad=True)
        model2.fc2.weight=nn.Parameter(model_merge.fc2.weight.clone().detach().to(device), requires_grad=True)
        model2.fc2.bias=nn.Parameter(model_merge.fc2.bias.clone().detach().to(device), requires_grad=True)




if __name__ == '__main__':
    main()