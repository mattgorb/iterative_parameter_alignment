from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import torch

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
