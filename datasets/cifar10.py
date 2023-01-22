from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

def get_datasets(args):
    # not using normalization
    #transform = transforms.Compose([
        #transforms.ToTensor(),
    #])
    normalize = transforms.Normalize(
        mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    if args.data_transform:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        train_transform=test_transform
    if args.baseline:
        dataset1 = datasets.CIFAR10(f'{args.base_dir}data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(f'{args.base_dir}data', train=False, transform=test_transform)
        train_loader = DataLoader(dataset1, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        return train_loader, test_loader
    else:
        dataset1 = datasets.CIFAR10(f'{args.base_dir}data', train=True, download=True, transform=train_transform)
        #dataset2 = datasets.CIFAR10(f'{args.base_dir}data', train=True, transform=train_transform)
        # split dataset in half by labels
        labels = np.unique(dataset1.targets)
        ds1_labels = labels[:len(labels) // 2]
        ds2_labels = labels[len(labels) // 2:]
        print(f'ds1_labels: {ds1_labels}')
        print(f'ds2_labels: {ds2_labels}')

        ds1_indices = [idx for idx, target in enumerate(dataset1.targets) if target in ds1_labels]
        ds2_indices = [idx for idx, target in enumerate(dataset1.targets) if target in ds2_labels]
        '''print('herre')
        print(ds1_indices[:100])
        print(ds2_indices[:100])
        print('here')
        print(ds1_labels)
        print(ds2_labels)'''

        if args.imbalanced:
            #use this code for p/1-p split.  need to test
            p=0.75
            ds1_indices=ds1_indices[:int(len(ds1_indices)*p)]+ds2_indices[int(len(ds2_indices)*p):]
            ds2_indices=ds1_indices[int(len(ds1_indices)*p):]+ds2_indices[:int(len(ds2_indices)*p)]

        '''print(ds1_indices[:100])
        print(ds2_indices[:100])
        sys.exit()'''
        '''
        #use this code to split dataset down middle. need to test
        dataset1.data, dataset1.targets = dataset1.data[:int(len(dataset1.targets)/2)], dataset1.targets[:int(len(dataset1.targets)/2)]
        dataset2.data, dataset2.targets = dataset2.data[int(len(dataset1.targets)/2):], dataset2.targets[int(len(dataset1.targets)/2):]
        '''

        dataset1.data,dataset1.targets = dataset1.data[ds1_indices],list(np.array(dataset1.targets)[ds1_indices])
        dataset2.data, dataset2.targets = dataset2.data[ds2_indices], list(np.array(dataset2.targets)[ds2_indices])
        #assert (set(ds1_indices).isdisjoint(ds2_indices))

        test_dataset = datasets.CIFAR10(f'{args.base_dir}data', train=False, transform=test_transform)
        train_loader1 = DataLoader(dataset1, batch_size=args.batch_size, shuffle=True)
        train_loader2 = DataLoader(dataset2, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        return train_loader1, train_loader2, test_loader
