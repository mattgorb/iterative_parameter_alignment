from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
import collections

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

        num_clients=args.num_clients
        dataset1 = datasets.MNIST(f'{args.base_dir}data', train=True, download=True, transform=transform)

        # split dataset in half by labels
        #labels = torch.unique(dataset1.targets)
        train_loaders = []
        if args.disjoint_classes:
            assert num_clients in [2,5]
            #ds1_labels = labels[:len(labels) // num_clients]
            #ds2_labels = labels[len(labels) // num_clients:]

            if num_clients==2:
                labels_iter=[[0,1,2,3,4],[5,6,7,8,9]]
            else:
                labels_iter=[[0,1],[2,3],[4,5],[6,7],[8,9]]
            print(f'label groupings: {labels_iter}')

            index_groupings=[]
            for label_list in labels_iter:
                index_group=[idx for idx, target in enumerate(dataset1.targets) if target in label_list]
                index_groupings.append(index_group)
            if args.imbalanced:
                assert num_clients.equals(2)
                # use this code for p/1-p split.  need to test
                p = 0.8
                d1 = index_groupings[0][:int(len(index_groupings[0]) * p)] + index_groupings[1][int(len(index_groupings[1]) * p):]
                d2 = index_groupings[0][int(len(index_groupings[0]) * p):] + index_groupings[1][:int(len(index_groupings[1]) * p)]
                dataset1 = datasets.MNIST(f'{args.base_dir}data', train=True, transform=transform)
                dataset2 = datasets.MNIST(f'{args.base_dir}data', train=True, transform=transform)

                dataset1.data, dataset1.targets = dataset1.data[d1], list(np.array(dataset1.targets)[d1])
                dataset2.data, dataset2.targets = dataset2.data[d2], list(np.array(dataset2.targets)[d2])
                assert (set(d1).isdisjoint(d2))
                train_loader1 = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
                train_loader2 = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
                train_loaders.append(train_loader1)
                train_loaders.append(train_loader2)
            else:
                for group in index_groupings:
                    dataset = datasets.MNIST(f'{args.base_dir}data', train=True, transform=transform)
                    dataset.data, dataset.targets = dataset.data[group], dataset.targets[group]
                    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
                    train_loaders.append(train_loader)
                #dataset1.data, dataset1.targets = dataset1.data[index_groupings[0]], dataset1.targets[index_groupings[0]]
                #dataset2.data, dataset2.targets = dataset1.data[index_groupings[1]], dataset1.targets[index_groupings[1]]
                if num_clients==2:
                    assert (set(index_groupings[0]).isdisjoint(index_groupings[1]))
                else:
                    assert (set(index_groupings[0]).isdisjoint(index_groupings[1]).isdisjoint(index_groupings[2]).isdisjoint(index_groupings[3]).isdisjoint(index_groupings[4]))


        else:

            np.array_split(lst, 5)

        test_dataset = datasets.MNIST(f'{args.base_dir}data', train=False, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


        print('Dataset summaries:')

        for i in range(len(train_loaders)):
            #print(f'Train set {i}: Length: {len(train_loaders[i].dataset)}, Labels: {collections.Counter(train_loaders[i].dataset.targets)}')
            print(train_loaders[i].dataset.targets.tolist())
            sys.exit()

        print(f'Test set: Length: {len(test_loader.dataset)}, Labels: {collections.Counter(test_loader.dataset.targets)}')
        sys.exit()
        return train_loaders, test_loader
