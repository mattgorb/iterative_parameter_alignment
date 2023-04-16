from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
import collections

import random
from datasets.dirichlet_partition import dirichlet,  record_net_data_stats

def get_datasets(args):
    # not using normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    if args.baseline:
        dataset1 = datasets.FashionMNIST(f'{args.base_dir}data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(f'{args.base_dir}data', train=False, transform=transform)
        train_loader = DataLoader(dataset1, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        return train_loader, test_loader
    else:

        num_clients=args.num_clients

        dataset1 = datasets.FashionMNIST(f'{args.base_dir}data', train=True, download=True, transform=transform)

        # split dataset in half by labels
        #labels = torch.unique(dataset1.targets)
        train_loaders = []
        if args.dataset_split=="disjoint_classes":
            assert num_clients in [2,5,10]

            if num_clients==2:
                labels_iter=[[0,1,2,3,4],[5,6,7,8,9]]
            elif num_clients==5:
                labels_iter=[[0,1],[2,3],[4,5],[6,7],[8,9]]
            elif num_clients==10:
                labels_iter=[[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]

            print(f'label groupings: {labels_iter}')

            index_groupings=[]
            for label_list in labels_iter:
                index_group=[idx for idx, target in enumerate(dataset1.targets) if target in label_list]
                index_groupings.append(index_group)
            if args.imbalanced:
                if num_clients!=2:
                    print('Clients needs to be 2')
                    sys.exit()
                # use this code for p/1-p split.  need to test
                p = 0.8
                d1 = index_groupings[0][:int(len(index_groupings[0]) * p)] + index_groupings[1][int(len(index_groupings[1]) * p):]
                d2 = index_groupings[0][int(len(index_groupings[0]) * p):] + index_groupings[1][:int(len(index_groupings[1]) * p)]


                dataset1 = datasets.FashionMNIST(f'{args.base_dir}data', train=True, transform=transform)
                dataset2 = datasets.FashionMNIST(f'{args.base_dir}data', train=True, transform=transform)

                dataset1.data, dataset1.targets = dataset1.data[d1], dataset1.targets[d1]
                dataset2.data, dataset2.targets = dataset2.data[d2], dataset2.targets[d2]
                assert (set(d1).isdisjoint(d2))
                train_loader1 = DataLoader(dataset1, batch_size=args.batch_size, shuffle=True)
                train_loader2 = DataLoader(dataset2, batch_size=args.batch_size, shuffle=True)
                train_loaders.append(train_loader1)
                train_loaders.append(train_loader2)
            else:
                for group in index_groupings:
                    dataset = datasets.FashionMNIST(f'{args.base_dir}data', train=True, transform=transform)
                    dataset.data, dataset.targets = dataset.data[group], dataset.targets[group]
                    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
                    train_loaders.append(train_loader)
                if num_clients==2:
                    assert (set(index_groupings[0]).isdisjoint(index_groupings[1]))
                else:
                    #random assertions
                    assert (set(index_groupings[0]).isdisjoint(index_groupings[1]))
                    assert (set(index_groupings[0]).isdisjoint(index_groupings[2]))
                    assert (set(index_groupings[1]).isdisjoint(index_groupings[2]))
                    assert (set(index_groupings[1]).isdisjoint(index_groupings[3]))
                    assert (set(index_groupings[3]).isdisjoint(index_groupings[4]))


        elif args.dataset_split=='iid':
            num_clients = args.num_clients
            lst=np.arange(len(dataset1))
            random.shuffle(lst)
            stats_dict={}
            i=0
            for group in np.array_split(lst, num_clients):
                dataset = datasets.FashionMNIST(f'{args.base_dir}data', train=True, transform=transform)
                dataset.data, dataset.targets = dataset.data[group], dataset.targets[group]
                train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
                train_loaders.append(train_loader)
                stats_dict[i]=group
                i+=1

            dataset = datasets.FashionMNIST(f'{args.base_dir}data', train=True, transform=transform)
            record_net_data_stats_iid(dataset.targets, stats_dict, args)

        elif args.dataset_split=='dirichlet':
            X_train, y_train = dataset1.data, dataset1.targets
            net_dataidx_map,y_train=dirichlet(y_train,  num_clients, alpha=args.dirichlet_alpha)
            record_net_data_stats(y_train, net_dataidx_map, args)

            for i,j in net_dataidx_map.items():
                dataset = datasets.FashionMNIST(f'{args.base_dir}data', train=True, transform=transform)
                dataset.data, dataset.targets = dataset.data[j], dataset.targets[j]
                #print(len(dataset.data))
                train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
                train_loaders.append(train_loader)

        else:
            print('Choose valid dataset partition')
            sys.exit()


        test_dataset = datasets.FashionMNIST(f'{args.base_dir}data', train=False, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


        print('Dataset summaries:')

        for i in range(len(train_loaders)):
            print(f'Train set {i}: Length: {len(train_loaders[i].dataset)}, Labels: {collections.Counter(train_loaders[i].dataset.targets.tolist())}')

        print(f'Test set: Length: {len(test_loader.dataset)}, Labels: {collections.Counter(test_loader.dataset.targets.tolist())}')

        weights=[]
        dataset_len = len(datasets.FashionMNIST(f'{args.base_dir}data', train=True,).targets)
        for loader in train_loaders:
            #original balance
            weights.append(len(loader.dataset.targets)/dataset_len)



            #cls_counter=collections.Counter(loader.dataset.targets.tolist())
            #cls_cnt_ls=list(cls_counter.values())+[0 for i in range(10-len(list(cls_counter.values())))]
            #cls_cnt_ls_pct=[i/sum(cls_cnt_ls) for i in cls_cnt_ls]

            #(n*√d-1)/(√d-1), n is l2 norm of class list
            #s_uni = (1- (np.linalg.norm(cls_cnt_ls_pct)*np.sqrt(10)-1)/(np.sqrt(10)-1))+1e-7

        print(weights)
        return train_loaders, test_loader, weights
