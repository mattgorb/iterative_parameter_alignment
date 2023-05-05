from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import collections
import random

from datasets.dirichlet_partition import dirichlet,  record_net_data_stats, record_net_data_stats_iid

def get_datasets(args):
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

        dataset1 = datasets.CIFAR10(f'{args.base_dir}{args.data_dir}', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(f'{args.base_dir}{args.data_dir}', train=False, transform=test_transform)
        train_loader = DataLoader(dataset1, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        return train_loader, test_loader

    else:
        num_clients = args.num_clients
        dataset1 = datasets.CIFAR10(f'{args.base_dir}{args.data_dir}', train=True, download=True, transform=train_transform)

        train_loaders = []
        if args.dataset_split=="disjoint_classes":
            assert num_clients in [2,3, 5, 10]

            if num_clients == 2:
                labels_iter = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
            elif num_clients == 3:
                labels_iter = [[0,1,2,3],[4,5,6],[7,8,9]]
            elif num_clients == 5:
                labels_iter = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
            elif num_clients == 10:
                labels_iter = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]

            if num_clients == 2 and args.uneven=='91':
                labels_iter = [[0, 1, 2, 3, 4, 5, 6, 7, 8, ], [9]]
            if num_clients == 2 and args.uneven=='82':
                labels_iter = [[0, 1, 2, 3, 4, 5, 6, 7,], [8, 9]]
            if num_clients == 2 and args.uneven=='73':
                labels_iter = [[0, 1, 2, 3, 4, 5, 6,], [7,8, 9]]
            print(f'label groupings: {labels_iter}')

            index_groupings = []
            for label_list in labels_iter:
                index_group = [idx for idx, target in enumerate(dataset1.targets) if target in label_list]
                index_groupings.append(index_group)
            if args.imbalanced:
                if num_clients != 2:
                    print('Clients needs to be 2')
                    sys.exit()
                # use this code for p/1-p split.  need to test
                p = 0.8
                d1 = index_groupings[0][:int(len(index_groupings[0]) * p)] + index_groupings[1][
                                                                             int(len(index_groupings[1]) * p):]
                d2 = index_groupings[0][int(len(index_groupings[0]) * p):] + index_groupings[1][
                                                                             :int(len(index_groupings[1]) * p)]

                dataset1 = datasets.CIFAR10(f'{args.base_dir}{args.data_dir}', train=True, transform=train_transform)
                dataset2 = datasets.CIFAR10(f'{args.base_dir}{args.data_dir}', train=True, transform=train_transform)

                dataset1.data, dataset1.targets = dataset1.data[d1], np.array(dataset1.targets)[d1]
                dataset2.data, dataset2.targets = dataset2.data[d2], np.array(dataset2.targets)[d2]
                assert (set(d1).isdisjoint(d2))
                train_loader1 = DataLoader(dataset1, batch_size=args.batch_size, shuffle=True)
                train_loader2 = DataLoader(dataset2, batch_size=args.batch_size, shuffle=True)
                train_loaders.append(train_loader1)
                train_loaders.append(train_loader2)
            else:
                for group in index_groupings:
                    dataset = datasets.CIFAR10(f'{args.base_dir}{args.data_dir}', train=True, transform=train_transform)
                    dataset.data, dataset.targets = dataset.data[group], np.array(dataset1.targets)[group]
                    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
                    train_loaders.append(train_loader)
                if num_clients == 2:
                    assert (set(index_groupings[0]).isdisjoint(index_groupings[1]))
                else:
                    # random assertions
                    assert (set(index_groupings[0]).isdisjoint(index_groupings[1]))
                    assert (set(index_groupings[0]).isdisjoint(index_groupings[2]))
                    assert (set(index_groupings[1]).isdisjoint(index_groupings[2]))



        elif args.dataset_split == 'iid':
            num_clients = args.num_clients
            lst = np.arange(len(dataset1))
            random.shuffle(lst)

            stats_dict={}
            i=0
            for group in np.array_split(lst, num_clients):
                dataset = datasets.CIFAR10(f'{args.base_dir}{args.data_dir}', train=True, transform=train_transform)
                dataset.data, dataset.targets = dataset.data[group], np.array(dataset1.targets)[group]
                train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
                train_loaders.append(train_loader)

                stats_dict[i]=group
                i+=1
            record_net_data_stats_iid(stats_dict, args)
        elif args.dataset_split=='dirichlet':
            dataset = datasets.CIFAR10(f'{args.base_dir}{args.data_dir}', train=True, transform=train_transform)
            _, y_train = dataset.data, np.array(dataset.targets)

            net_dataidx_map, y_train=dirichlet(y_train,  num_clients, alpha=args.dirichlet_alpha)
            record_net_data_stats(y_train, net_dataidx_map, args)

            for i,j in net_dataidx_map.items():
                dataset = datasets.CIFAR10(f'{args.base_dir}{args.data_dir}', train=True, transform=train_transform)
                dataset.data, dataset.targets = dataset.data[j], np.array(dataset.targets)[j]
                #print(len(dataset.data))
                train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
                train_loaders.append(train_loader)

        elif args.dataset_split == 'classimbalance':
            train_datasets, validation_dataset, test_dataset = prepare_dataset(name)
            n_classes = 10
            data_indices = [torch.nonzero(train_dataset.targets == class_id).view(-1).tolist() for class_id in
                            range(n_classes)]
            class_sizes = np.linspace(1, n_classes, n_agents, dtype='int')
            print("class_sizes for each party", class_sizes)
            party_mean = self.sample_size_cap // self.n_agents

            from collections import defaultdict
            party_indices = defaultdict(list)
            for party_id, class_sz in enumerate(class_sizes):
                classes = range(class_sz)  # can customize classes for each party rather than just listing
                each_class_id_size = party_mean // class_sz
                # print("party each class size:", party_id, each_class_id_size)
                for i, class_id in enumerate(classes):
                    # randomly pick from each class a certain number of samples, with replacement
                    selected_indices = random.choices(data_indices[class_id], k=each_class_id_size)

                    # randomly pick from each class a certain number of samples, without replacement
                    '''
                    NEED TO MAKE SURE THAT EACH CLASS HAS MORE THAN each_class_id_size for no replacement sampling
                    selected_indices = random.sample(data_indices[class_id],k=each_class_id_size)
                    '''
                    party_indices[party_id].extend(selected_indices)

                    # top up to make sure all parties have the same number of samples
                    if i == len(classes) - 1 and len(party_indices[party_id]) < party_mean:
                        extra_needed = party_mean - len(party_indices[party_id])
                        party_indices[party_id].extend(data_indices[class_id][:extra_needed])
                        data_indices[class_id] = data_indices[class_id][extra_needed:]

            indices_list = [party_index_list for party_id, party_index_list in party_indices.items()]

        elif args.dataset_split == 'powerlaw':
            indices_list = powerlaw(list(range(len(self.train_dataset))), n_agents)

        else:
            print('choose dataset split!')
        test_dataset = datasets.CIFAR10(f'{args.base_dir}{args.data_dir}', train=False, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        print('Dataset summaries:')

        for i in range(len(train_loaders)):
            print( f'\tTrain set {i}: Length: {len(train_loaders[i].dataset)}, Labels: {collections.Counter(train_loaders[i].dataset.targets.tolist())}')

        print(f'\tTest set: Length: {len(test_loader.dataset)}, Labels: {collections.Counter(test_loader.dataset.targets)}')

        weights=[]
        dataset_len = len(datasets.CIFAR10(f'{args.base_dir}{args.data_dir}', train=True,).targets)
        for loader in train_loaders:
            weights.append(len(loader.dataset.targets)/dataset_len)
        print(weights)

        return train_loaders, test_loader,weights


from torchvision.datasets import CIFAR10, CIFAR100


class FastCIFAR10(CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Scale data to [0,1]
        from torch import from_numpy
        self.data = from_numpy(self.data)
        self.data = self.data.float().div(255)
        self.data = self.data.permute(0, 3, 1, 2)

        self.targets = torch.Tensor(self.targets).long()

        # https://github.com/kuangliu/pytorch-cifar/issues/16
        # https://github.com/kuangliu/pytorch-cifar/issues/8
        for i, (mean, std) in enumerate(zip((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))):
            self.data[:, i].sub_(mean).div_(std)

        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data, self.targets
        print('CIFAR10 data shape {}, targets shape {}'.format(self.data.shape, self.targets.shape))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


def prepare_dataset(self, name='mnist'):
    if name == 'mnist':

        train = FastMNIST('.data', train=True, download=True)
        test = FastMNIST('.data', train=False, download=True)

        train_indices, valid_indices = get_train_valid_indices(len(train), self.train_val_split_ratio,
                                                               self.sample_size_cap)

        train_set = Custom_Dataset(train.data[train_indices], train.targets[train_indices], device=self.device)
        validation_set = Custom_Dataset(train.data[valid_indices], train.targets[valid_indices], device=self.device)
        test_set = Custom_Dataset(test.data, test.targets, device=self.device)

        del train, test

        return train_set, validation_set, test_set

    elif name == 'cifar10':

        train = FastCIFAR10('.data', train=True, download=True)  # , transform=transform_train)
        test = FastCIFAR10('.data', train=False, download=True)  # , transform=transform_test)

        train_indices, valid_indices = get_train_valid_indices(len(train), self.train_val_split_ratio,
                                                               self.sample_size_cap)

        train_set = Custom_Dataset(train.data[train_indices], train.targets[train_indices], device=self.device)
        validation_set = Custom_Dataset(train.data[valid_indices], train.targets[valid_indices], device=self.device)
        test_set = Custom_Dataset(test.data, test.targets, device=self.device)
        del train, test

        return train_set, validation_set, test_set
