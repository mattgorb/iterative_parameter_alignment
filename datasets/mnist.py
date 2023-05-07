from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
import collections
from torchvision.datasets import MNIST
import random
from datasets.dirichlet_partition import dirichlet,  record_net_data_stats, record_net_data_stats_iid
from torch.utils.data import DataLoader, Dataset
import random
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import CIFAR10, CIFAR100
from collections import Counter


def get_datasets(args):
    # not using normalization
    transform = transforms.Compose([
        transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))
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


                dataset1 = datasets.MNIST(f'{args.base_dir}data', train=True, transform=transform)
                dataset2 = datasets.MNIST(f'{args.base_dir}data', train=True, transform=transform)

                dataset1.data, dataset1.targets = dataset1.data[d1], dataset1.targets[d1]
                dataset2.data, dataset2.targets = dataset2.data[d2], dataset2.targets[d2]
                assert (set(d1).isdisjoint(d2))
                train_loader1 = DataLoader(dataset1, batch_size=args.batch_size, shuffle=True)
                train_loader2 = DataLoader(dataset2, batch_size=args.batch_size, shuffle=True)
                train_loaders.append(train_loader1)
                train_loaders.append(train_loader2)
            else:
                for group in index_groupings:
                    dataset = datasets.MNIST(f'{args.base_dir}data', train=True, transform=transform)
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
                dataset = datasets.MNIST(f'{args.base_dir}data', train=True, transform=transform)
                dataset.data, dataset.targets = dataset.data[group], dataset.targets[group]
                train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
                train_loaders.append(train_loader)
                stats_dict[i]=group
                i+=1

            dataset = datasets.MNIST(f'{args.base_dir}data', train=True, transform=transform)
            record_net_data_stats_iid(stats_dict, args)

        elif args.dataset_split=='dirichlet':
            X_train, y_train = dataset1.data, dataset1.targets
            net_dataidx_map,y_train=dirichlet(y_train,  num_clients, alpha=args.dirichlet_alpha)
            record_net_data_stats(y_train, net_dataidx_map, args)

            for i,j in net_dataidx_map.items():
                dataset = datasets.MNIST(f'{args.base_dir}data', train=True, transform=transform)
                dataset.data, dataset.targets = dataset.data[j], dataset.targets[j]
                #print(len(dataset.data))
                train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
                train_loaders.append(train_loader)
        elif args.dataset_split == 'classimbalance':
            data_prepper = Data_Prepper(
                'mnist', train_batch_size=args.batch_size, n_agents=num_clients,
                sample_size_cap=6000,
                train_val_split_ratio=0.8, device=args.device,
                args_dict=args)

            train_loaders = data_prepper.get_train_loaders(num_clients, 'classimbalance')

            #for i in train_loaders:
                #print(i.dataset.__len__())
                #print(i.dataset[0][0].size())
                #print(i.dataset[0][1])
            #sys.exit()

        elif args.dataset_split == 'powerlaw':
            data_prepper = Data_Prepper(
                'mnist', train_batch_size=args.batch_size, n_agents=num_clients,
                sample_size_cap=6000,
                train_val_split_ratio=0.8, device=args.device,
                args_dict=args)

            train_loaders = data_prepper.get_train_loaders(num_clients, 'powerlaw')

        elif args.dataset_split == 'uniform':
            data_prepper = Data_Prepper(
                'mnist', train_batch_size=args.batch_size, n_agents=num_clients,
                sample_size_cap=6000,
                train_val_split_ratio=0.8, device=args.device,
                args_dict=args)

            train_loaders = data_prepper.get_train_loaders(num_clients, 'uniform')


        else:
            print('Choose valid dataset partition')
            sys.exit()


        test_dataset = datasets.MNIST(f'{args.base_dir}data', train=False, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


        print('Dataset summaries:')

        for i in range(len(train_loaders)):
            print(f'Train set {i}: Length: {len(train_loaders[i].dataset)}, Labels: {collections.Counter(train_loaders[i].dataset.targets.tolist())}')

        print(f'Test set: Length: {len(test_loader.dataset)}, Labels: {collections.Counter(test_loader.dataset.targets.tolist())}')

        weights=[]
        dataset_len = len(datasets.MNIST(f'{args.base_dir}data', train=True,).targets)
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









class Data_Prepper:
    def __init__(self, name, train_batch_size, n_agents,
                 sample_size_cap=-1, test_batch_size=100, valid_batch_size=None,
                 train_val_split_ratio=0.8, device=None, args_dict=None):
        self.args = None
        self.args_dict = args_dict
        self.name = name
        self.device = device
        self.n_agents = n_agents
        self.sample_size_cap = sample_size_cap
        self.train_val_split_ratio = train_val_split_ratio

        self.init_batch_size(train_batch_size, test_batch_size, valid_batch_size)

        if name in ['mnist', 'cifar10']:
            self.train_dataset, self.validation_dataset, self.test_dataset = self.prepare_dataset(name)

            print('------')
            print("Train to split size: {}. Validation size: {}. Test size: {}".format(len(self.train_dataset),
                                                                                       len(self.validation_dataset),
                                                                                       len(self.test_dataset)))
            print('------')

            self.valid_loader = DataLoader(self.validation_dataset, batch_size=self.test_batch_size)
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size)

        else:
            raise NotImplementedError

    def init_batch_size(self, train_batch_size, test_batch_size, valid_batch_size):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.valid_batch_size = valid_batch_size if valid_batch_size else test_batch_size

    def get_valid_loader(self):
        return self.valid_loader

    def get_test_loader(self):
        return self.test_loader

    def get_train_loaders(self, n_agents, split='powerlaw', batch_size=None):
        if not batch_size:
            batch_size = self.train_batch_size

        if self.name not in ['sst', 'mr', 'mnist', 'cifar10']: raise NotImplementedError

        if self.name in ['mnist', 'cifar10']:

            if split == 'classimbalance':
                if self.name not in ['mnist', 'cifar10']:
                    raise NotImplementedError(
                        "Calling on dataset {}. Only mnist and cifar10 are implemnted for this split".format(self.name))

                n_classes = 10
                data_indices = [torch.nonzero(self.train_dataset.targets == class_id).view(-1).tolist() for class_id in range(n_classes)]
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


            elif split == 'powerlaw':
                indices_list = powerlaw(list(range(len(self.train_dataset))), n_agents)

            elif split == 'uniform':
                indices_list = random_split(sample_indices=list(range(len(self.train_dataset))), m_bins=n_agents,
                                            equal=True)


        self.train_datasets = [Custom_Dataset(self.train_dataset.data[indices], self.train_dataset.targets[indices]) for
                               indices in indices_list]



        self.shard_sizes = [len(indices) for indices in indices_list]
        #agent_train_loaders = [
            #DataLoader(self.train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(indices)) for indices in
            #indices_list]

        agent_train_loaders = [DataLoader(dataset, batch_size=batch_size,)
                for dataset in self.train_datasets]
        self.train_loaders = agent_train_loaders
        return agent_train_loaders

    def prepare_dataset(self, name='mnist'):

        if name == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
            ])

            train = datasets.MNIST(f'.data', train=True, download=True, transform=transform)
            test = datasets.MNIST(f'.data', train=False, transform=transform)

            #train = FastMNIST('.data', train=True, download=True)
            #test = FastMNIST('.data', train=False, download=True)

            train_indices, valid_indices = get_train_valid_indices(len(train), self.train_val_split_ratio,
                                                                   self.sample_size_cap)

            train_set = Custom_Dataset(train.data[train_indices], train.targets[train_indices], device=self.device)
            validation_set = Custom_Dataset(train.data[valid_indices], train.targets[valid_indices], device=self.device)
            test_set = Custom_Dataset(test.data, test.targets, device=self.device)

            del train, test

            return train_set, validation_set, test_set

        elif name == 'cifar10':

            train = FastCIFAR10(f'.data', train=True, download=True)  # , transform=transform_train)
            test = FastCIFAR10(f'.data', train=False, download=True)  # , transform=transform_test)

            train_indices, valid_indices = get_train_valid_indices(len(train), self.train_val_split_ratio,
                                                                   self.sample_size_cap)

            train_set = Custom_Dataset(train.data[train_indices], train.targets[train_indices], device=self.device)
            validation_set = Custom_Dataset(train.data[valid_indices], train.targets[valid_indices], device=self.device)
            test_set = Custom_Dataset(test.data, test.targets, device=self.device)
            del train, test

            return train_set, validation_set, test_set

        else:
            raise NotImplementedError




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


class Custom_Dataset(Dataset):

    def __init__(self, X, y, device=None, transform=None):
        self.data = X.to(device)
        self.targets = y.to(device)
        self.count = len(X)
        self.device = device
        self.transform = transform

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.data[idx]), self.targets[idx]

        return self.data[idx], self.targets[idx]


def random_split(sample_indices, m_bins, equal=True):
    np.random.seed(1111)
    sample_indices = np.asarray(sample_indices)
    if equal:
        indices_list = np.array_split(sample_indices, m_bins)
    else:
        split_points = np.random.choice(
            n_samples - 2, m_bins - 1, replace=False) + 1
        split_points.sort()
        indices_list = np.split(sample_indices, split_points)

    return indices_list


def powerlaw(sample_indices, n_agents, alpha=1.65911332899, shuffle=False):
    # the smaller the alpha, the more extreme the division
    if shuffle:
        random.seed(1234)
        random.shuffle(sample_indices)

    from scipy.stats import powerlaw
    import math
    party_size = int(len(sample_indices) / n_agents)
    b = np.linspace(powerlaw.ppf(0.01, alpha), powerlaw.ppf(0.99, alpha), n_agents)
    shard_sizes = list(map(math.ceil, b / sum(b) * party_size * n_agents))
    indices_list = []
    accessed = 0
    for agent_id in range(n_agents):
        indices_list.append(sample_indices[accessed:accessed + shard_sizes[agent_id]])
        accessed += shard_sizes[agent_id]
    return indices_list


def get_train_valid_indices(n_samples, train_val_split_ratio, sample_size_cap=None):
    indices = list(range(n_samples))
    random.seed(1111)
    random.shuffle(indices)
    split_point = int(n_samples * train_val_split_ratio)
    train_indices, valid_indices = indices[:split_point], indices[split_point:]
    if sample_size_cap is not None:
        train_indices = indices[:min(split_point, sample_size_cap)]

    return train_indices, valid_indices


def split_torchtext_dataset_ratios(data, ratios):
    train_datasets = []
    while len(ratios) > 1:
        split_ratio = ratios[0] / sum(ratios)
        ratios.pop(0)
        train_dataset, data = data.split(split_ratio=split_ratio, random_state=random.seed(1234))
        train_datasets.append(train_dataset)
    train_datasets.append(data)
    return train_datasets


class FastMNIST(MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = self.data.unsqueeze(1).float().div(255)
        from torch.nn import ZeroPad2d
        pad = ZeroPad2d(2)
        self.data = torch.stack([pad(sample.data) for sample in self.data])

        self.targets = self.targets.long()

        self.data = self.data.sub_(self.data.mean()).div_(self.data.std())
        # self.data = self.data.sub_(0.1307).div_(0.3081)
        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data, self.targets
        print('MNIST data shape {}, targets shape {}'.format(self.data.shape, self.targets.shape))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target
