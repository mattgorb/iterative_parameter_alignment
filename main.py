
from __future__ import print_function
import sys
import torch
import numpy as np
from args import args
from utils.trainer import Trainer
from utils.merge_iterator import Merge_Iterator
from utils.model_utils import model_selector

def get_dataloaders(args):
    print(f'Data config: num_clients: {args.num_clients}, disjoint classes: {args.disjoint_classes}, imbalanced:{args.imbalanced}')
    if args.dataset=='MNIST':
        from datasets.mnist import get_datasets
    elif args.dataset=='CIFAR10':
        from datasets.cifar10 import get_datasets
    else:
        sys.exit()
    return get_datasets(args)


def main():
    # Training settings
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f'Using Device {args.device}')
    print(args)

    weight_dir = f'{args.base_dir}iwa_weights/'
    if args.baseline:
        train_loader1, test_dataset = get_dataloaders(args)
        model = model_selector(args)

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(params)
        for n,p in model.named_parameters():
            print(n)
            print(p.requires_grad)


        save_path = f'{weight_dir}cifar10_baseline.pt'
        trainer = Trainer(args, [train_loader1, test_dataset], model, args.device, save_path, 'model_baseline')
        trainer.fit(log_output=True)
    else:
        train_loader1, train_loader2, test_dataset = get_dataloaders(args)
        merge_iterator = Merge_Iterator(args, [train_loader1, train_loader2, test_dataset], args.device, weight_dir)
        merge_iterator.run()


if __name__ == '__main__':
    main()

