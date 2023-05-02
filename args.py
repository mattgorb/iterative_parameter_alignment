import argparse
import sys
import yaml

from configs import parser as _parser

args = None


def parse_arguments():
    parser = argparse.ArgumentParser(description="Iterative Weight Alignment")
    parser.add_argument('--batch-size', type=int, default=256, help='input batch size for training (default: 64)')
    parser.add_argument('--local_epochs', type=int, default=1, help='number of epochs to train on a client')
    parser.add_argument('--merge_iter', type=int, default=1000,  help='number of iterations to merge')
    parser.add_argument('--data_transform', type=bool, default=False)
    parser.add_argument('--weight_init', type=str, default=None)

    parser.add_argument('--bias', type=bool, default=False)
    parser.add_argument('--align_loss', type=str, default=None)
    parser.add_argument('--weight_align_factor', type=float, default=1.0, )
    parser.add_argument('--delta', type=float, default=None, )
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR')
    parser.add_argument('--optimizer', type=str, default='Adam',)

    #ablation args
    parser.add_argument('--single_model', type=bool, default=False, )


    parser.add_argument('--random_topology', type=bool, default=False, )
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--weight_seed', type=int, default=1, )
    parser.add_argument('--same_initialization', type=bool, default=False, )
    parser.add_argument('--gpu', type=int, default=1, )
    parser.add_argument('--save-model', action='store_true', default=False,help='For Saving the current Model')
    parser.add_argument('--baseline', type=bool, default=False, help='train base model')


    parser.add_argument('--base_dir', type=str, default="/s/luffy/b/nobackup/mgorb/", help='Directory for data and weights')
    parser.add_argument('--data_dir', type=str, default="data")

    parser.add_argument('--config', type=str, default=None, help='config file to use')




    #dataset configs
    parser.add_argument('--dataset', type=str, default=None, )
    parser.add_argument('--num_clients', type=int, default=None, )
    parser.add_argument('--uneven', type=str, default=None, )
    parser.add_argument('--imbalanced', type=bool, default=False)
    parser.add_argument('--dataset_split', type=str, default=None)
    parser.add_argument('--dirichlet_alpha', type=float, default=None)

    args = parser.parse_args()

    # Allow for use from notebook without config file
    if len(sys.argv) > 1:
        get_config(args)

    return args


def get_config(args):
    # get commands from command line
    override_args = _parser.argv_to_vars(sys.argv)

    # load yaml file
    yaml_txt = open(args.config).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML config from {args.config}")
    args.__dict__.update(loaded_yaml)


def run_args():
    global args
    if args is None:
        args = parse_arguments()




run_args()
