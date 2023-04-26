from __future__ import print_function
import torch
import torch.nn as nn
import math
import random
import sys
import numpy as np

def model_selector(args):
    if args.model=='MLP':
        from models.mlp import MLP
        model = MLP(args, weight_merge=not args.baseline).to(args.device)
    elif args.model=='Conv2':
        from models.conv2 import Conv2
        model = Conv2(args, weight_merge=not args.baseline).to(args.device)
    elif args.model=='Conv4':
        from models.conv4 import Conv4
        model = Conv4(args, weight_merge=not args.baseline).to(args.device)
    elif args.model == 'Conv4_Cifar100':
        from models.conv4_Cifar100 import Conv4
        model = Conv4(args, weight_merge=not args.baseline).to(args.device)
    return model

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def _init_weight(args,weight):
    set_seed(args.weight_seed)
    scale_fan=False
    mode='fan_in'
    nonlinearity='relu'
    if args.weight_init == "signed_constant":
        #using signed constant from iterand code
        fan = nn.init._calculate_correct_fan(weight, 'fan_in')
        gain = nn.init.calculate_gain('relu')
        std = gain / math.sqrt(fan)
        nn.init.kaiming_normal_(weight)  # use only its sign
        weight.data = weight.data.sign() * std
        #weight.data *= scale
    elif args.weight_init == "unsigned_constant":

        fan = nn.init._calculate_correct_fan(weight, mode)
        if scale_fan:
            fan = fan * (1 - args.prune_rate)

        gain = nn.init.calculate_gain(nonlinearity)
        std = gain / math.sqrt(fan)
        weight.data = torch.ones_like(weight.data) * std

    elif args.weight_init == "kaiming_normal":

        if scale_fan:
            fan = nn.init._calculate_correct_fan(weight, mode)
            fan = fan * (1 - args.lin_prune_rate)
            gain = nn.init.calculate_gain(nonlinearity)
            std = gain / math.sqrt(fan)
            with torch.no_grad():
                weight.data.normal_(0, std)
        else:


            nn.init.kaiming_normal_(
                weight, mode=mode, nonlinearity=nonlinearity
            )
        print(f"Using {args.weight_init} weight initialization")

    elif args.weight_init == "kaiming_uniform":
        nn.init.kaiming_uniform_(
            weight, mode=mode, nonlinearity=nonlinearity
        )


    elif args.weight_init == "xavier_normal":
        nn.init.xavier_normal_(weight)
    elif args.weight_init == "xavier_constant":

        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weight)
        std = math.sqrt(2.0 / float(fan_in + fan_out))
        weight.data = weight.data.sign() * std

    elif args.weight_init == "standard":
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    else:
        print("Set default weight initialization")
        #sys.exit()

    return weight