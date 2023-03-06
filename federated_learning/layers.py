from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from matplotlib import pyplot as plt
import pandas as pd
from typing import Any, cast, Dict, List, Optional, Union
import numpy as np
import sys
#from utils.model_utils import set_seed, _init_weight


def linear_init(in_dim, out_dim, device, args=None,):
    layer = LinearMerge(in_dim, out_dim, bias=True)
    layer.init(args,device)
    return layer


def conv_init(in_channels, out_channels, kernel_size=3, stride=1,padding=1, device=None, args=None, ):
    layer = ConvMerge(in_channels, out_channels, kernel_size, stride=stride,padding=padding, bias=True)
    layer.init(args,device)
    return layer

class ConvMerge(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_align_list = nn.ParameterList([])
        self.bias_align_list=nn.ParameterList([])
        self.train_weight_list=[]


    def init(self, args,device):
        self.args = args
        self.device=device
        self.align_loss='ae'
        set_seed(self.args.weight_seed)
        #_init_weight(args, self.weight)
        # self.args.weight_seed+=1

        print(f'Conv layer info: Weight size: {self.weight.size()} Bias: {self.bias}, Kernel Size:{self.kernel_size}, Stride: {self.stride}, Padding: {self.padding}')

    def forward(self, x):
        x = F.conv2d(
            x, self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups
        )
        weights_diff = torch.tensor(0).float().to(self.device)#.cuda()#
        if len(self.weight_align_list) > 0:
            for wa in range(len(self.weight_align_list)):
                if self.align_loss == 'ae':
                    #weights_diff += torch.sum((self.weight - self.weight_align_list[wa]).abs())
                    weights_diff += (self.train_weight_list[wa]*torch.sum((self.weight - self.weight_align_list[wa]).abs()))
                elif self.align_loss == 'se':
                    weights_diff += torch.sum(torch.square(self.weight - self.weight_align_list[wa]))
                elif self.align_loss == 'pd':
                    weights_diff += (self.train_weight_list[wa]*torch.sum((self.weight - self.weight_align_list[wa]).abs().pow(self.args.delta)))
                else:
                    sys.exit(1)


            if self.args.bias == True:
                for ba in range(len(self.bias_align_list)):
                    if self.align_loss == 'ae':
                        #weights_diff += torch.sum((self.bias - self.bias_align_list[ba]).abs())
                        weights_diff += (self.train_weight_list[wa]*torch.sum((self.bias - self.bias_align_list[ba]).abs()))
                    elif self.align_loss == 'se':
                        weights_diff += torch.sum(torch.square(self.bias - self.bias_align_list[ba]))
                    elif self.align_loss == 'pd':
                        weights_diff += (self.train_weight_list[wa]*torch.sum((self.bias - self.bias_align_list[ba]).abs().pow(self.args.delta)))
                    else:
                        sys.exit(1)
        return x, weights_diff





class LinearMerge(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.weight_align_list = nn.ParameterList([])
        self.bias_align_list=nn.ParameterList([])
        self.train_weight_list=[]

    def init(self, args,device):
        self.args = args
        self.device=device
        self.align_loss='ae'
        #set_seed(self.args.weight_seed)
        #_init_weight(args, self.weight)
        #print(f'Linear layer info: Weight size: {self.weight.size()} Bias: {self.args.bias}')
        # self.args.weight_seed+=1

    def forward(self, x):
        x = F.linear(x, self.weight, self.bias)
        weights_diff = torch.tensor(0).float().to(self.device)#.cuda()
        if len(self.weight_align_list) > 0:
            for wa in range(len(self.weight_align_list)):
                if self.align_loss == 'ae':

                    weights_diff += (self.train_weight_list[wa]*torch.sum((self.weight - self.weight_align_list[wa]).abs()))
                elif self.align_loss == 'se':
                    weights_diff += torch.sum(torch.square(self.weight - self.weight_align_list[wa]))
                elif self.align_loss == 'pd':
                    weights_diff += (self.train_weight_list[wa]*torch.sum((self.weight - self.weight_align_list[wa]).abs().pow(self.args.delta)))
                else:
                    sys.exit(1)


            if self.args.bias == True:
                for ba in range(len(self.bias_align_list)):
                    if self.args.align_loss == 'ae':
                        #weights_diff += torch.sum((self.bias - self.bias_align_list[ba]).abs())
                        weights_diff += (self.train_weight_list[wa]*torch.sum((self.bias - self.bias_align_list[ba]).abs()))
                    elif self.args.align_loss == 'se':
                        weights_diff += torch.sum(torch.square(self.bias - self.bias_align_list[ba]))
                    elif self.args.align_loss == 'pd':
                        weights_diff += (self.train_weight_list[wa]*torch.sum((self.bias - self.bias_align_list[ba]).abs().pow(self.args.delta)))
                    else:
                        sys.exit(1)
        return x, weights_diff








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
