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
from utils.model_utils import set_seed, _init_weight


def linear_init(in_dim, out_dim,  args=None, ):
    layer = LinearMerge(in_dim, out_dim, bias=args.bias)
    layer.init(args)
    return layer


def conv_init(in_channels, out_channels, kernel_size=3, stride=1,padding=1, args=None, ):
    layer = ConvMerge(in_channels, out_channels, kernel_size, stride=stride,padding=padding, bias=args.bias)
    layer.init(args)
    return layer

class ConvMerge(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_align_list = nn.ParameterList([])
        self.bias_align_list=nn.ParameterList([])
        #self.weight_align = None
        #self.bias_align=None


    def init(self, args):
        self.args = args
        set_seed(self.args.weight_seed)
        _init_weight(args, self.weight)
        # self.args.weight_seed+=1

        print(f'Conv layer info: Weight size: {self.weight.size()} Bias: {self.args.bias}, Kernel Size:{self.kernel_size}, Stride: {self.stride}, Padding: {self.padding}')

    def forward(self, x):
        x = F.conv2d(
            x, self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups
        )
        weights_diff = torch.tensor(0).float().to(self.args.device)#.cuda()#
        if len(self.weight_align_list) > 0:
            for wa in self.weight_align_list:
                if self.args.align_loss == 'ae':
                    weights_diff += torch.sum((self.weight - wa).abs().pow(1.0))
                elif self.args.align_loss == 'se':
                    weights_diff += torch.sum(torch.square(self.weight - wa))
                else:
                    sys.exit(1)
            if self.args.bias == True:
                for ba in self.bias_align_list:
                    if self.args.align_loss == 'ae':
                        weights_diff += torch.sum((self.bias - ba).abs().pow(1.0))
                    elif self.args.align_loss == 'se':
                        weights_diff += torch.sum(torch.square(self.bias - ba))
                    else:
                        sys.exit(1)
        return x, weights_diff





class LinearMerge(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.weight_align_list = nn.ParameterList([])
        self.bias_align_list=nn.ParameterList([])


    def init(self, args):
        self.args = args
        set_seed(self.args.weight_seed)
        _init_weight(args, self.weight)
        print(f'Linear layer info: Weight size: {self.weight.size()} Bias: {self.args.bias}')
        if self.args.bias:
            print(self.bias.size())
        # self.args.weight_seed+=1

    def forward(self, x):
        x = F.linear(x, self.weight, self.bias)
        weights_diff = torch.tensor(0).float().to(self.args.device)#.cuda()
        if len(self.weight_align_list) > 0:
            for wa in self.weight_align_list:
                if self.args.align_loss == 'ae':
                    weights_diff += torch.sum((self.weight - wa).abs().pow(1.0))
                elif self.args.align_loss == 'se':
                    weights_diff += torch.sum((self.weight - wa) ** 2)
                else:
                    sys.exit(1)
            if self.args.bias == True:
                for ba in self.bias_align_list:
                    if self.args.align_loss == 'ae':
                        weights_diff += torch.sum((self.bias - ba).abs().pow(1.0))
                    elif self.args.align_loss == 'se':
                        weights_diff += torch.sum(torch.square(self.bias - ba))
                    else:
                        sys.exit(1)
        return x, weights_diff

