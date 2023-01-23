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

'''
def conv_init(in_channels, out_channels, kernel_size, stride, bias=False, args=None, ):
    layer = ConvMerge(in_channels, out_channels, kernel_size, stride=stride, bias=bias)
    layer.init(args)
    return layer

class ConvMerge(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_align = None

    def init(self, args):
        self.args = args
        set_seed(self.args.weight_seed)
        # this isn't default initialization.  not sure if necessary, need to test.
        #if self.args.kn_init:
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")
        # models do NOT need to be initialized the same, however they appeared to converge slightly faster with same init

    def forward(self, x):
        x = F.conv2d(
            x, self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups
        )
        weights_diff = torch.tensor(0)
        if self.weight_align is not None:
            # using absolute error here.
            weights_diff = torch.sum((self.weight - self.weight_align).abs())
            # MSE loss -- not able to get as good results using this loss fn.
            # weights_diff=torch.mean((self.weight-self.weight_align)**2)
        return x, weights_diff


def linear_init(in_dim, out_dim, bias=False, args=None, ):
    layer = LinearMerge(in_dim, out_dim, bias=bias)
    layer.init(args)
    return layer


class LinearMerge(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_align = None

    def init(self, args):
        self.args = args
        set_seed(self.args.weight_seed)
        #set_seed(1)
        # this isn't default initialization.  not sure if necessary, need to test.
        #if self.args.kn_init:
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")
        # models do NOT need to be initialized the same, however they appeared to converge slightly faster with same init
        #self.args.weight_seed+=1

    def forward(self, x):
        x = F.linear(x, self.weight, self.bias)
        weights_diff = torch.tensor(0)
        if self.weight_align is not None:
            # using absolute error here.
            weights_diff = torch.sum((self.weight - self.weight_align).abs())
            # MSE loss -- not able to get as good results using this loss fn.
            # weights_diff=torch.mean((self.weight-self.weight_align)**2)
        return x, weights_diff
'''
def linear_init(in_dim, out_dim,  args=None, ):
    layer = LinearMerge(in_dim, out_dim, bias=args.bias)
    layer.init(args)
    return layer


def conv_init(in_channels, out_channels, kernel_size=3, stride=1,padding=0, args=None, ):
    layer = ConvMerge(in_channels, out_channels, kernel_size, stride=stride,padding=padding, bias=args.bias)
    layer.init(args)
    return layer

class ConvMerge(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_align = None
        self.bias_align=None



    def init(self, args):
        self.args = args
        print(self.weight)
        _init_weight(args, self.weight)
        # self.args.weight_seed+=1
        print(self.weight[0][:10])
        sys.exit()
        print(f'Conv layer info: Weight size: {self.weight.size()} Bias: {self.args.bias}, Kernel Size:{self.kernel_size}, Stride: {self.stride}, Padding: {self.padding}')

    def forward(self, x):
        x = F.conv2d(
            x, self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups
        )
        weights_diff = torch.tensor(0)
        if self.weight_align is not None:
            # using absolute error here.
            if self.args.align_loss=='ae':
                weights_diff = torch.sum((self.weight - self.weight_align).abs())
                if self.args.bias==True:
                    weights_diff += torch.sum((self.bias - self.bias_align).abs())
            elif self.args.align_loss=='se':
                weights_diff=torch.sum((self.weight-self.weight_align)**2)
                if self.args.bias==True:
                    weights_diff += torch.sum((self.bias - self.bias_align)**2)
            else:
                sys.exit(1)
        return x, weights_diff





class LinearMerge(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_align = None
        self.bias_align=None

    def init(self, args):
        self.args = args
        _init_weight(args, self.weight)
        print(f'Linear layer info: Weight size: {self.weight.size()} Bias: {self.args.bias}')
        if self.args.bias  :
            print(self.bias.size())
        #sys.exit()

        # self.args.weight_seed+=1
    def forward(self, x):
        weights_diff = torch.tensor(0)
        if self.weight_align is not None:
            if self.args.align_loss=='ae':
                weights_diff = torch.sum((self.weight - self.weight_align).abs())
                if self.args.bias==True:
                    weights_diff += torch.sum((self.bias - self.bias_align).abs())
            elif self.args.align_loss=='se':
                weights_diff=torch.sum((self.weight-self.weight_align)**2)
                if self.args.bias==True:
                    weights_diff += torch.sum((self.bias - self.bias_align)**2)
            else:
                sys.exit(1)
        return x, weights_diff


