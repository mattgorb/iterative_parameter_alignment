import torch
import torch.nn as nn
from models.layers import *
import torch.nn.functional as F

class Conv2(nn.Module):
    def __init__(self, args, weight_merge=False):
        super(Conv2, self).__init__()
        self.args=args
        self.weight_merge=weight_merge
        self.bias = self.args.bias

        if self.weight_merge:
            self.conv1 = conv_init(1, 32, 3, 1 , args=self.args, )
            self.conv2 = conv_init(32, 64, 3, 1, args=self.args, )

            self.fc1 = linear_init(9216, 128, args=self.args, )
            self.fc2 = linear_init(128, 10, args=self.args, )
        else:
            self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=self.bias)
            self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=self.bias)

            self.fc1 = nn.Linear(9216, 128, bias=self.bias)
            self.fc2 = nn.Linear(128, 10, bias=self.bias)

    def forward(self, x):
        if self.weight_merge:
            x,wd1 = self.conv1(x)
            x = F.relu(x)

            x,wd2 = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = torch.flatten(x, 1)
            print(x.size())
            x,wd3 = self.fc1(x)
            x = F.relu(x)

            print(x.size())
            x,wd4 = self.fc2(x)

            print(x.size())

            wd = wd1+wd2+wd3+wd4
            return x, wd
        else:
            x = self.conv1(x)
            x = F.relu(x)

            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = torch.flatten(x, 1)

            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            return x, torch.tensor(0)