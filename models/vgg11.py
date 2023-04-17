import math

import torch.nn as nn
import torch.nn.init as init

import torch
import torch.nn as nn
from models.layers import *


class VGG11(nn.Module):
    def __init__(self, args, weight_merge=False):
        super(VGG11, self).__init__()
        self.args=args
        self.weight_merge=weight_merge
        self.bias = self.args.bias

        if self.weight_merge:
            self.layer1 = conv_init(3, 5, 3, 1 , args=self.args, )
            self.layer2 = conv_init(64, 128, 3, 1, args=self.args, )
            self.layer3 = conv_init(128, 256, 3, 1, args=self.args, )
            self.layer4 = conv_init(256, 256, 3, 1, args=self.args, )
            self.layer5 = conv_init(256, 512, 3, 1, args=self.args, )
            self.layer6 = conv_init(512, 512, 3, 1, args=self.args, )
            self.layer7 = conv_init(512, 512, 3, 1, args=self.args, )
            self.layer8 = conv_init(512, 512, 3, 1, args=self.args, )
            self.fc = linear_init(512, 512, args=self.args, )
            self.fc1 = linear_init(512, 512, args=self.args, )
            self.fc2 = linear_init(512, 10, args=self.args, )
        else:
            self.layer1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
            self.layer2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.layer3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
            self.layer4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            self.layer5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
            self.layer6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            self.layer7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            self.layer8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

            self.fc = nn.Linear( 512, 512)
            self.fc1 = nn.Linear(512, 512)
            self.fc2 = nn.Linear(512, 10)

        self.relu=nn.ReLU(True)

        self.max_pool=nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.weight_merge:
            out,wd1 = self.layer1(x)
            out = self.relu(out)
            out = self.max_pool(out)

            out,wd2 = self.layer2(out)
            out = self.relu(out)
            out = self.max_pool(out)

            out,wd3 = self.layer3(out)
            out = self.relu(out)
            out,wd4 = self.layer4(out)
            out = self.relu(out)
            out = self.max_pool(out)

            out,wd5 = self.layer5(out)
            out = self.relu(out)
            out,wd6 = self.layer6(out)
            out = self.relu(out)
            out = self.max_pool(out)

            out,wd7 = self.layer7(out)
            out = self.relu(out)

            out,wd8 = self.layer8(out)
            out = self.relu(out)
            out = self.max_pool(out)

            out = out.reshape(out.size(0), -1)


            out,wd9 = self.fc(out)
            out = self.relu(out)

            out,wd10 = self.fc1(out)
            out = self.relu(out)
            out,wd11 = self.fc2(out)


            wd = wd1+wd2+wd3+wd4+wd5+wd6+wd7+wd8+wd9+wd10+wd11
            return out, wd
        else:
            out = self.layer1(x)
            out=self.relu(out)
            out=self.max_pool(out)

            out = self.layer2(out)
            out=self.relu(out)
            out=self.max_pool(out)

            out = self.layer3(out)
            out=self.relu(out)
            out = self.layer4(out)
            out=self.relu(out)
            out=self.max_pool(out)

            out = self.layer5(out)
            out=self.relu(out)
            out = self.layer6(out)
            out=self.relu(out)
            out=self.max_pool(out)

            out = self.layer7(out)
            out=self.relu(out)

            out = self.layer8(out)
            out=self.relu(out)
            out=self.max_pool(out)
            out = out.reshape(out.size(0), -1)


            out = self.fc(out)
            out=self.relu(out)

            out = self.fc1(out)
            out=self.relu(out)
            out = self.fc2(out)
            return out, torch.tensor(0)



