import torch
import torch.nn as nn
from models.layers import *


#https://github.com/gaoliang13/FedDC/blob/main/utils_models.py

class LeNetCifar10(nn.Module):
    def __init__(
        self,  args=None, weight_merge=False ) -> None:
        super().__init__()


        self.args=args
        self.weight_merge=weight_merge
        self.bias=self.args.bias

        if self.weight_merge:
            #fix this section
            self.conv1 = conv_init(3, 64,kernel_size=5, args=self.args)
            self.conv2 = conv_init(64, 64, kernel_size=5,args=self.args)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2,)
            self.fc1=linear_init(64*5*5, 384, args=self.args)
            self.fc2=linear_init(384, 192, args=self.args)
            self.fc3=linear_init(192, 10, args=self.args)
        else:
            self.n_cls = 10
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5,  padding=1, bias=self.bias)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5,  padding=1, bias=self.bias)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2,)
            self.fc1 = nn.Linear(64*5*5, 384, bias=self.bias)
            self.fc2 = nn.Linear(384, 192, bias=self.bias)
            self.fc3 = nn.Linear(192, 10, bias=self.bias)



        self.max_pool=nn.MaxPool2d((2, 2))
        self.relu=nn.ReLU(True)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.weight_merge:

            x,wd1 = self.conv1(x)
            x=F.relu(x)
            x=self.pool(x)
            x,wd2 = self.conv2(x)
            x=F.relu(x)
            x=self.pool(x)
            x = x.view(-1, 64*5*5)
            x,wd3 = self.fc1(x)
            x=F.relu(x)
            x = self.fc2(x)
            x=F.relu(x)
            x,wd4 = self.fc3(x)


            wd=wd1+wd2+wd3+wd4
            return x, wd
        else:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64*5*5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x, torch.tensor(0)