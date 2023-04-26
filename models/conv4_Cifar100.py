import torch
import torch.nn as nn
from models.layers import *

class Conv4(nn.Module):
    def __init__(
        self,  args=None, weight_merge=False ) -> None:
        super().__init__()


        self.args=args
        self.weight_merge=weight_merge
        self.bias=self.args.bias

        if self.weight_merge:
            self.conv1 = conv_init(3, 64, args=self.args)
            self.conv2 = conv_init(64, 64, args=self.args)
            self.conv3 = conv_init(64, 128, args=self.args)
            self.conv4 = conv_init(128, 128, args=self.args)
            self.fc1=linear_init(32*32*8, 256, args=self.args)
            self.fc2=linear_init(256, 256, args=self.args)
            self.fc3=linear_init(256, 100, args=self.args)
        else:

            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=self.bias)
            self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=self.bias)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=self.bias)
            self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=self.bias)
            self.fc1=nn.Linear(32*32*8, 1024, bias=self.bias)
            self.fc2=nn.Linear(1024, 512, bias=self.bias)
            self.fc3=nn.Linear(512, 100, bias=self.bias)

        self.max_pool=nn.MaxPool2d((2, 2))
        self.relu=nn.ReLU(True)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.weight_merge:
            x,wd1 = self.conv1(x)
            x = self.relu(x)
            x,wd2 = self.conv2(x)
            x = self.relu(x)
            x = self.max_pool(x)
            x,wd3 = self.conv3(x)
            x = self.relu(x)
            x,wd4 = self.conv4(x)
            x = self.relu(x)
            x = self.max_pool(x)
            x = x.view(x.size(0), 8192)
            x,wd5 = self.fc1(x)
            x = self.relu(x)
            x,wd6 = self.fc2(x)
            x = self.relu(x)
            x,wd7 = self.fc3(x)
            wd=wd1+wd2+wd3+wd4+wd5+wd6+wd7
            return x, wd
        else:
            x=self.conv1(x)
            x=self.relu(x)
            x=self.conv2(x)
            x=self.relu(x)
            x=self.max_pool(x)
            x=self.conv3(x)
            x=self.relu(x)
            x=self.conv4(x)
            x=self.relu(x)

            x=self.max_pool(x)
            x = x.view(x.size(0), 8192)
            x = self.fc1(x)

            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            return x, torch.tensor(0)