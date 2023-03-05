import torch
import torch.nn as nn
from models.layers import *
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, args, weight_merge=False):
        super(MLP, self).__init__()
        self.args = args
        self.weight_merge = weight_merge
        if self.weight_merge:
            self.fc1 = linear_init(28 * 28, 200, args=self.args, )
            self.fc2 = linear_init(200, 200, args=self.args, )
            self.fc3 = linear_init(200, 10,  args=self.args, )
        else:
            self.fc1 = nn.Linear(28 * 28, 200, bias=self.args.bias)
            self.fc2 = nn.Linear(200, 200, bias=self.args.bias)
            self.fc3 = nn.Linear(200, 10, bias=self.args.bias)


    def forward(self, x, ):
        if self.weight_merge:
            x, wa1 = self.fc1(x.view(-1, 28 * 28))
            x = F.relu(x)
            x, wa2 = self.fc2(x)
            x = F.relu(x)
            x, wa3 = self.fc2(x)
            print("HERE")
            score_diff = wa1 + wa2 + wa3
            return x, score_diff
        else:
            x = self.fc1(x.view(-1, 28 * 28))
            x = F.relu(x)
            x = self.fc2(x)
            x = F.relu(x)
            x = self.fc3(x)
            return x, torch.tensor(0)
