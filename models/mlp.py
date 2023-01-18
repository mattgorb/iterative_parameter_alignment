import torch
import torch.nn as nn
from models.layers import *
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, args, weight_merge=False):
        super(MLP, self).__init__()
        self.args = args
        self.weight_merge = weight_merge
        # bias False for now, have not tested adding bias to the loss fn.
        if self.weight_merge:
            self.fc1 = linear_init(28 * 28, 1024, bias=False, args=self.args, )
            self.fc2 = linear_init(1024, 10, bias=False, args=self.args, )
        else:
            self.fc1 = nn.Linear(28 * 28, 1024, bias=False)
            self.fc2 = nn.Linear(1024, 10, bias=False)


    def forward(self, x, ):
        self.wd=torch.tensor(0)
        if self.weight_merge:
            x, wa1 = self.fc1(x.view(-1, 28 * 28))
            x = F.relu(x)
            x, wa2 = self.fc2(x)
            score_diff = wa1 + wa2
            return x, score_diff
        else:
            x = self.fc1(x.view(-1, 28 * 28))
            x = F.relu(x)
            x = self.fc2(x)
            return x, torch.tensor(0)
