import torch
import torch.nn as nn
from torch.nn import functional as F

class hyper(nn.Module):
    def __init__(self, state_dim, action_dim, low = 0.5, high = 2.0):
        super(hyper, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        self.low = low
        self.high = high

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        w = F.relu(self.l1(sa))
        w = F.relu(self.l2(w))
        # adv as init point
        w = self.l3(w)
        w = torch.tanh(w)
        w = self.low + (self.high - self.low) * (w + 1.) / 2.

        return w