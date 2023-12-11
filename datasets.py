import torch
from numpy import random

from groups import *


"""
Contrastive learning dataset for groups. A datapoint is a pair of (noisy) complex vectors in the same orbit. 
"""
class group_dset(torch.utils.data.Dataset):
    def __init__(self, group, std=1., noise=0.):
        self.group = group
        self.std = std
        self.noise = noise

    def __getitem__(self, index):
        x_re = self.std * random.randn(self.group.order) 
        x_im = self.std * random.randn(self.group.order)
        x = x_re + 1j * x_im
        y = self.group.act(x)
        
        perturb_re = self.noise * random.randn(self.group.order) 
        perturb_im = self.noise * random.randn(self.group.order) 
        x += perturb_re + 1j * perturb_im

        return x, y

    def __len__(self):
        return 1000


