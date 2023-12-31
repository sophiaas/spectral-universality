import torch
from numpy import random
import os 
from groups import *
import numpy as np

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



"""
Contrastive learning dataset for class labels. A datapoint is a pair of (noisy) data with the same label. 
"""
class label_dset(torch.utils.data.Dataset):
    def __init__(self, root, noise=0.):
        self.noise = noise
        self.data = [np.load(os.path.join(root, P)) for P in os.listdir(root)]

    def __getitem__(self, index):
        label = random.randint(low=0, high=len(self.data))
        num_data = len(self.data[label])
        idx1 = random.randint(low=0, high=num_data)
        idx2 = random.randint(low=0, high=num_data)

        x_re = self.data[label][idx1]
        y_re = self.data[label][idx2]
        x = x_re + 1j * 0.
        y = y_re + 1j * 0.
        
        perturb_x = self.noise * random.randn(x_re.shape[0]) 
        x += perturb_x + 1j * 0.

        return x, y

    def __len__(self):
        return 1000
