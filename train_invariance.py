import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import math

from utils import *
from models import *
from datasets import *      
from groups import *

# # torch.manual_seed(42)
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: ', device)
# device = 'cpu' 


group = dihedral(4)

num_ep = 10000      #number of epochs
batch_size = 16     
weight = .001    #coeffcient of regularization
loginterval = 1



dset = group_dset(group)   
train_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=True)


model = spectral_net(group.order, group.irrep_dims).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


perms = perm_matrices(group.order).to(device)
cayley_true = group.cayley_table.to(device)


def train(epoch, data_loader):
    # print('min, max ', model.W.abs().min(), model.W.abs().max())
    cayley = model.get_table()
    cayley_score = perm_frobenius(cayley_true, cayley, perms)
    # cayley_score = is_isomorphic(cayley, cayley_true)
    print(f"Epoch: {epoch}, Cayley score: {cayley_score:.3}")
    print(cayley)

    
    for batch_idx, (x, y) in enumerate(data_loader):

        model.train()   
        optimizer.zero_grad()

        x = x.to(device) 
        y = y.to(device)


        loss = model.loss(x, y).mean()
        reg = model.reg()
        tot_loss = loss + weight * reg

        tot_loss.backward()
        optimizer.step()


        if batch_idx % 10 == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx} of {len(data_loader)} Loss: {loss:.3} Reg: {reg:.3}")


  

if __name__ == "__main__":
    for i in range(1, num_ep):
        print(f'Epoch {i}')
        train(i, train_loader)
