import torch

from utils import *
from models_torch import *
from datasets import *      
from groups import *

# # torch.manual_seed(42)       
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: ', device)
# device = 'cpu'


"""
Parameters of te model
"""
num_ep = 50   #number of epochs
batch_size = 4
weight = 10.        #coeffcient of regularization
loginterval = 1
noise = 0.
path = './test/'

    

"""
Initialize dataset
"""
dset = label_dset(path, noise)
train_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=True)
dim = dset.__getitem__(0)[0].shape[-1]
irrep_dims = [1] * dim

 
"""
Initialize model and optimizer
"""
model = spectral_net(dim, irrep_dims).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



"""
Training loop
"""
def train(epoch, data_loader):
    for batch_idx, (x, y) in enumerate(data_loader):

        model.train()
        optimizer.zero_grad()

        x = x.to(device)
        y = y.to(device)

        loss = model.loss(x, y).mean()
        reg = model.reg()
        tot_loss = weight * reg + loss

        tot_loss.backward()
        optimizer.step()


        if batch_idx % 10 == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx} of {len(data_loader)} Loss: {loss:.3} Reg: {reg:.3}")




if __name__ == "__main__":
    for i in range(1, num_ep + 1):
        print(f'Epoch {i}')
        train(i, train_loader)

