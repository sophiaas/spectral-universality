import numpy as np

import jax.numpy as jnp
from jax import grad, value_and_grad, jit, local_devices, device_put
from jax.example_libraries.optimizers import adam

from utils import *
from models_JAX import *
from datasets import *      
from groups import *

device = local_devices(backend='gpu')[0]
print(f'Using device: {device}')


"""
Parameters of the model
"""
num_ep = 100  #number of epochs
batch_size = 4
rho = 10.        #coeffcient of regularization
std = 1.    
loginterval = 1
noise = 0.


"""
Initialize group
"""
group = dihedral(3)
group.check_dims()


"""
Initialize dataset
"""
dset = group_dset(group, std, noise)   
train_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=True)


"""
Initialize weights and optimizer
"""
init_fun, update_fun, get_params = adam(step_size=0.001)   
W = init_weights(group.order, group.irrep_dims)
W = device_put(W, device)
opt_state = init_fun(W)


"""
Initialize Cayley table
"""
perms = perm_matrices(group.order)
cayley_true = group.cayley_table



"""
Weight update function
"""
@jit
def update(opt_state, x, y, epoch):
    loss_fun = lambda V, a, b: loss(V, a, b).mean() + rho * reg(V, group.irrep_dims, group.order)
    loss_val, grads = value_and_grad(loss_fun)(get_params(opt_state), x, y)
    return loss_val, update_fun(epoch, grads, opt_state)


"""
Training loop
"""
def train(epoch, data_loader, opt_state):
    cayley = np.array(get_table(get_params(opt_state), group.order))
    cayley_score = perm_frobenius(cayley_true, cayley, perms, group.order)
    print(f"Epoch: {epoch}, Cayley score: {cayley_score:.3}")
    print(cayley)

    for batch_idx, (x, y) in enumerate(data_loader):
        x = device_put(jnp.array(x.numpy()), device)
        y = device_put(jnp.array(y.numpy()), device)
        loss_val, opt_state = update(opt_state, x, y, epoch)

        if batch_idx % 10 == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx} of {len(data_loader)} Loss: {loss_val:.3}")
            # print(get_params(opt_state)[0])
    return opt_state

for i in range(1, num_ep + 1):
    print(f'Epoch {i}')
    opt_state = train(i, train_loader, opt_state)



    