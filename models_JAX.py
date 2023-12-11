import jax.numpy as jnp
import jax
from jax.lax import complex

from utils import *



# initializer = jax.nn.initializers.glorot_uniform(in_axis=-3, out_axis=-2)

initializer = jax.nn.initializers.uniform(scale=1.)

def init_weights(group_order, irrep_dims):
    keys = jax.random.split(jax.random.PRNGKey(42), len(irrep_dims))
    return [(2. / d_i) * initializer(k, (group_order - 1, d_i, d_i, 2), jnp.float32) - (1. / d_i)
            for k, d_i in zip(keys, irrep_dims)
        ]


def pad_eye(W_i):
    d_i = W_i.shape[-1]
    eyecm = complex(jnp.eye(d_i), jnp.zeros((d_i, d_i)))
    return jnp.concatenate([jnp.expand_dims(eyecm, 0), W_i], 0)


def total_weight(W, irrep_dims, group_order):
    W_list = []
    for W_i, d_i in zip(W, irrep_dims):
        Wcm = complex(W_i[..., 0], W_i[..., 1])
        W_cm_ext = jnp.reshape(pad_eye(Wcm), (group_order, d_i * d_i))
        W_list.append(W_cm_ext)
    return jnp.concatenate(W_list, -1)



def forward(W, x):
    res = []
    for W_i in W:
        Wcm = complex(W_i[..., 0], W_i[..., 1])
        W_cm_ext = pad_eye(Wcm)

        W_i_x = (jnp.expand_dims(W_cm_ext, 0) * jnp.expand_dims(jnp.expand_dims(x, -1), -1)).sum(1)
        W_i_x_T = jnp.conjugate(jnp.transpose(W_i_x, axes=(0, -1, -2)))

        res.append(W_i_x @ W_i_x_T)
    return res
   


def loss(W, x, y):
    res_x = forward(W, x)
    res_y = forward(W, y)

    res_loss = jnp.zeros(x.shape[0])
    for (res_x_i, res_y_i) in zip(res_x, res_y):
        res_loss += (jnp.abs((res_x_i - res_y_i))**2).mean(-1).mean(-1)
    
    return res_loss / len(res_x)
    

def reg(W, irrep_dims, group_order):
    d_tot = jnp.array(irrep_dims).sum()
    eyecm = (d_tot) * complex(jnp.eye(group_order), jnp.zeros((group_order, group_order)))

    W_tot = total_weight(W, irrep_dims, group_order)
    W_tot_T = jnp.conjugate(jnp.transpose(W_tot, axes=(-1, -2)))
    return (jnp.abs((eyecm - W_tot @ W_tot_T ))**2).mean()



"""
Function recovering the Cayley table from the weights of the model
"""
def get_table(W, group_order):

    res = jnp.zeros((group_order, group_order))
    for g in range(group_order):
        for h in range(group_order):
                
            diffs = jnp.zeros(group_order)
            for W_i in W:
                Wcm = complex(W_i[..., 0], W_i[..., 1])
                W_cm_ext = jnp.conjugate(jnp.transpose(pad_eye(Wcm), axes=(0, -1, -2)))
                W_gh = W_cm_ext[g] @ W_cm_ext[h]
                diffs += (jnp.abs(jnp.expand_dims(W_gh, 0) - W_cm_ext)**2).mean(-1).mean(-1)
            
            res = res.at[g, h].set(jnp.argmin(diffs))

    return res




