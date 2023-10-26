import numpy as np
import matplotlib
import torch
import torch.nn.functional as F

import math
import itertools as it


def complex_array_to_rgb(X, theme='light', rmax=None):
    '''Takes an array of complex number and converts it to an array of [r, g, b],
    where phase gives hue and saturaton/value are given by the absolute value.
    Especially for use with imshow for complex plots.'''
    absmax = rmax or np.abs(X).max()
    Y = np.zeros(X.shape + (3,), dtype='float')
    Y[..., 0] = np.angle(X) / (2 * np.pi) % 1
    if theme == 'light':
        Y[..., 1] = np.clip(np.abs(X) / absmax, 0, 1)
        Y[..., 2] = 1
    elif theme == 'dark':
        Y[..., 1] = 1
        Y[..., 2] = np.clip(np.abs(X) / absmax, 0, 1)
    Y = matplotlib.colors.hsv_to_rgb(Y)
    return Y


def matmul_complex(t1, t2):
    return torch.view_as_complex(torch.stack((t1.real @ t2.real - t1.imag @ t2.imag, t1.real @ t2.imag + t1.imag @ t2.real),dim=-1))


def kron_batched(a, b):
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    res_real = torch.view_as_real(res).reshape(siz0 + siz1 + (2,))
    return torch.view_as_complex(res_real)


def pad_eye(W_i):
    d_i = W_i.shape[-1]
    eyecm = torch.complex(torch.eye(d_i) , torch.zeros(d_i, d_i)).to(W_i.device)
    return torch.cat([eyecm.unsqueeze(0), W_i], dim=0)


def perm_matrices(n): 
    fact_n = math.factorial(n)
    res = np.zeros((fact_n, n, n))
    for idx, perm in enumerate(it.permutations(range(n))):
        for i in range(n):
            res[idx, i, perm[i]] = 1.
         
    return torch.FloatTensor(res)


def perm_frobenius(A, B, perms):
    #This does not support batching
    B_hot = F.one_hot(B.long()).unsqueeze(-1).unsqueeze(0).float()
    B_perm = torch.argmax((perms.unsqueeze(1).unsqueeze(1) @  B_hot).squeeze(-1), dim=-1).float()
                                                                                                                        
    diffs = ((A.unsqueeze(0) - ( perms.transpose(-2,-1) @ B_perm @ perms) )**2).mean(-1).mean(-1)
    return torch.min(diffs)


# def is_isomorphic(A, B):
#     perms = list(it.permutations(range(A.shape[0])))
#     diffs = torch.zeros(len(perms)).to(A.device) 
#     for idx, p in enumerate(perms):
#         p = list(p)
#         mapping = {i: a for i, a in enumerate(p)}
#         tmp = torch.zeros_like(A)
#         for i, row in enumerate(A):
#             for j, val in enumerate(row):
#                 tmp[j, i] = mapping[val.item()]
#         tmp = tmp[p][:, p]
#         diffs[idx] = ((tmp - B)**2).mean()
#     return torch.min(diffs)


def gen_partitions(n):
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1]

def hook_length(P, N):
    P = sorted(P, reverse=True)
    res = 1
    for i in range(len(P)):
        for j in range(P[i]):
            cells_row = P[i] - j
            cells_col = len([k for k in P[i:] if (k >= j + 1)])
            res *= cells_row + cells_col - 1
    return int(float(math.factorial(N)) / float(res))



if __name__ == "__main__":
    for i in range(1, num_ep):
        print(f'Epoch {i}')
        train(i, train_loader)

