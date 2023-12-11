import numpy as np
import matplotlib
import math
import itertools as it
from torch.nn.functional import one_hot
import torch


"""
Plotting method for complex numbers 
"""
def complex_array_to_rgb(X, theme='light', rmax=None):
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


"""
Generate permutation matrices
"""
def perm_matrices(n): 
    fact_n = math.factorial(n)
    res = np.zeros((fact_n, n, n))
    for idx, perm in enumerate(it.permutations(range(n))):
        for i in range(n):
            res[idx, i, perm[i]] = 1.
    return res


"""
Permutation-invariant Frobenius distance for matrieces. 
Does not support batching. 
"""
def perm_frobenius(A, B, perms, group_order):
    B_hot = one_hot(torch.Tensor(B).long(), group_order).numpy().astype(int)
    B_hot = np.expand_dims(np.expand_dims(B_hot, -1), 0).astype(float)
    B_perm = np.argmax(np.squeeze(np.expand_dims(np.expand_dims(perms, 1), 1) @  B_hot, -1), -1).astype(float)
                                                                                                                        
    diffs = ((np.expand_dims(A, 0) - ( np.transpose(perms, axes=(0,-1,-2)) @ B_perm @ perms) )**2).mean(-1).mean(-1)
    return np.min(diffs)


"""
Generate partitions of a number. Taken from https://stackoverflow.com/questions/10035752/elegant-python-code-for-integer-partitioning
"""
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


"""
Hook length formula for dimensions of irreps of the symmetric group
"""
def hook_length(P, N):
    P = sorted(P, reverse=True)
    res = 1
    for i in range(len(P)):
        for j in range(P[i]):
            cells_row = P[i] - j
            cells_col = len([k for k in P[i:] if (k >= j + 1)])
            res *= cells_row + cells_col - 1
    return int(float(math.factorial(N)) / float(res))




