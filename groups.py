import numpy as np
from numpy import random
import math
import itertools as it

from utils import *


"""
Abstract class representing a finite group. 
"""
class abstr_group():
    def __init__(self, order, cayley_table, irrep_dims):
        self.order = order
        self.cayley_table = cayley_table
        self.irrep_dims = irrep_dims

    def act(self, x):
        g = random.randint(low=0, high=self.order)
        return x[self.cayley_table[g]]
    
    def check_dims(self):
        irrep_dims = np.array(self.irrep_dims)
        assert (irrep_dims**2).sum() == self.order


"""
Cyclic groups
"""
class cyclic(abstr_group):
    def __init__(self, N): 
        self.order = N
        self.irrep_dims = [1]*N

        self.cayley_table = np.zeros((N, N))
        for i in range(N):
            self.cayley_table[i] = np.roll(np.arange(0, N), -i)
        self.cayley_table = self.cayley_table.astype(int)


    
"""
Dihedral groups
"""
class dihedral(abstr_group):
    def __init__(self, N):
        self.order = 2*N

        if N % 2 == 0:
            self.irrep_dims = [1]*4 + [2]*int(N / 2 - 1)
        else:
            self.irrep_dims = [1]*2 + [2]*int((N - 1) / 2)


        reflection = np.array([0] + [N-i for i in range(1, N)]).astype(int)
        self.group_elems = np.zeros((2*N, N))
        for i in range(N):
            cycle = np.roll(np.arange(0, N), i)
            self.group_elems[i] = cycle
            self.group_elems[N+i] = cycle[reflection]
        self.group_elems = self.group_elems.astype(int)

        self.cayley_table = np.zeros((2*N, 2*N))
        for i in range(2*N):
            for j in range(2*N):
                comp = self.group_elems[i][self.group_elems[j]]
                self.cayley_table[i, j] = np.argmin( ((np.expand_dims(comp, 0) - self.group_elems)**2).sum(-1) )

        if N == 2:
            C = [
                [0, 1, 2, 3], 
                [1, 0, 3, 2],
                [2, 3, 0, 1],
                [3, 2, 1, 0]
                 ]
            self.cayley_table = np.array(C)

        self.cayley_table = self.cayley_table.astype(int)


"""
Symmetric groups
"""
class symmetric(abstr_group):
    def __init__(self, N):
        self.order = math.factorial(N)

        self.irrep_dims = [hook_length(P, N) for P in list(gen_partitions(N))]
        
        self.group_elems = np.zeros((self.order, N))
        for i, perm in enumerate(it.permutations(range(N))):
            self.group_elems[i] = np.array(list(perm))
        self.group_elems = self.group_elems.astype(int)

        self.cayley_table = np.zeros((self.order, self.order))
        for i in range(self.order):
            for j in range(self.order):
                comp = self.group_elems[i][self.group_elems[j]]
                self.cayley_table[i, j] = np.argmin( ((np.expand_dims(comp, 0) - self.group_elems)**2).sum(-1) )

        self.cayley_table = self.cayley_table.astype(int)



"""
Direct product of groups
"""
def direct_product(group_1, group_2): 
    order_1 = group_1.order    
    order_2 = group_2.order
    order_res = order_1 * order_2

    cayley_1 = group_1.cayley_table
    cayley_2 = group_2.cayley_table
    cayley_res = np.zeros((order_res, order_res))
    for i_1 in range(order_1):
        for i_2 in range(order_2):
                for j_1 in range(order_1):
                    for j_2 in range(order_2):
                        g_1 = cayley_1[i_1, j_1]
                        g_2 = cayley_2[i_2, j_2]
                        cayley_res[i_1*order_2 + i_2, j_1*order_2 + j_2] = g_1*order_2 + g_2
    cayley_res = cayley_res.astype(int)

    irrep_dims_1 = group_1.irrep_dims
    irrep_dim_2 = group_2.irrep_dims
    irrep_dims_res = []
    for d_1 in irrep_dims_1:
        for d_2 in irrep_dim_2:
            irrep_dims_res.append(d_1 * d_2)

    return abstr_group(order_res, cayley_res, irrep_dims_res)



"""
Semidirect product of groups
"""
def direct_product(group_1, group_2, phi):  
    #phi: (group2, group1) 
    order_1 = group_1.order    
    order_2 = group_2.order
    order_res = order_1 * order_2


    cayley_1 = group_1.cayley_table
    cayley_2 = group_2.cayley_table
    cayley_res = np.zeros((order_res, order_res))
    for i_1 in range(order_1):
        for i_2 in range(order_2):
                for j_1 in range(order_1):
                    for j_2 in range(order_2):
                        g_1 = cayley_1[i_1, j_1]
                        g_2 = cayley_2[i_2, j_2]
                        cayley_res[i_1*order_2 + i_2, j_1*order_2 + j_2] = g_1*order_2 + g_2
    cayley_res = cayley_res.astype(int)

    irrep_dims_1 = group_1.irrep_dims
    irrep_dim_2 = group_2.irrep_dims
    irrep_dims_res = []
    for d_1 in irrep_dims_1:
        for d_2 in irrep_dim_2:
            irrep_dims_res.append(d_1 * d_2)

    return abstr_group(order_res, cayley_res, irrep_dims_res)
