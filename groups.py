import numpy as np
import torch
import math
import itertools as it

from utils import *

class abstr_group():
    def __init__(self):
        self.order = None
        self.cayley_table = None
        self.irrep_dims = None

    def act(self, x):
        g = torch.randint(low=0, high=self.order, size=(1,)).item()
        return x[self.cayley_table[g]]
    
    def check_dims(self):
        irrep_dims = torch.tensor(self.irrep_dims)
        assert (irrep_dims**2).sum().item() == self.order


     
class cyclic(abstr_group):
    def __init__(self, N): 
        self.order = N
        self.irrep_dims = [1]*N

        self.cayley_table = torch.zeros(N, N)
        for i in range(N):
            self.cayley_table[i] = torch.roll(torch.arange(0, N), -i)
        self.cayley_table = self.cayley_table.long()


    

class dihedral(abstr_group):
    def __init__(self, N):
        self.order = 2*N

        if N % 2 == 0:
            self.irrep_dims = [1]*4 + [2]*int(N / 2 - 1)
        else:
            self.irrep_dims = [1]*2 + [2]*int((N - 1) / 2)


        reflection = torch.Tensor([0] + [N-i for i in range(1, N)]).long()
        self.group_elems = torch.zeros(2*N, N)
        for i in range(N):
            cycle = torch.roll(torch.arange(0, N), i)
            self.group_elems[i] = cycle
            self.group_elems[N+i] = cycle[reflection]
        self.group_elems = self.group_elems.long()

        self.cayley_table = torch.zeros(2*N, 2*N)
        for i in range(2*N):
            for j in range(2*N):
                comp = self.group_elems[i][self.group_elems[j]]
                self.cayley_table[i, j] = torch.argmin( ((comp.unsqueeze(0) - self.group_elems)**2).sum(-1) )

        if N == 2:
            C = [
                [0, 1, 2, 3], 
                [1, 0, 3, 2],
                [2, 3, 0, 1],
                [3, 2, 1, 0]
                 ]
            self.cayley_table = torch.Tensor(C)

        self.cayley_table = self.cayley_table.long()



class symmetric(abstr_group):
    def __init__(self, N):
        self.order = math.factorial(N)

        self.irrep_dims = [hook_length(P, N) for P in list(gen_partitions(N))]
        
        self.group_elems = torch.zeros(self.order, N)
        for i, perm in enumerate(it.permutations(range(N))):
            self.group_elems[i] = torch.Tensor(list(perm))
        self.group_elems = self.group_elems.long()

        self.cayley_table = torch.zeros(self.order, self.order)
        for i in range(self.order):
            for j in range(self.order):
                comp = self.group_elems[i][self.group_elems[j]]
                self.cayley_table[i, j] = torch.argmin( ((comp.unsqueeze(0) - self.group_elems)**2).sum(-1) )

        self.cayley_table = self.cayley_table.long()
