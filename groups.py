import numpy as np
import torch
import math


class abstr_group():
    def __init__(self):
        self.order = None
        self.cayley_table = None
        self.irrep_dims = None

    def act(self, x):
        g = torch.randint(low=0, high=self.order, size=(1,)).item()
        return x[self.cayley_table[g]]

     
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
        self.cayley_table = self.cayley_table.long()

