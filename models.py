import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
from torch import nn
import math

from utils import *


class spectral_net(nn.Module):
    def __init__(self, group_order, irrep_dims):
        super().__init__()

        self.W = nn.ParameterList()
        for d_i in irrep_dims:
            W_i = torch.zeros((group_order - 1, d_i, d_i, 2))
            k = 1. / d_i
            W_i.uniform_(-k, k)
            W_i.requires_grad = True    
            self.W.append(Parameter(W_i))

        self.group_order = group_order
        self.irrep_dims = irrep_dims





    def forward(self, x):
        res = []
        for W_i in self.W:
            
            Wcm = torch.view_as_complex(W_i)
            W_cm_ext = pad_eye(Wcm)

            W_i_x = (W_cm_ext.unsqueeze(0) * x.unsqueeze(-1).unsqueeze(-1)).sum(1)
            W_i_x_T = torch.conj(W_i_x.transpose(-2, -1))


            res.append(kron_batched(W_i_x, W_i_x_T))

        return res
   


    def loss(self, x, y):
        res_x = self(x)
        res_y = self(y)

        res_loss = torch.zeros(x.shape[0]).to(x.device)
        for (res_x_i, res_y_i) in zip(res_x, res_y):
            res_loss += ((res_x_i - res_y_i).abs()**2).mean(-1).mean(-1)
        
        return res_loss / len(res_x)
    

    def reg(self):
        device = self.W[0].device
        res_reg = 0.
        eyecm = torch.complex(torch.eye(self.group_order), torch.zeros(self.group_order, self.group_order)).to(device)

        for W_i in self.W:
            
            d_i = W_i.shape[-2]
            Wcm = torch.view_as_complex(W_i)
            W_cm_ext = pad_eye(Wcm).view(self.group_order, d_i * d_i)
            W_cm_ext_T = torch.conj(W_cm_ext.transpose(-1, -2))

            res_reg += (( (d_i **2) * eyecm - matmul_complex(W_cm_ext, W_cm_ext_T) ).abs()**2).mean()

        return res_reg / len(self.W)




    def get_table(self):
        device = self.W[0].device 

        d = self.group_order
        res = torch.zeros((d, d)).to(device)
        for g in range(d):
            for h in range(d):
                 
                diffs = torch.zeros(d).to(device)
                for W_i in self.W:
                    Wcm = torch.view_as_complex(W_i)
                    W_cm_ext = torch.conj(pad_eye(Wcm).transpose(-2, -1))
                    W_gh = matmul_complex(W_cm_ext[g], W_cm_ext[h])
                    diffs += ((W_gh.unsqueeze(0) - W_cm_ext).abs()**2).mean(-1).mean(-1)
                
                res[g, h] = torch.argmin(diffs)
        
        # pad = torch.arange(0, d).to(device)
        # res[-1, :] = pad
        # res[:, -1] = pad
        return res.detach()


