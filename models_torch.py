import torch
from torch.nn.parameter import Parameter
from torch import nn

from utils import *


def matmul_complex(t1, t2):
    return torch.view_as_complex(torch.stack((t1.real @ t2.real - t1.imag @ t2.imag, t1.real @ t2.imag + t1.imag @ t2.real),dim=-1))


def pad_eye(W_i):
    d_i = W_i.shape[-1]
    eyecm = torch.complex(torch.eye(d_i) , torch.zeros(d_i, d_i)).to(W_i.device)
    return torch.cat([eyecm.unsqueeze(0), W_i], dim=0)


class spectral_net(nn.Module):
    def __init__(self, group_order, irrep_dims, orthognal_init=False):
        super().__init__()

        self.W = nn.ParameterList() 
        for d_i in irrep_dims:
            W_i = torch.zeros((group_order - 1, d_i, d_i, 2))
            if orthognal_init:
                torch.nn.init.orthogonal_(W_i) 
            else:
                k = 1. / d_i
                W_i.uniform_(-k, k)

            W_i.requires_grad = True    
            self.W.append(Parameter(W_i))

        self.group_order = group_order
        self.irrep_dims = irrep_dims

    def total_weight(self):
        W_list = []
        for W_i in self.W:
            d_i = W_i.shape[-2]
            Wcm = torch.view_as_complex(W_i)
            W_cm_ext = pad_eye(Wcm).view(self.group_order, d_i * d_i)
            W_list.append(W_cm_ext)
        return torch.cat(W_list, dim=-1)




    def forward(self, x):
        res = []
        for W_i in self.W:
            Wcm = torch.view_as_complex(W_i)
            W_cm_ext = pad_eye(Wcm)

            W_i_x = (W_cm_ext.unsqueeze(0) * x.unsqueeze(-1).unsqueeze(-1)).sum(1)
            W_i_x_T = torch.conj(W_i_x.transpose(-2, -1))

            res.append(matmul_complex(W_i_x, W_i_x_T))

        return res
   


    def loss(self, x, y):
        res_x = self(x)
        res_y = self(y)

        res_loss = torch.zeros(x.shape[0]).to(x.device)
        for (res_x_i, res_y_i) in zip(res_x, res_y):
            res_loss += ((res_x_i - res_y_i).abs()**2).mean(-1).mean(-1)
            # AB_T = torch.view_as_real(matmul_complex(res_x_i, torch.conj(res_y_i.transpose(-2, -1))))
            # res_loss -= (torch.diagonal(AB_T, dim1=-2, dim2=-3).sum(-1)**2).sum(-1)
        
        return res_loss / len(res_x)
    

    def reg(self):
        device = self.W[0].device

        # d_diag = []
        # for d_i in self.irrep_dims:
        #     d_diag += [ 1. / (d_i ** 2) ] * (d_i ** 2)  
        # eye_diag = self.group_order * torch.diag(torch.Tensor(d_diag))
        # eyecm = torch.complex(eye_diag, torch.zeros(self.group_order, self.group_order)).to(device)

        d_tot = torch.Tensor(self.irrep_dims).to(device).sum().float()
        eyecm = (d_tot) * torch.complex(torch.eye(self.group_order), torch.zeros(self.group_order, self.group_order)).to(device)


        W_tot = self.total_weight()
        W_tot_T = torch.conj(W_tot.transpose(-1, -2))
        return ((eyecm - matmul_complex(W_tot, W_tot_T) ).abs()**2).mean()




    """
    Function recovering the Cayley table from the weights of the model
    """
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




