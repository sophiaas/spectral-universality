import torch
from torchvision import transforms, datasets
import numpy as np

from groups import *


class group_dset(torch.utils.data.Dataset):
    def __init__(self, group):
        self.group = group

    def __getitem__(self, index):
        x_re = torch.randn((self.group.order,))
        x_im = torch.randn((self.group.order,))
        x = torch.complex(x_re, x_im)
        y = self.group.act(x)

        return x, y

    def __len__(self):
        return 1000



# class RegCyclic(torch.utils.data.Dataset):
#     def __init__(self, N):
#         self.N = N

#     def __getitem__(self, index):
#         x_re = torch.randn((self.N,))
#         x_im = torch.randn((self.N,))

#         shift = torch.randint(low=1, high=self.N, size=(1,)).item()

#         y_re = torch.roll(x_re, shift)
#         y_im = torch.roll(x_im, shift)

#         x = torch.complex(x_re, x_im)
#         y = torch.complex(y_re, y_im)

#         return x, y

#     def __len__(self):
#         return 1000


# class RegBiCyclic(torch.utils.data.Dataset):
#     def __init__(self, A, B):
#         self.A = A
#         self.B = B

#     def __getitem__(self, index):
#         x_re_A = torch.randn((self.A,))
#         x_im_A = torch.randn((self.A,))

#         shift = torch.randint(low=0, high=self.A, size=(1,)).item()  #The index needs to start from 0 since in a product of groups the identities matter

#         y_re_A = torch.roll(x_re_A, shift)
#         y_im_A = torch.roll(x_im_A, shift)

#         x_re_B = torch.randn((self.B,))
#         x_im_B = torch.randn((self.B,))

#         shift = torch.randint(low=0, high=self.B, size=(1,)).item()  #The index needs to start from 0 since in a product of groups the identities matter

#         y_re_B = torch.roll(x_re_B, shift)
#         y_im_B = torch.roll(x_im_B, shift)


#         x_re = torch.cat([x_re_A, x_re_B], dim=-1)
#         x_im =torch.cat([x_im_A, x_im_B], dim=-1)
#         y_re = torch.cat([y_re_A, y_re_B], dim=-1)
#         y_im =torch.cat([y_im_A, y_im_B], dim=-1)
#         x = torch.complex(x_re, x_im)
#         y = torch.complex(y_re, y_im)

#         return x, y

#     def __len__(self):
#         return 1000
