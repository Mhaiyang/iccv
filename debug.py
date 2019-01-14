"""
  @Time    : 2019-1-2 01:38
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : iccv
  @File    : debug.py
  @Function: 
  
"""
import torch
import torch.nn as nn

from torch.autograd import Variable

# truth = torch.ones(3, 1, 4, 5)
#
# N_p = torch.tensor(torch.sum(torch.sum(truth, -1), -1), dtype=torch.float)
# N = torch.tensor(torch.numel(truth[0, :, :, :]), dtype=torch.float).expand_as(N_p)
# N_n = N - N_p
#
# print(N_p)
# print(N)
# print(N_n)

# sigmoid = nn.Sigmoid()
# target = Variable(torch.rand(3))
#
# weight = target.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(3, 1, 4, 4)
# print(target)
# print(weight)

a = torch.rand(4)
b = torch.rand(4)
c = a * b
print(a)
print(b)
print(c)


