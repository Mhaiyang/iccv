"""
  @Time    : 2019-1-2 01:38
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : iccv
  @File    : debug.py
  @Function: 
  
"""
import torch
import torch.nn.functional as F

# truth = torch.ones(3, 1, 4, 5)
#
# N_p = torch.tensor(torch.sum(torch.sum(truth, -1), -1), dtype=torch.float)
# N = torch.tensor(torch.numel(truth[0, :, :, :]), dtype=torch.float).expand_as(N_p)
# N_n = N - N_p
#
# print(N_p)
# print(N)
# print(N_n)

truth = torch.rand(3, 1, 4, 4)

N_p = torch.tensor(torch.sum(torch.sum(truth, -1), -1), dtype=torch.float).unsqueeze(-1).unsqueeze(-1).expand_as(truth)

print(N_p)
print(N_p.shape)
