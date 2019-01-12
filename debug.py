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
print(truth)
print(1 - truth)

pred = torch.rand(3, 1, 4, 4)
truth_mask = torch.where(truth > 0.5, torch.tensor(1.), torch.tensor(0.))
pred_mask = torch.where(pred > 0.5, torch.tensor(1.), torch.tensor(2.))

TP = torch.where(pred_mask == truth_mask, torch.tensor(1.), torch.tensor(0.))

print(truth_mask)
print(pred_mask)
print(TP)
