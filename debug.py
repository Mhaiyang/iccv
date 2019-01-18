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

a = torch.ones(3, 2, 4, 4)
b = torch.ones(3, 2, 4, 4) * 5
c = torch.ones(3, 2, 4, 4) * 2

print(a)
print(b)
print(a.mul(1 - b).mul(c))



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

# logits = torch.rand((3, 4, 4))
# labels = torch.rand((3, 4, 4))
# c = labels.view(-1)
# print(c.size())
# print(logits)
# print(labels)
# for log, lab in zip(logits, labels):
#     print(log.size(), lab.size())


