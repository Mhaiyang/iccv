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

a = torch.ones(3, 2)
b = torch.ones(1, 2)

print(a)
print(b)

c = torch.cat([a, b], 0)
print(c)