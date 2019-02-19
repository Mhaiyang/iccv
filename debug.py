"""
  @Time    : 2019-1-2 01:38
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : iccv
  @File    : debug.py
  @Function: 
  
"""
# import skimage.io
#
# path = '/media/iccd/TAYLORMEI/ikea_beforerename/IMG_20190217_091047.jpg'
#
# image = skimage.io.imread(path)
#
# print(image.shape)

import numpy as np

a = np.ones([3, 3])
b = a - np.max(a)*0.6
print(a)
print(b)