"""
  @Time    : 2019-1-6 22:15
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : iccv
  @File    : reverse.py
  @Function: 
  
"""
import os
import numpy as np
import skimage.io

input_path = "/media/taylor/TAYLORMEI/1.6/112_512x640_f.png"
output_path = "/media/taylor/TAYLORMEI/1.6/112_512x640_b.png"

f = skimage.io.imread(input_path)
b = np.where(f == 255, 0, 255)
skimage.io.imsave(output_path, b.astype(np.uint8))
print("OK!")