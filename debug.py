"""
  @Time    : 2019-1-2 01:38
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : iccv
  @File    : debug.py
  @Function: 
  
"""
import skimage.io

path = '/media/iccd/TAYLORMEI/ikea_beforerename/IMG_20190217_091047.jpg'

image = skimage.io.imread(path)

print(image.shape)