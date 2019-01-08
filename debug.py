"""
  @Time    : 2019-1-2 01:38
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : iccv
  @File    : debug.py
  @Function: 
  
"""
import skimage.io
import numpy as np

input_path = "/home/iccd/data/MSD9/test/image/4614_512x640.jpg"
image = skimage.io.imread(input_path)
image = image/255.0
output = image * image
output = (output *255)
skimage.io.imsave('/home/iccd/Desktop/output.jpg', output)
print(np.max(image))
