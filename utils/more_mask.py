"""
 @Time    : 203/15/19 14:48
 @Author  : TaylorMei
 @Email   : mhy845879017@gmail.com
 
 @Project : iccv
 @File    : more_mask.py
 @Function:
 
"""
import os
import numpy as np
import skimage.io

image_path = '/home/iccd/data/2019/31/image'
mask_path = '/home/iccd/data/2019/31/mask'

imglist = os.listdir(image_path)
for i, imgname in enumerate(imglist):
    print(i, imgname)
    image = skimage.io.imread(os.path.join(image_path, imgname))
    mask = np.zeros_like(image)
    skimage.io.imsave(os.path.join(mask_path, imgname[:-4] + '.png'), mask)

print("ok")
