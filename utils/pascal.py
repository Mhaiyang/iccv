"""
 @Time    : 203/10/19 15:31
 @Author  : TaylorMei
 @Email   : mhy845879017@gmail.com
 
 @Project : iccv
 @File    : pascal.py
 @Function:
 
"""
import os
import numpy as np
import skimage.io

input_path = "/home/iccd/data/MSRA10K/PASCAL-S/masks/"
output_path = "/home/iccd/data/MSRA10K/PASCAL-S/mask/"

imglist = os.listdir(input_path)
for i, imgname in enumerate(imglist):
    print(i, imgname)
    mask = skimage.io.imread(input_path + imgname)
    print(np.max(mask))
    mask = np.where(mask >= 127.5, 255, 0).astype(np.uint8)
    mask = skimage.io.imsave(output_path + imgname, mask)
