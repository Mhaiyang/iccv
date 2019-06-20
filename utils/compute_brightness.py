"""
 @Time    : 202/20/19 09:41
 @Author  : TaylorMei
 @Email   : mhy845879017@gmail.com
 
 @Project : iccv
 @File    : compute_brightness.py
 @Function:
 
"""
import os
import numpy as np
import cv2
import skimage.io
import skimage.color
from misc import data_write

image_path = '/home/iccd/data/2019/msd9_all/all_images/'
mask_path = '/home/iccd/data/2019/msd9_all/all_masks/'

imglist = os.listdir(image_path)

inside_mirror = []
outside_mirror = []

for i, imgname in enumerate(imglist):
    print(i, imgname)

    image = skimage.io.imread(image_path + imgname)
    image = skimage.color.rgb2gray(image)

    name = imgname[:-4]
    mask = skimage.io.imread(mask_path + name + '.png')
    mask_f = np.where(mask != 0, 1, 0).astype(np.uint8)
    mask_b = np.where(mask == 0, 1, 0).astype(np.uint8)

    if np.sum(mask_f) == 0:
        print('llllllllllllllllllllllllllllllllllllllllllllllll')
        continue

    mirror_region = np.where(mask_f == 1, image, 0).astype(np.float32)
    non_mirror_region = np.where(mask_b == 1, image, 0).astype(np.float32)

    inside_mirror.append(np.sum(mirror_region) / np.sum(mask_f))
    outside_mirror.append(np.sum(non_mirror_region) / np.sum(mask_b))

mean_inside_mirror = np.sum(inside_mirror) / len(inside_mirror)
mean_outside_mirror = np.sum(outside_mirror) / len(outside_mirror)

print(mean_inside_mirror)
print(mean_outside_mirror)
