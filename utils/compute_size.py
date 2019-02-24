"""
 @Time    : 202/16/19 14:17
 @Author  : TaylorMei
 @Email   : mhy845879017@gmail.com
 
 @Project : iccv
 @File    : compute_size.py
 @Function:
 
"""
import os
import numpy as np
import skimage.io
from misc import data_write

image_path = '/home/iccd/data/2019/msd5_all/all_images'
mask_json_path = '/home/iccd/data/2019/msd5_all/all_masks/'

imglist = os.listdir(image_path)
print(len(imglist))

output = []

for i, imgname in enumerate(imglist):
    print(i, imgname)
    name = imgname.split('.')[0]

    mask = skimage.io.imread(mask_json_path + name + '.png')
    mask = np.where(mask != 0, 1, 0).astype(np.uint8)

    height = mask.shape[0]
    width = mask.shape[1]
    total_area = height * width
    if total_area != 640*512:
        print('size error!')

    mirror_area = np.sum(mask)
    proportion = mirror_area / total_area
    output.append(proportion)
data_write('./proportion.xlsx', [output])


