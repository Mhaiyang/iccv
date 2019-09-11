"""
 @Time    : 9/9/19 17:00
 @Author  : TaylorMei
 @Email   : mhy845879017@gmail.com
 
 @Project : iccv
 @File    : validate_label.py
 @Function:
 
"""
import os
import numpy as np
import skimage.io

image_path = '/media/iccd/disk/release/MSD/test/image'
mask_path = '/media/iccd/disk/release/MSD/test/mask'
output_path = '/home/iccd/iccv/release/test'

if not os.path.exists(output_path):
    os.mkdir(output_path)

color= [0, 1, 0]

imglist = os.listdir(image_path)
for i, imgname in enumerate(imglist):
    print(i, imgname)
    image = skimage.io.imread(os.path.join(image_path, imgname))
    mask = skimage.io.imread(os.path.join(mask_path, imgname[:-4] + '.png'))

    output = np.zeros_like(image)

    for j in range(image.shape[2]):
        output[:, :, j] = np.where(mask != 0, image[:, :, j] * 0.4 + color[j] * 255 *0.6, image[:, :, j])


    skimage.io.imsave(os.path.join(output_path, imgname), output)