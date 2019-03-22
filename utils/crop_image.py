"""
 @Time    : 203/21/19 17:11
 @Author  : TaylorMei
 @Email   : mhy845879017@gmail.com
 
 @Project : iccv
 @File    : crop_image.py
 @Function:
 
"""
import os
import numpy as np
import skimage.io

input_path = '/media/iccd/TAYLORMEI/depth/image'
output_path = '/media/iccd/TAYLORMEI/depth/crop'
if not os.path.exists(output_path):
    os.mkdir(output_path)

imglist = os.listdir(input_path)
for i, imgname in enumerate(imglist):
    print(i, imgname)
    image = skimage.io.imread(os.path.join(input_path, imgname))
    print(np.sum(image[80, :, :]))
    for j in range(640):
        if np.sum(image[j, :, :]) !=0 and np.sum(image[j, :, :]) !=367200:
            print(j)
            break
    # crop = image[80:560, :, :]
    # skimage.io.imsave(os.path.join(output_path, imgname), crop)

