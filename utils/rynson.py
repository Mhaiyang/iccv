"""
 @Time    : 203/10/19 18:20
 @Author  : TaylorMei
 @Email   : mhy845879017@gmail.com
 
 @Project : iccv
 @File    : rynson.py
 @Function:
 
"""
import os
import numpy as np
import skimage.io

image_path = '/media/iccd/TAYLORMEI/rynson2/image/'
mask_path = '/media/iccd/TAYLORMEI/rynson2/picanet/'

output_path = '/media/iccd/TAYLORMEI/rynson2/color/picanet/'
if not os.path.exists(output_path):
    os.mkdir(output_path)

color = [0, 1, 0, 0]

imglist = os.listdir(image_path)
for i, imgname in enumerate(imglist):
    print(i, imgname)
    image = skimage.io.imread(image_path + imgname)
    mask = skimage.io.imread(mask_path + imgname[:-4] + '.png')

    output = np.zeros_like(image)

    for j in range(image.shape[2]):
        if j != 3:
            output[:, :, j] = np.where(mask >= 127.5, image[:, :, j] * 0.4 + 0.6 * color[j] * 255, image[:, :, j])
        else:
            output[:, :, j] = image[:, :, j]

    skimage.io.imsave(output_path + imgname, output)


