"""
 @Time    : 203/22/19 10:40
 @Author  : TaylorMei
 @Email   : mhy845879017@gmail.com
 
 @Project : iccv
 @File    : mask_detection.py
 @Function:
 
"""
"""
 @Time    : 203/12/19 19:00
 @Author  : TaylorMei
 @Email   : mhy845879017@gmail.com

 @Project : iccv
 @File    : mask_mirror.py
 @Function:

"""
import os
import numpy as np
import skimage.io

image_path = '/media/iccd/TAYLORMEI/ke/mask_rcnn_crop_resize/'
mask_path = '/media/iccd/TAYLORMEI/ke/taylor5_512448/'
output_path = '/media/iccd/TAYLORMEI/ke/mask_detection/'

if not os.path.exists(output_path):
    os.mkdir(output_path)

color = [0, 1, 0]

imglist = os.listdir(image_path)
for i, imgname in enumerate(imglist):
    print(i, imgname)
    image = skimage.io.imread(image_path + imgname)
    mask = skimage.io.imread(mask_path + imgname[:-4] + '.png')
    print(image.shape)
    print(mask.shape)

    output = np.zeros_like(image)

    for j in range(image.shape[2]):
        if j != 3:
            output[:, :, j] = np.where(mask >= 127.5, image[:, :, j] * 0.4 + 0.6 * color[j] * 255, image[:, :, j])
        else:
            output[:, :, j] = image[:, :, j]

    skimage.io.imsave(output_path + imgname, output)