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

detection_path = '/media/iccd/TAYLORMEI/ke/mask_rcnn_white_c_crop_resize/'
image_path = '/media/iccd/TAYLORMEI/ke/color/512448/'
mask_path = '/media/iccd/TAYLORMEI/ke/taylor5_512448/'
output_path = '/media/iccd/TAYLORMEI/ke/green_detection/'

if not os.path.exists(output_path):
    os.mkdir(output_path)

imglist = os.listdir(detection_path)
for i, imgname in enumerate(imglist):
    print(i, imgname)
    detection = skimage.io.imread(detection_path + imgname)
    image = skimage.io.imread(image_path + imgname[:-4] + '.jpg')
    mask = skimage.io.imread(mask_path + imgname)
    print(detection.shape)
    print(mask.shape)

    output = np.zeros_like(detection)

    for j in range(detection.shape[2]):
        if j != 3:
            output[:, :, j] = np.where(mask >= 127.5, image[:, :, j], detection[:, :, j])
        else:
            output[:, :, j] = detection[:, :, j]

    skimage.io.imsave(output_path + imgname, output)