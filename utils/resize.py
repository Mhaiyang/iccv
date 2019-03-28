"""
 @Time    : 203/22/19 10:32
 @Author  : TaylorMei
 @Email   : mhy845879017@gmail.com
 
 @Project : iccv
 @File    : resize.py
 @Function:
 
"""

import os
import random
from skimage import io, transform

IMAGE_DIR = '/media/iccd/TAYLORMEI/ban/mask_rcnn_white_c_crop'
OUTPUT_DIR = '/media/iccd/TAYLORMEI/ban/mask_rcnn_white_c_crop_resize/'

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

imglist = os.listdir(IMAGE_DIR)
random.shuffle(imglist)
print("Total {} images will be renamed and resized to 640!".format(len(imglist)))

for i, imgname in enumerate(imglist):
    print(imgname)
    image_path = IMAGE_DIR + "/" + imgname

    image = io.imread(image_path)

    fixed_size = (466, 480)

    fixed_image = transform.resize(image, fixed_size, order=3)
    io.imsave(OUTPUT_DIR + imgname, fixed_image)
