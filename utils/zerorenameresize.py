"""
 @Time    : 202/17/19 19:04
 @Author  : TaylorMei
 @Email   : mhy845879017@gmail.com
 
 @Project : iccv
 @File    : zerorenameresize.py
 @Function:
 
"""
import os
import random
from skimage import io, transform

IMAGE_DIR = '/media/iccd/TAYLORMEI/yang_ori'
OUTPUT_DIR = '/media/iccd/TAYLORMEI/yang/'

init_number = 5900

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

imglist = os.listdir(IMAGE_DIR)
random.shuffle(imglist)
print("Total {} images will be renamed and resized to 640!".format(len(imglist)))

for i, imgname in enumerate(imglist):
    print(init_number, imgname)
    filestr = imgname.split(".")[0]
    image_path = IMAGE_DIR + "/" + imgname

    image = io.imread(image_path)

    height = image.shape[0]
    width = image.shape[1]
    if height > width:
        fixed_size = (640, 512)
    else:
        fixed_size = (512, 640)
    fixed_image = transform.resize(image, fixed_size, order=3)
    io.imsave(OUTPUT_DIR + str(init_number) + "_" + str(fixed_size[1]) + "x" + str(fixed_size[0]) + ".jpg", fixed_image)

    # io.imsave(OUTPUT_DIR + str(init_number) + ".jpg", image)
    init_number += 1