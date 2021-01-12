"""
 @Time    : 202/17/19 18:21
 @Author  : TaylorMei
 @Email   : mhy845879017@gmail.com
 
 @Project : iccv
 @File    : thirdmakemask.py
 @Function:
 
"""
import os
import shutil
import numpy as np
import skimage.io

# image_path = "/home/iccd/data/msd9/train/image/"
# mask_json_path = "/home/iccd/data/2019/mask_json_true/"
# destination_path1 = "/home/iccd/data/msd9/train/mask_json/"
# destination_path2 = "/home/iccd/data/msd9/train/mask/"
# image_path = "/home/iccd/data/2019/msd9_all/all_images/"
# mask_json_path = "/home/iccd/data/2019/mask_json_true/"
# destination_path2 = "/home/iccd/data/2019/msd9_all/all_masks/"

mask_json_path = "/home/iccd/data/msd9_c/test/correction/cor_json/"
destination_path1 = "/home/iccd/data/msd9_c/test/correction/cor_mask_json/"
destination_path2 = "/home/iccd/data/msd9_c/test/correction/cor_mask/"

imglist = os.listdir(mask_json_path)

for i, imgname in enumerate(imglist):
    if imgname.endswith('.json'):
        print(i, imgname)
        name = imgname.split(".")[0]

        mask = skimage.io.imread(mask_json_path + name + "_json/label8.png")
        mask = np.where(mask != 0, 255, 0).astype(np.uint8)

        skimage.io.imsave(destination_path2 + name + ".png", mask)
        shutil.copyfile(mask_json_path + name + "_json/label8.png", destination_path1 + name + '.png')

print("ok!")