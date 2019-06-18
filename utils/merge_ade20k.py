"""
 @Time    : 6/17/19 21:02
 @Author  : TaylorMei
 @Email   : mhy845879017@gmail.com
 
 @Project : iccv
 @File    : merge_ade20k.py
 @Function:
 
"""
import os
import shutil

input_dir = '/home/iccd/ADE20K_2016_07_26/images/validation'
output_dir = '/home/iccd/ADE20K_2016_07_26/images/all_images'

total_number = 0

sub_dir = os.listdir(input_dir)
for i, sub_dir_name in enumerate(sub_dir):
    print(sub_dir_name)
    start_number = total_number
    sub_path = os.path.join(input_dir, sub_dir_name)
    sub_sub_dir = os.listdir(sub_path)
    for sub_sub_name in sub_sub_dir:
        sub_sub_path = os.path.join(sub_path, sub_sub_name)
        imglist = os.listdir(sub_sub_path)
        for imgname in imglist:
            img_path = os.path.join(sub_sub_path, imgname)
            if imgname.endswith('.jpg'):
                shutil.copyfile(img_path, os.path.join(output_dir, imgname))
                total_number += 1
    end_number = total_number
    print(end_number - start_number)

print("OJBK!")