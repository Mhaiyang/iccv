"""
 @Time    : 4/30/21 19:35
 @Author  : TaylorMei
 @Email   : mhy666@mail.dlut.edu.cn
 
 @Project : iccv
 @File    : rename.py
 @Function:
 
"""
import os
import shutil

input_path = '/media/iccd/disk1/mirror/data/ylt/test/image'
output_path = '/home/iccd/Desktop/more_mirror_image'

images = os.listdir(input_path)

for i, image in enumerate(images):
    src = os.path.join(input_path, image)
    dst = os.path.join(output_path, image[:-4] + '_a' + image[-4:])

    shutil.copy(src, dst)

print('ok')