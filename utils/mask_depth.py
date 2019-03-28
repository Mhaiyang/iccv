"""
 @Time    : 203/18/19 23:31
 @Author  : TaylorMei
 @Email   : mhy845879017@gmail.com
 
 @Project : iccv
 @File    : mask_depth.py
 @Function:
 
"""
import os
import numpy as np
import skimage.io
import matplotlib.image
import matplotlib.pyplot as plt
plt.set_cmap("jet")

mask_path = '/media/iccd/TAYLORMEI/figure1/taylor5_480'
depth_path = '/media/iccd/TAYLORMEI/figure1/depth'
output_path = '/media/iccd/TAYLORMEI/figure1/masked_depth'
if not os.path.exists(output_path):
    os.mkdir(output_path)

image_list = os.listdir(depth_path)
for i, imgname in enumerate(image_list):
    print(i, imgname)
    mask = skimage.io.imread(mask_path + '/' + imgname)
    depth = skimage.io.imread(depth_path + '/' + imgname)
    if i == 0:
        for j in range(3):
            depth[:, :, j] = np.where(mask != 0, depth[50, 50, j], depth[:, :, j])
        matplotlib.image.imsave(output_path + "/" + imgname, depth)
    # if i == 1:
    #     for j in range(3):
    #         depth[:, :, j] = np.where(mask != 0, depth[20, 400, j], depth[:, :, j])
    #     matplotlib.image.imsave(output_path + "/" + imgname, depth)
print("ok!")
# for i, imgname in enumerate(image_list):
#     print(i, imgname)
#     mask = skimage.io.imread(mask_path + '/' + imgname)
#     depth = skimage.io.imread(depth_path + '/' + imgname)
#     for j in range(3):
#         depth[:, :, j] = np.where(mask != 0, depth[240, 240, j], depth[:, :, j])
#     matplotlib.image.imsave(output_path + "/" + imgname, depth)
# print("ok!")