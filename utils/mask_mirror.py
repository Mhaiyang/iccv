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

# image_path = '/media/iccd/TAYLORMEI/Depth-Prediction/nyu_depth_v2/mirror/image/'
# mask_path = '/media/iccd/TAYLORMEI/Depth-Prediction/nyu_depth_v2/mirror/mirror_map_448/'
# output_path = '/media/iccd/TAYLORMEI/Depth-Prediction/nyu_depth_v2/mirror/mask_mirror/448/'
image_path = '/media/iccd/TAYLORMEI/ban/image/'
mask_path = '/media/iccd/TAYLORMEI/ban/taylor5_416/'
output_path = '/media/iccd/TAYLORMEI/ban/mask_mirror_white/'

if not os.path.exists(output_path):
    os.mkdir(output_path)

imglist = os.listdir(image_path)
for i, imgname in enumerate(imglist):
    print(i, imgname)
    image = skimage.io.imread(image_path + imgname)
    mask = skimage.io.imread(mask_path + imgname[:-4] + '.png')

    masked_image = np.zeros_like(image)
    for j in range(image.shape[2]):
        masked_image[:, :, j] = np.where(mask >= 127.5, 150, image[:, :, j])
    skimage.io.imsave(output_path + imgname, masked_image)