"""
  @Time    : 2019-1-2 01:38
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : iccv
  @File    : debug.py
  @Function: 
  
"""
# import skimage.io
#
# path = '/media/iccd/TAYLORMEI/ikea_beforerename/IMG_20190217_091047.jpg'
#
# image = skimage.io.imread(path)
#
# print(image.shape)

# import numpy as np
# import skimage.io
# import skimage.transform
#
# PREDICT_DIR = "/home/iccd/iccv/utils/spatial_train.png"
#
# predict_mask = skimage.io.imread(PREDICT_DIR)
# print(np.max(predict_mask))
# predict_mask = skimage.transform.resize(predict_mask, [512, 512], 0)

import os
import numpy as np
import skimage.io

# mask_path = '/home/iccd/data/msd10/train/mask/5536_640x512.png'
mask_path = '/home/iccd/data/msd10/train/mask/5540_512x640.png'

mask = skimage.io.imread(mask_path)
print(mask.shape)

# input_path = '/media/iccd/TAYLORMEI/saliency_dataset/DUTS/train/image/'
#
# imglist = os.listdir(input_path)
#
# for i, imgname in enumerate(imglist):
#     image = Image.open(input_path + imgname)
#     if image.mode != "RGB":
#         print(imgname)
# print('ok')
