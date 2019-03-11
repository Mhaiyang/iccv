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
from PIL import Image

input_path = '/media/iccd/TAYLORMEI/saliency_dataset/DUTS/train/image/'

imglist = os.listdir(input_path)

for i, imgname in enumerate(imglist):
    if not imgname.endswith('.jpg'):
        print(imgname)
print('ok')
