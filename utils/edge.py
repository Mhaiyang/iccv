"""
  @Time    : 2019-1-3 18:09
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : iccv
  @File    : edge.py
  @Function: 
  
"""

import os
import numpy as np
import cv2
import skimage.io
from misc import crf_refine

DATA_DIR = "/home/iccd/data/msd9/train"
IMAGE_DIR = os.path.join(DATA_DIR, "image")
MASK_DIR = os.path.join(DATA_DIR, "mask")
EDGE_DIR = os.path.join(DATA_DIR, "edge")
if not os.path.exists(EDGE_DIR):
    os.mkdir(EDGE_DIR)

masklist = os.listdir(MASK_DIR)
print("Total {} masks will be extracted edge!".format(len(masklist)))

for i, maskname in enumerate(masklist):

    image_path = os.path.join(IMAGE_DIR, maskname[:-4] + '.jpg')
    mask_path = os.path.join(MASK_DIR, maskname)
    edge_path = os.path.join(EDGE_DIR, maskname)

    image = skimage.io.imread(image_path)
    mask = skimage.io.imread(mask_path)

    edge = cv2.Canny(mask, 0, 255)
    edge = crf_refine(image, edge)
    # edge = np.where(edge != 0, 255, 0).astype(np.uint8)

    cv2.imwrite(edge_path, edge)
    print("{}  {}".format(i, edge_path))
