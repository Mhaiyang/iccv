"""
 @Time    : 202/20/19 09:41
 @Author  : TaylorMei
 @Email   : mhy845879017@gmail.com
 
 @Project : iccv
 @File    : compute_contrast.py
 @Function:
 
"""
import os
import numpy as np
import cv2
import skimage.io
from misc import data_write

# image_path = '/home/iccd/data/2019/msd9_all/all_images/'
# mask_path = '/home/iccd/data/2019/msd9_all/all_masks/'
# image_path = '/home/iccd/data/MSRA10K/DUT-OMRON/image/'
# mask_path = '/home/iccd/data/MSRA10K/DUT-OMRON/mask/'
image_path = '/home/iccd/data/SBU-all/image/'
mask_path = '/home/iccd/data/SBU-all/mask/'

imglist = os.listdir(image_path)

chi_sq_color = []


def chi2(arr1, arr2):

    return np.sum((arr1 - arr2)**2 / (arr1 + arr2 + np.finfo(np.float).eps))


for i, imgname in enumerate(imglist):
    print(i, imgname)

    image = skimage.io.imread(image_path + imgname)

    # name = imgname.split('.')[0]
    name = imgname[:-4]
    mask = skimage.io.imread(mask_path + name + '.png')
    mask_f = np.where(mask != 0, 1, 0).astype(np.uint8)
    mask_b = np.where(mask == 0, 1, 0).astype(np.uint8)

    if np.sum(mask_f) == 0:
        print('llllllllllllllllllllllllllllllllllllllllllllllll')
        continue

    hist_f_r = cv2.calcHist([image], [0], mask_f, [256], [0,256])
    hist_f_g = cv2.calcHist([image], [1], mask_f, [256], [0,256])
    hist_f_b = cv2.calcHist([image], [2], mask_f, [256], [0,256])
    hist_b_r = cv2.calcHist([image], [0], mask_b, [256], [0,256])
    hist_b_g = cv2.calcHist([image], [1], mask_b, [256], [0,256])
    hist_b_b = cv2.calcHist([image], [2], mask_b, [256], [0,256])

    chi_sq_r = chi2(hist_f_r.flatten()/np.sum(mask_f), hist_b_r.flatten()/np.sum(mask_b))
    chi_sq_g = chi2(hist_f_g.flatten()/np.sum(mask_f), hist_b_g.flatten()/np.sum(mask_b))
    chi_sq_b = chi2(hist_f_b.flatten()/np.sum(mask_f), hist_b_b.flatten()/np.sum(mask_b))

    chi_sq_color.append(((chi_sq_r + chi_sq_g + chi_sq_b) / 3).item())

chi_sq_color = np.array(chi_sq_color)
chi_sq_color = (chi_sq_color - np.min(chi_sq_color)) / (np.max(chi_sq_color - np.min(chi_sq_color)))

print(chi_sq_color)
data_write('./shadow_chi_sq.xlsx', [chi_sq_color])
