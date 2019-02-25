"""
 @Time    : 202/16/19 16:33
 @Author  : TaylorMei
 @Email   : mhy845879017@gmail.com
 
 @Project : iccv
 @File    : compute_overlap.py
 @Function:
 
"""
import os
import numpy as np
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

image_path = '/home/iccd/data/msd7/train/image/'
mask_json_path = '/home/iccd/data/msd7/train/mask/'
# image_path = '/home/iccd/data/2019/msd5_all/all_images/'
# mask_json_path = '/home/iccd/data/2019/msd5_all/all_masks/'

imglist = os.listdir(image_path)
print(len(imglist))

overlap = np.zeros([256, 256], dtype=np.float64)
tall, wide = 0, 0

for i, imgname in enumerate(imglist):
    print(i, imgname)
    name = imgname.split('.')[0]

    mask = skimage.io.imread(mask_json_path + name + '.png')

    height = mask.shape[0]
    width = mask.shape[1]
    if height > width:
        tall += 1
    else:
        wide += 1
    mask = skimage.transform.resize(mask, [256, 256], order=0)
    mask = np.where(mask != 0, 1, 0).astype(np.float64)
    overlap += mask

overlap = overlap / len(imglist)
overlap_normalized = (overlap - np.min(overlap)) / (np.max(overlap) - np.min(overlap))
skimage.io.imsave('./msd7_train.png', (overlap*255).astype(np.uint8))
skimage.io.imsave('./msd7_train_normalized.png', overlap_normalized)

print(tall, wide)

f, ax = plt.subplots()
sns.set()
ax = sns.heatmap(overlap, ax=ax, cmap=cm.summer, cbar=False)
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.xticks([])
plt.yticks([])
plt.show()


