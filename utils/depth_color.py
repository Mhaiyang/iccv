"""
 @Time    : 203/7/19 15:03
 @Author  : TaylorMei
 @Email   : mhy845879017@gmail.com
 
 @Project : iccv
 @File    : depth_color.py
 @Function:
 
"""
import os
import skimage.io
import matplotlib.image
import matplotlib.pyplot as plt
plt.set_cmap("jet")

input_path = '/media/iccd/大白菜U盘/RgbDepth'
output_path = '/media/iccd/大白菜U盘/color'
if not os.path.exists(output_path):
    os.mkdir(output_path)

depth_list = os.listdir(input_path)
for i, imgname in enumerate(depth_list):
    print(i, imgname)
    depth = skimage.io.imread(input_path + '/' + imgname)
    matplotlib.image.imsave(output_path + "/" + imgname, depth)
print("ok!")