import os
import shutil

root_dir = '/home/iccd/iccv/plus/ablation_methods'
output_dir = '/home/iccd/iccv/plus/ablation_all'
methods = os.listdir(root_dir)
for method in methods:
    images = os.listdir(os.path.join(root_dir, method))
    for image in images:
        src = os.path.join(root_dir, method, image)
        dst = os.path.join(output_dir, image[:-4] + '_' + method + image[-4:])
        shutil.copy(src, dst)

print('OK!')


