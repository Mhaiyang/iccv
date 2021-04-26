import os
import shutil

root_dir = '/home/iccd/iccv/visual/methods'
output_dir = '/home/iccd/iccv/visual/all'
methods = os.listdir(root_dir)
for method in methods:
    images = os.listdir(os.path.join(root_dir, method))
    for image in images:
        src = os.path.join(root_dir, method, image)
        dst = os.path.join(output_dir, image[:-4] + '_' + method + image[-4:])
        shutil.move(src, dst)

print('OK!')


