"""
 @Time    : 1/12/21 18:41
 @Author  : TaylorMei
 @Email   : mhy666@mail.dlut.edu.cn
 
 @Project : iccv
 @File    : infer.py
 @Function:
 
"""
import numpy as np
import os
import time
import sys
sys.path.append("..")

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from config import msd_testing_root
from config import more_testing_root
from misc import check_mkdir, crf_refine
# from mirrornet_plus import MirrorNet_Plus
# from mirrornet_plus_gb import MirrorNet_Plus_GB
from mirrornet_plus_rb import MirrorNet_Plus_RB

device_ids = [1]
torch.cuda.set_device(device_ids[0])

ckpt_path = './ckpt'
# ckpt_path = './'
# ckpt_path = '/media/iccd/disk1/tip_mirror_ckpt'
# exp_name = 'MirrorNet_Plus_3'
# exp_name = 'results'
# exp_name = 'MirrorNet_Plus_GB_1'
exp_name = 'MirrorNet_Plus_RB_2'
# pth_name = 'epoch_190_ber_6.03693.pth'
# pth_name = 'MirrorNet+.pth'
pth_name = 'epoch_150_ber_5.86.pth'
args = {
    'snapshot': '150',
    'scale': 384,
    'crf': True
}

img_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_test = {'MSD': msd_testing_root}
# to_test = {'more': more_testing_root}

to_pil = transforms.ToPILImage()


def main():
    net = MirrorNet_Plus_RB().cuda(device_ids[0])

    if len(args['snapshot']) > 0:
        print('Load snapshot {} for testing'.format(args['snapshot']))
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, pth_name)))
        print('Load {} succeed!'.format(os.path.join(ckpt_path, exp_name, pth_name)))

    net.eval()
    with torch.no_grad():
        for name, root in to_test.items():
            img_list = [img_name for img_name in os.listdir(os.path.join(root, 'image'))]
            start = time.time()
            for idx, img_name in enumerate(img_list):
                print('predicting for {}: {:>4d} / {}'.format(name, idx + 1, len(img_list)))
                # check_mkdir(os.path.join(ckpt_path, exp_name, '%s_%s' % (exp_name, pth_name[:-4]) + "_crf"))
                check_mkdir(os.path.join(ckpt_path, exp_name, pth_name[:-4]))
                img = Image.open(os.path.join(root, 'image', img_name))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    print("{} is a gray image.".format(name))
                w, h = img.size
                img_var = Variable(img_transform(img).unsqueeze(0)).cuda(device_ids[0])
                f_4, f_3, f_2, f_1 = net(img_var)
                f_4 = f_4.data.squeeze(0).cpu()
                f_3 = f_3.data.squeeze(0).cpu()
                f_2 = f_2.data.squeeze(0).cpu()
                f_1 = f_1.data.squeeze(0).cpu()
                f_4 = np.array(transforms.Resize((h, w))(to_pil(f_4)))
                f_3 = np.array(transforms.Resize((h, w))(to_pil(f_3)))
                f_2 = np.array(transforms.Resize((h, w))(to_pil(f_2)))
                f_1 = np.array(transforms.Resize((h, w))(to_pil(f_1)))
                if args['crf']:
                    f_1 = crf_refine(np.array(img.convert('RGB')), f_1)

                Image.fromarray(f_1).save(
                    # os.path.join(ckpt_path, exp_name, '%s_%s' % (exp_name, pth_name[:-4]) + "_crf",
                    #              img_name[:-4] + ".png"))
                    os.path.join(ckpt_path, exp_name, pth_name[:-4], img_name[:-4] + ".png"))

            end = time.time()
            print("Average Time Is : {:.2f}".format((end - start) / len(img_list)))


if __name__ == '__main__':
    main()
