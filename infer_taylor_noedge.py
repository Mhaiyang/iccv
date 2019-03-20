"""
 @Time    : 201/28/19 09:07
 @Author  : TaylorMei
 @Email   : mhy845879017@gmail.com
 
 @Project : iccv
 @File    : infer_mhy.py
 @Function:
 
"""
import numpy as np
import os
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import skimage.io

from config import msd_testing_root
from misc import check_mkdir, crf_refine
from model.taylor5 import TAYLOR5

device_ids = [0]
torch.cuda.set_device(device_ids[0])

ckpt_path = './ckpt'
exp_name = 'TAYLOR5_MSRA_BCE'
args = {
    'snapshot': '100',
    'scale': 384,
    'crf': True
}

img_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    # transforms.Resize((256, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# detectron
to_test = {'msd9': msd_testing_root}

to_pil = transforms.ToPILImage()


def main():
    net = TAYLOR5().cuda(device_ids[0])

    if len(args['snapshot']) > 0:
        print('Load snapshot {} for testing'.format(args['snapshot']))
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        print('Load {} succeed!'.format(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))

    net.eval()
    with torch.no_grad():
        for name, root in to_test.items():
            # img_list = [img_name for img_name in os.listdir(os.path.join(root, 'image')) if img_name.endswith('.jpg')]
            img_list = [img_name for img_name in os.listdir(os.path.join(root, 'image'))]
            start = time.time()
            for idx, img_name in enumerate(img_list):
                print('predicting for {}: {:>4d} / {}'.format(name, idx + 1, len(img_list)))
                check_mkdir(os.path.join(ckpt_path, exp_name, '%s_%s' % (exp_name, args['snapshot'])))
                img = Image.open(os.path.join(root, 'image', img_name))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    print("{} is a gray image.".format(name))
                w, h = img.size
                img_var = Variable(img_transform(img).unsqueeze(0)).cuda()
                # f_4, f_3, f_2, f_1, e = net(img_var)
                f_4, f_3, f_2, f_1 = net(img_var)
                # output = f.data.squeeze(0).cpu()
                # edge = e.data.squeeze(0).cpu()
                f_4 = f_4.data.squeeze(0).cpu()
                f_3 = f_3.data.squeeze(0).cpu()
                f_2 = f_2.data.squeeze(0).cpu()
                f_1 = f_1.data.squeeze(0).cpu()
                # prediction = np.array(transforms.Resize((h, w))(to_pil(output)))
                # edge = np.array(transforms.Resize((h, w))(to_pil(edge)))
                f_4 = np.array(transforms.Resize((h, w))(to_pil(f_4)))
                f_3 = np.array(transforms.Resize((h, w))(to_pil(f_3)))
                f_2 = np.array(transforms.Resize((h, w))(to_pil(f_2)))
                f_1 = np.array(transforms.Resize((h, w))(to_pil(f_1)))
                if args['crf']:
                    # prediction = crf_refine(np.array(img.convert('RGB')), prediction)
                    f_1 = crf_refine(np.array(img.convert('RGB')), f_1)
                    # f_2 = crf_refine(np.array(img.convert('RGB')), f_2)
                    # f_3 = crf_refine(np.array(img.convert('RGB')), f_3)
                    # f_4 = crf_refine(np.array(img.convert('RGB')), f_4)
                    # edge = crf_refine(np.array(img.convert('RGB')), edge)

                # Image.fromarray(prediction).save(os.path.join(ckpt_path, exp_name, '%s_%s' % (exp_name, args['snapshot']), 'map', img_name[:-4] + ".png"))
                # Image.fromarray(edge).save(os.path.join(ckpt_path, exp_name, '%s_%s' % (exp_name, args['snapshot']), 'edge', img_name[:-4] + ".png"))
                # Image.fromarray(f_4).save(os.path.join(ckpt_path, exp_name, '%s_%s' % (exp_name, args['snapshot']), 'f4', img_name[:-4] + ".png"))
                # Image.fromarray(f_3).save(os.path.join(ckpt_path, exp_name, '%s_%s' % (exp_name, args['snapshot']), 'f3', img_name[:-4] + ".png"))
                # Image.fromarray(f_2).save(os.path.join(ckpt_path, exp_name, '%s_%s' % (exp_name, args['snapshot']), 'f2', img_name[:-4] + ".png"))
                Image.fromarray(f_1).save(os.path.join(ckpt_path, exp_name, '%s_%s' % (exp_name, args['snapshot']), 'f1', img_name[:-4] + ".png"))
                # skimage.io.imsave(os.path.join(ckpt_path, exp_name, '%s_%s' % (exp_name, args['snapshot']), 'f1', img_name[:-4] + ".png"), np.where(f_1>=127.5, 255, 0).astype(np.uint8))
                # skimage.io.imsave(os.path.join(ckpt_path, exp_name, '%s_%s' % (exp_name, args['snapshot']), img_name[:-4] + ".png"), prediction.astype(np.uint8))
                # check_mkdir(os.path.join(msd_testing_root, 'mei_' + str(args['scale'])))
                # Image.fromarray(f_1).save(os.path.join(msd_testing_root, 'mei_' + str(args['scale']), img_name[:-4] + ".png"))
            end = time.time()
            print("Average Time Is : {:.2f}".format((end - start) / len(img_list)))


if __name__ == '__main__':
    main()
