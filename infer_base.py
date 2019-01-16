"""
 @Time    : 201/10/19 09:34
 @Author  : TaylorMei
 @Email   : mhy845879017@gmail.com
 
 @Project : iccv
 @File    : infer_DSC.py
 @Function:
 
"""
import numpy as np
import os
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from config import msd_testing_root
from misc import check_mkdir, crf_refine
from model.base3 import BASE3

device_ids = [0]
torch.cuda.set_device(device_ids[0])

ckpt_path = './ckpt'
exp_name = 'BASE3_ML'
args = {
    'snapshot': '100',
    'scale': 512,
    'crf': True
}

img_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_test = {'MSD9': msd_testing_root}

to_pil = transforms.ToPILImage()


def main():
    net = BASE3().cuda(device_ids[0])

    if len(args['snapshot']) > 0:
        print('Load snapshot {} for testing'.format(args['snapshot']))
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        print('Load {} succeed!'.format(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))

    net.eval()
    with torch.no_grad():
        for name, root in to_test.items():
            img_list = [img_name for img_name in os.listdir(os.path.join(root, 'image')) if img_name.endswith('.jpg')]
            start = time.time()
            for idx, img_name in enumerate(img_list):
                print('predicting for {}: {:>4d} / {}'.format(name, idx + 1, len(img_list)))
                check_mkdir(os.path.join(ckpt_path, exp_name, '%s_%s' % (exp_name, args['snapshot'])))
                img = Image.open(os.path.join(root, 'image', img_name))
                w, h = img.size
                img_var = Variable(img_transform(img).unsqueeze(0)).cuda()
                _, _, _, _, f = net(img_var)
                output = f.data.squeeze(0).cpu()
                prediction = np.array(transforms.Resize((h, w))(to_pil(output)))
                if args['crf']:
                    prediction = crf_refine(np.array(img.convert('RGB')), prediction)

                Image.fromarray(prediction).save(
                    os.path.join(ckpt_path, exp_name, '%s_%s' % (exp_name, args['snapshot']), img_name[:-4] + ".png"))
            end = time.time()
            print("Average Time Is : {:.2f}".format((end - start) / len(img_list)))


if __name__ == '__main__':
    main()
