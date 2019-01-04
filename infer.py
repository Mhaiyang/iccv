import numpy as np
import os
import time

import torch
from torch import nn
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from config import msd_testing_root
from misc import check_mkdir, crf_refine
from model.edge import EDGE

torch.cuda.set_device(0)

ckpt_path = './ckpt'
exp_name = 'EDGE'
args = {
    'snapshot': '100',
    'scale': 416,
    'crf': True,
    'multi_gpu': False
}

img_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_test = {'MSD9': msd_testing_root}

to_pil = transforms.ToPILImage()


def main():
    net = EDGE().cuda()
    if args['multi_gpu']:
        net = nn.DataParallel(net)

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
                edge, res = net(img_var)
                edge = np.array(transforms.Resize((h, w))(to_pil(edge.data.squeeze(0).cpu())))
                prediction = np.array(transforms.Resize((h, w))(to_pil(res.data.squeeze(0).cpu())))
                if args['crf']:
                    prediction = crf_refine(np.array(img.convert('RGB')), prediction)

                Image.fromarray(edge).save(
                    os.path.join(ckpt_path, exp_name, '%s_%s' % (exp_name, args['snapshot']), img_name[:-4] + "_edge.png"))
                Image.fromarray(prediction).save(
                    os.path.join(ckpt_path, exp_name, '%s_%s' % (exp_name, args['snapshot']), img_name[:-4] + ".png"))
            end = time.time()
            print("Average Time Is : {:.2f}".format((end - start) / len(img_list)))


if __name__ == '__main__':
    main()
