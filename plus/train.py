"""
 @Time    : 1/12/21 18:37
 @Author  : TaylorMei
 @Email   : mhy666@mail.dlut.edu.cn
 
 @Project : iccv
 @File    : train.py
 @Function:
 
"""
import datetime
import os
import time
import sys
sys.path.append("..")

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm
from PIL import Image

import joint_transforms
from config import msd_training_root, msd_testing_root
from config import backbone_path
from dataset import ImageFolder
from misc import AvgMeter, check_mkdir, crf_refine, compute_ber
# from mirrornet_plus import MirrorNet_Plus
from mirrornet_plus_gb import MirrorNet_Plus_GB
# from mirrornet_plus_rb import MirrorNet_Plus_RB

import numpy as np
import loss as L

cudnn.benchmark = True

device_ids = [0]

ckpt_path = './ckpt'
# ckpt_path = '/media/iccd/disk1/tip_mirror_ckpt'
exp_name = 'MirrorNet_Plus_GB_7'
# exp_name = 'MirrorNet_Plus_RB_3'

args = {
    'epoch_num': 200,
    'epoch_thres': 150,
    'train_batch_size': 10,
    'last_epoch': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'scale': 384,
    'save_point': [],
    'add_graph': False,
    'poly_train': True,
    'optimizer': 'SGD',
    'crf': True
}

# Path.
check_mkdir(ckpt_path)
check_mkdir(os.path.join(ckpt_path, exp_name))
vis_path = os.path.join(ckpt_path, exp_name, 'log')
check_mkdir(vis_path)
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')
writer = SummaryWriter(log_dir=vis_path, comment=exp_name)

# Transform Data.
joint_transform = joint_transforms.Compose([
    joint_transforms.Resize((args['scale'], args['scale'])),
    joint_transforms.RandomRotate()
])
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # maybe can be optimized.
])
target_transform = transforms.ToTensor()

test_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Prepare Data Set.
train_set = ImageFolder(msd_training_root, joint_transform, img_transform, target_transform)
print("Train set: {}".format(train_set.__len__()))
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=8, shuffle=True)

total_epoch = args['epoch_num'] * len(train_loader)

to_test = {'MSD': msd_testing_root}
to_pil = transforms.ToPILImage()
IMAGE_DIR = os.path.join(msd_testing_root, "image")
MASK_DIR = os.path.join(msd_testing_root, "mask")

best_ber = 100


def main():
    global best_ber
    print(args)
    print(exp_name)

    # net = MirrorNet_Plus(backbone_path).cuda(device_ids[0]).train()
    net = MirrorNet_Plus_GB(backbone_path).cuda(device_ids[0]).train()
    # net = MirrorNet_Plus_RB(backbone_path).cuda(device_ids[0]).train()
    if args['add_graph']:
        writer.add_graph(net, input_to_model=torch.rand(
            args['train_batch_size'], 3, args['scale'], args['scale']).cuda(device_ids[0]))

    if args['optimizer'] == 'Adam':
        print("Adam")
        optimizer = optim.Adam([
            {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
             'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
        ])
    else:
        print("SGD")
        optimizer = optim.SGD([
            {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
             'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
        ], momentum=args['momentum'])

    if len(args['snapshot']) > 0:
        print('Training Resumes From \'%s\'' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        total_epoch = (args['epoch_num'] - int(args['snapshot'])) * len(train_loader)
        print(total_epoch)

    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)
    writer.close()


def train(net, optimizer):
    global best_ber
    curr_iter = 1
    start_time = time.time()

    for epoch in range(args['last_epoch'] + 1, args['last_epoch'] + 1 + args['epoch_num']):
        loss_4_record, loss_3_record, loss_2_record, loss_1_record, \
        loss_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

        train_iterator = tqdm(train_loader, total=len(train_loader))
        for data in train_iterator:
            if args['poly_train']:
                base_lr = args['lr'] * (1 - float(curr_iter) / float(total_epoch)) ** args['lr_decay']
                optimizer.param_groups[0]['lr'] = 2 * base_lr
                optimizer.param_groups[1]['lr'] = 1 * base_lr

            inputs, labels = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda(device_ids[0])
            labels = Variable(labels).cuda(device_ids[0])

            optimizer.zero_grad()

            predict_4, predict_3, predict_2, predict_1 = net(inputs)

            loss_4 = L.lovasz_hinge(predict_4, labels)
            loss_3 = L.lovasz_hinge(predict_3, labels)
            loss_2 = L.lovasz_hinge(predict_2, labels)
            loss_1 = L.lovasz_hinge(predict_1, labels)

            loss = loss_4 + loss_3 + loss_2 + loss_1

            loss.backward()

            optimizer.step()

            loss_record.update(loss.data, batch_size)
            loss_4_record.update(loss_4.data, batch_size)
            loss_3_record.update(loss_3.data, batch_size)
            loss_2_record.update(loss_2.data, batch_size)
            loss_1_record.update(loss_1.data, batch_size)

            if curr_iter % 50 == 0:
                writer.add_scalar('loss', loss, curr_iter)
                writer.add_scalar('loss_4', loss_4, curr_iter)
                writer.add_scalar('loss_3', loss_3, curr_iter)
                writer.add_scalar('loss_2', loss_2, curr_iter)
                writer.add_scalar('loss_1', loss_1, curr_iter)

            log = '[%3d], [%6d], [%.6f], [%.5f], [L4: %.5f], [L3: %.5f], [L2: %.5f], [L1: %.5f]' % \
                  (epoch, curr_iter, base_lr, loss_record.avg, loss_4_record.avg, loss_3_record.avg, loss_2_record.avg,
                   loss_1_record.avg)
            train_iterator.set_description(log)
            open(log_path, 'a').write(log + '\n')

            curr_iter += 1

        if epoch in args['save_point']:
            net.cpu()
            torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % epoch))
            net.cuda(device_ids[0])

        if epoch >= args['epoch_thres'] and epoch % 5 == 0:
            ber = test(net)
            print("mean ber of %d epoch is %.5f" % (epoch, ber))
            if ber < best_ber:
                net.cpu()
                torch.save(net.state_dict(),
                           os.path.join(ckpt_path, exp_name, 'epoch_%d_ber_%.2f.pth' % (epoch, ber)))
                print("The optimized epoch is %04d" % epoch)
            net = net.cuda(device_ids[0]).train()

        if epoch >= args['epoch_num']:
            net.cpu()
            torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % epoch))
            print("Total Training Time: {}".format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
            print(exp_name)
            print("Optimization Have Done!")
            return


def test(net):
    print("Testing Mode ......")
    global best_ber

    net.eval()
    BER = []

    for name, root in to_test.items():
        img_list = [img_name for img_name in os.listdir(os.path.join(root, 'image'))]

        for idx, img_name in enumerate(img_list):
            # print('predicting for {}: {:>4d} / {}'.format(name, idx + 1, len(img_list)))
            # check_mkdir(os.path.join(ckpt_path, exp_name, '%s_%s_%s' % (exp_name, args['snapshot'], 'nocrf')))
            img = Image.open(os.path.join(root, 'image', img_name))
            gt = Image.open(os.path.join(root, 'mask', img_name[:-4] + '.png'))
            gt = np.array(gt)
            if img.mode != 'RGB':
                img = img.convert('RGB')
                print("{} is a gray image.".format(name))
            w, h = img.size
            img_var = Variable(test_transform(img).unsqueeze(0)).cuda(device_ids[0])
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

            gt = np.where(gt == 255, 1, 0).astype(np.float32)
            f_1 = np.where(f_1 * 255.0 >= 127.5, 1, 0).astype(np.float32)

            ber = compute_ber(f_1, gt)
            # print("The %d pics ber is %.5f" % (idx + 1, ber))
            BER.append(ber)
        mean_BER = 100 * sum(BER) / len(BER)
    return mean_BER


if __name__ == '__main__':
    main()
