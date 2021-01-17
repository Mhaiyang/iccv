"""
 @Time    : 2021/1/16 22:08
 @Author  : TaylorMei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : iccv
 @File    : train_resnet50.py
 @Function:
 
"""
import datetime
import time
import os
import sys

sys.path.append("..")

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm

import joint_transforms
from config import sod_training_root
from config import backbone_path
from dataset import ImageFolder
from misc import AvgMeter, check_mkdir
from mirrornet_plus_resnet50 import MirrorNet_Plus_ResNet50

import loss

cudnn.benchmark = True

torch.manual_seed(2021)
device_ids = [1]

# ckpt_path = '/media/iccd/disk2/tip_mirror_ckpt'
ckpt_path = './ckpt'
exp_name = 'MirrorNet_Plus_sod_resnet50_3'

args = {
    'epoch_num': 120,
    'train_batch_size': 24,
    'last_epoch': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'scale': 384,
    'save_point': [100],
    'poly_train': False,
    'poly_warmup': False,
    'warmup_epoch': 0,
    'cosine_warmup': False,
    'f3_sche': True,
    'optimizer': 'SGD',
    'w1': [1, 1, 1],
    'w2': [1, 1, 1, 1],
}

print(torch.__version__)

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
    joint_transforms.RandomRotate(),

])
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()

# Prepare Data Set.
train_set = ImageFolder(sod_training_root, joint_transform, img_transform, target_transform)
print("Train set: {}".format(train_set.__len__()))
train_loader = DataLoader(train_set, collate_fn=train_set.collate, batch_size=args['train_batch_size'], num_workers=8,
                          shuffle=True, drop_last=True)

total_epoch = args['epoch_num'] * len(train_loader)

# hybrid loss
bce_loss = nn.BCEWithLogitsLoss(size_average=True).cuda(device_ids[0])
ssim_loss = loss.SSIM(window_size=11, size_average=True).cuda(device_ids[0])
iou_loss = loss.IOU(size_average=True).cuda(device_ids[0])
edge_loss = loss.EdgeLoss().cuda(device_ids[0])


def bce_ssim_iou_loss(pred, target):
    bce_out = bce_loss(pred, target)
    ssim_out = 1 - ssim_loss(pred, target)
    iou_out = iou_loss(pred, target)

    loss = args['w1'][0] * bce_out + args['w1'][1] * ssim_out + args['w1'][2] * iou_out

    return loss


def bce_iou_edge_loss(pred, target):
    bce_out = bce_loss(pred, target)
    iou_out = iou_loss(pred, target)
    edge_out = edge_loss(pred, target)

    loss = args['w1'][0] * bce_out + args['w1'][1] * iou_out + args['w1'][2] * edge_out

    return loss


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def main():
    print(args)
    print(exp_name)

    net = MirrorNet_Plus_ResNet50(backbone_path).cuda(device_ids[0]).train()

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
    global total_epoch
    curr_iter = 1
    start_time = time.time()

    for epoch in range(args['last_epoch'] + 1, args['last_epoch'] + 1 + args['epoch_num']):
        loss_record, loss_4_record, loss_3_record, loss_2_record, loss_1_record, = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

        train_iterator = tqdm(train_loader, total=len(train_loader))
        for data in train_iterator:
            if args['poly_train']:
                base_lr = args['lr'] * (1 - float(curr_iter) / float(total_epoch)) ** args['lr_decay']
                optimizer.param_groups[0]['lr'] = 2 * base_lr
                optimizer.param_groups[1]['lr'] = 1 * base_lr

            if args['poly_warmup']:
                if curr_iter < args['warmup_epoch']:
                    base_lr = 1 / args['warmup_epoch'] * (1 + curr_iter)
                else:
                    curr_iter = curr_iter - args['warmup_epoch'] + 1
                    total_epoch = total_epoch - args['warmup_epoch'] + 1
                    base_lr = args['lr'] * (1 - float(curr_iter) / float(total_epoch)) ** args['lr_decay']
                optimizer.param_groups[0]['lr'] = 2 * base_lr
                optimizer.param_groups[1]['lr'] = 1 * base_lr

            if args['cosine_warmup']:
                if curr_iter < args['warmup_epoch']:
                    base_lr = 1 / args['warmup_epoch'] * (1 + curr_iter)
                else:
                    curr_iter = curr_iter - args['warmup_epoch'] + 1
                    total_epoch = total_epoch - args['warmup_epoch'] + 1
                    base_lr = args['lr'] * (1 + np.cos(np.pi * float(curr_iter) / float(total_epoch))) / 2
                optimizer.param_groups[0]['lr'] = 2 * base_lr
                optimizer.param_groups[1]['lr'] = 1 * base_lr

            if args["f3_sche"]:
                base_lr = args['lr'] * (1 - abs((curr_iter + 1) / (total_epoch + 1) * 2 - 1))
                optimizer.param_groups[0]['lr'] = 2 * base_lr
                optimizer.param_groups[1]['lr'] = 1 * base_lr

            inputs, labels = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda(device_ids[0])
            labels = Variable(labels).cuda(device_ids[0])

            optimizer.zero_grad()

            predict_4, predict_3, predict_2, predict_1 = net(inputs)

            loss_4 = bce_iou_edge_loss(predict_4, labels)
            loss_3 = bce_iou_edge_loss(predict_3, labels)
            loss_2 = bce_iou_edge_loss(predict_2, labels)
            loss_1 = bce_iou_edge_loss(predict_1, labels)

            loss = args['w2'][0] * loss_4 + args['w2'][1] * loss_3 + args['w2'][2] * loss_2 + args['w2'][3] * loss_1

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

            log = '[%3d], [%6d], [%.6f], [%.5f], [%.5f], [%.5f], [%.5f], [%.5f]' % \
                  (epoch, curr_iter, base_lr, loss_record.avg, loss_4_record.avg, loss_3_record.avg,
                   loss_2_record.avg, loss_1_record.avg)
            train_iterator.set_description(log)
            open(log_path, 'a').write(log + '\n')

            curr_iter += 1

        if epoch in args['save_point']:
            net.cpu()
            torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % epoch))
            net.cuda(device_ids[0])

        if epoch >= args['epoch_num']:
            net.cpu()
            torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % epoch))
            print("Total Training Time: {}".format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
            print("Optimization Have Done!")
            return


if __name__ == '__main__':
    main()
