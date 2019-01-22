"""
 @Time    : 201/22/19 19:22
 @Author  : TaylorMei
 @Email   : mhy845879017@gmail.com
 
 @Project : iccv
 @File    : train_base3_mse.py
 @Function:
 
"""
import datetime
import os

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm

import joint_transforms
from config import msd_training_root
from config import backbone_path
from dataset import ImageFolder
from misc import AvgMeter, check_mkdir
from model.base3 import BASE3

import loss as L

cudnn.benchmark = True

device_ids = [1]

ckpt_path = './ckpt'
exp_name = 'BASE3_MSE'

args = {
    'epoch_num': 100,
    'train_batch_size': 8,
    'last_epoch': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'scale': 512,
    'save_point': [40, 60, 80],
    'add_graph': True,
    'poly_train': True
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
    joint_transforms.RandomRotate(),
    joint_transforms.Resize((args['scale'], args['scale']))
])
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # maybe can optimized.
])
target_transform = transforms.ToTensor()

# Prepare Data Set.
train_set = ImageFolder(msd_training_root, joint_transform, img_transform, target_transform)
print("Train set: {}".format(train_set.__len__()))
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=64, shuffle=True)

mse = nn.MSELoss().cuda(device_ids[0])


def main():
    print(args)

    net = BASE3(backbone_path).cuda(device_ids[0]).train()
    if args['add_graph']:
        writer.add_graph(net, input_to_model=torch.rand(
            args['train_batch_size'], 3, args['scale'], args['scale']).cuda(device_ids[0]))

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])

    if len(args['snapshot']) > 0:
        print('Training Resumes From \'%s\'' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

    net = nn.DataParallel(net, device_ids=device_ids)

    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)
    writer.close()


def train(net, optimizer):
    curr_iter = 1

    for epoch in range(args['last_epoch'] + 1, args['last_epoch'] + 1 + args['epoch_num']):
        iou_4_record, iou_3_record, iou_2_record, iou_1_record, iou_f_record, \
        mse_4_record, mse_3_record, mse_2_record, mse_1_record, mse_f_record, loss_record \
            = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), \
              AvgMeter(), AvgMeter(), AvgMeter()

        train_iterator = tqdm(train_loader, total=len(train_loader))
        for data in train_iterator:
            if args['poly_train']:
                base_lr = args['lr'] * (1 - float(curr_iter) / (args['epoch_num'] * len(train_loader))) ** args[
                    'lr_decay']
                optimizer.param_groups[0]['lr'] = 2 * base_lr
                optimizer.param_groups[1]['lr'] = 1 * base_lr

            inputs, labels = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda(device_ids[0])
            labels = Variable(labels).cuda(device_ids[0])

            optimizer.zero_grad()

            predict_4, predict_3, predict_2, predict_1, predict_f = net(inputs)

            iou_4 = L.lovasz_hinge(predict_4, labels)
            iou_3 = L.lovasz_hinge(predict_3, labels)
            iou_2 = L.lovasz_hinge(predict_2, labels)
            iou_1 = L.lovasz_hinge(predict_1, labels)
            iou_f = L.lovasz_hinge(predict_f, labels)

            mse_4 = mse(torch.sigmoid(predict_4), labels)
            mse_3 = mse(torch.sigmoid(predict_3), labels)
            mse_2 = mse(torch.sigmoid(predict_2), labels)
            mse_1 = mse(torch.sigmoid(predict_1), labels)
            mse_f = mse(torch.sigmoid(predict_f), labels)

            loss = iou_1 + iou_2 + iou_3 + iou_4 + iou_f + mse_1 + mse_2 + mse_3 + mse_4 + mse_f

            loss.backward()

            optimizer.step()

            loss_record.update(loss.data, batch_size)
            iou_4_record.update(iou_4.data, batch_size)
            iou_3_record.update(iou_3.data, batch_size)
            iou_2_record.update(iou_2.data, batch_size)
            iou_1_record.update(iou_1.data, batch_size)
            iou_f_record.update(iou_f.data, batch_size)
            mse_4_record.update(mse_4.data, batch_size)
            mse_3_record.update(mse_3.data, batch_size)
            mse_2_record.update(mse_2.data, batch_size)
            mse_1_record.update(mse_1.data, batch_size)
            mse_f_record.update(mse_f.data, batch_size)

            if curr_iter % 50 == 0:
                writer.add_scalar('loss', loss, curr_iter)
                writer.add_scalar('iou_4', iou_4, curr_iter)
                writer.add_scalar('iou_3', iou_3, curr_iter)
                writer.add_scalar('iou_2', iou_2, curr_iter)
                writer.add_scalar('iou_1', iou_1, curr_iter)
                writer.add_scalar('iou_f', iou_f, curr_iter)
                writer.add_scalar('mse_4', mse_4, curr_iter)
                writer.add_scalar('mse_3', mse_3, curr_iter)
                writer.add_scalar('mse_2', mse_2, curr_iter)
                writer.add_scalar('mse_1', mse_1, curr_iter)
                writer.add_scalar('mse_f', mse_f, curr_iter)

            log = '[%3d],[%5d],[%.6f],[%.5f],[i4:%.5f],[i3:%.5f],[i2:%.5f],[i1:%.5f],[if:%.5f],' \
                  '[m4:%.5f],[m3:%.5f],[m2:%.5f],[m1:%.5f],[mf:%.5f]' % \
                  (epoch, curr_iter, base_lr, loss_record.avg, iou_4_record.avg, iou_3_record.avg, iou_2_record.avg,
                   iou_1_record.avg, iou_f_record.avg, mse_4_record.avg, mse_3_record.avg, mse_2_record.avg,
                   mse_1_record.avg, mse_f_record.avg)
            train_iterator.set_description(log)
            open(log_path, 'a').write(log + '\n')

            curr_iter += 1

        if epoch in args['save_point']:
            net.cpu()
            torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % epoch))
            net.cuda(device_ids[0])

        if epoch >= args['epoch_num']:
            net.cpu()
            torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % epoch))
            print("Optimization Have Done!")
            return


if __name__ == '__main__':
    main()
