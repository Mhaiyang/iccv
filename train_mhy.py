"""
 @Time    : 201/27/19 16:25
 @Author  : TaylorMei
 @Email   : mhy845879017@gmail.com
 
 @Project : iccv
 @File    : train_mhy.py
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
from model.taylor1 import TAYLOR1

import loss as L

cudnn.benchmark = True

device_ids = [1]

ckpt_path = './ckpt'
exp_name = 'TAYLOR1'

# mirror
args = {
    'epoch_num': 140,
    'train_batch_size': 10,
    'last_epoch': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'scale': 384,
    'save_point': [100, 120, 140],
    'add_graph': True,
    'poly_train': True,
    'optimizer': 'SGD'
}

# shadow
# args = {
#     'epoch_num': 140,
#     'train_batch_size': 10,
#     'last_epoch': 0,
#     'lr': 5e-3,
#     'lr_decay': 0.9,
#     'weight_decay': 5e-4,
#     'momentum': 0.9,
#     'snapshot': '',
#     'scale': 384,
#     'save_point': [80, 100, 120, 140],
#     'add_graph': True,
#     'poly_train': True,
#     'optimizer': 'SGD'
# }

# saliency
# args = {
#     'epoch_num': 100,
#     'train_batch_size': 1,
#     'last_epoch': 0,
#     'lr': 1e-3,
#     'lr_decay': 0.9,
#     'weight_decay': 5e-4,
#     'momentum': 0.9,
#     'snapshot': '',
#     'scale': 384,
#     'save_point': [60, 80, 100],
#     'add_graph': True,
#     'poly_train': True,
#     'optimizer': 'SGD'
# }

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
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # maybe can be optimized.
])
target_transform = transforms.ToTensor()

# Prepare Data Set.
train_set = ImageFolder(msd_training_root, joint_transform, img_transform, target_transform)
print("Train set: {}".format(train_set.__len__()))
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=32, shuffle=True)

total_epoch = args['epoch_num'] * len(train_loader)

bce_logit = nn.BCEWithLogitsLoss().cuda(device_ids[0])

def main():
    print(args)
    print(exp_name)

    net = TAYLOR1(backbone_path).cuda(device_ids[0]).train()
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

    net = nn.DataParallel(net, device_ids=device_ids)
    print("Using {} GPU(s) to Train.".format(len(device_ids)))

    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)
    writer.close()


def train(net, optimizer):
    curr_iter = 1

    for epoch in range(args['last_epoch'] + 1, args['last_epoch'] + 1 + args['epoch_num']):
        loss_4_record, loss_3_record, loss_2_record, loss_1_record, \
        loss_f_record, loss_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

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

            predict_4, predict_3, predict_2, predict_1, predict_f = net(inputs)

            loss_4 = L.lovasz_hinge(predict_4, labels)
            loss_3 = L.lovasz_hinge(predict_3, labels)
            loss_2 = L.lovasz_hinge(predict_2, labels)
            loss_1 = L.lovasz_hinge(predict_1, labels)
            loss_f = L.lovasz_hinge(predict_f, labels)

            # loss_4 = bce_logit(predict_4, labels)
            # loss_3 = bce_logit(predict_3, labels)
            # loss_2 = bce_logit(predict_2, labels)
            # loss_1 = bce_logit(predict_1, labels)
            # loss_f = bce_logit(predict_f, labels)

            loss = loss_4 + loss_3 + loss_2 + loss_1 + loss_f

            loss.backward()

            optimizer.step()

            loss_record.update(loss.data, batch_size)
            loss_4_record.update(loss_4.data, batch_size)
            loss_3_record.update(loss_3.data, batch_size)
            loss_2_record.update(loss_2.data, batch_size)
            loss_1_record.update(loss_1.data, batch_size)
            loss_f_record.update(loss_f.data, batch_size)

            if curr_iter % 50 == 0:
                writer.add_scalar('loss', loss, curr_iter)
                writer.add_scalar('loss_4', loss_4, curr_iter)
                writer.add_scalar('loss_3', loss_3, curr_iter)
                writer.add_scalar('loss_2', loss_2, curr_iter)
                writer.add_scalar('loss_1', loss_1, curr_iter)
                writer.add_scalar('loss_f', loss_f, curr_iter)

            log = '[%3d], [%6d], [%.6f], [%.5f], [L4: %.5f], [L3: %.5f], [L2: %.5f], [L1: %.5f], [Lf: %.5f]' % \
                  (epoch, curr_iter, base_lr, loss_record.avg, loss_4_record.avg, loss_3_record.avg, loss_2_record.avg,
                   loss_1_record.avg, loss_f_record.avg)
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
