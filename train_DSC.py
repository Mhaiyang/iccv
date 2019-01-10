"""
  @Time    : 2019-1-9 04:28
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : iccv
  @File    : train_DSC.py
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
from config import msd_training_root, msd_testing_root
from config import backbone_path
from dataset import ImageFolder
from misc import AvgMeter, check_mkdir
from model.dsc import DSC

cudnn.benchmark = True

# device_ids = [0]
device_ids = [2, 3, 4, 5, 7, 8]
# device_ids = [1, 0]

ckpt_path = './ckpt'
exp_name = 'DSC'

# batch size of 8 with resolution of 416*416 is exactly OK for the GTX 1080Ti GPU
args = {
    'epoch_num': 60,
    'train_batch_size': 24,
    'val_batch_size': 8,
    'last_epoch': 0,
    'lr': 5e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'scale': 416,
    'save_point': [40, 50],
    'add_graph': True,
    'poly_train': False
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
val_joint_transform = joint_transforms.Compose([
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
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=8, shuffle=True)
val_set = ImageFolder(msd_testing_root, val_joint_transform, img_transform, target_transform)
print("Validation Set: {}".format(val_set.__len__()))
val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=8, shuffle=False)

# Loss Functions.
bce_logit = nn.BCEWithLogitsLoss().cuda(device_ids[0])


def main():
    print(args)

    net = DSC().cuda(device_ids[0]).train()
    if args['add_graph']:
        writer.add_graph(net, input_to_model=torch.rand(
            args['train_batch_size'], 3, args['scale'], args['scale']).cuda(device_ids[0]))
    net = nn.DataParallel(net, device_ids=device_ids)

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias' and 'predict' not in name],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4] == 'bias' and 'predict' in name],
         'lr': 0.2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias' and 'predict' not in name],
         'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']},
        {'params': [param for name, param in net.named_parameters() if name[-4] != 'bias' and 'predict' in name],
         'lr': 0.1 * args['lr'], 'weight_decay': args['weight_decay']},
    ], momentum=args['momentum'])

    # for name, param in net.named_parameters():
    #     print(name)

    # for m in net.modules():
    #     print(m)

    if len(args['snapshot']) > 0:
        print('training resumes from \'%s\'' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)
    writer.close()


def train(net, optimizer):
    curr_iter = 1

    for epoch in range(args['last_epoch'] + 1, args['last_epoch'] + 1 + args['epoch_num']):
        loss_4_record, loss_3_record, loss_2_record, loss_1_record, loss_0_record, \
        loss_g_record, loss_f_record, loss_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), \
                                                    AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

        train_iterator = tqdm(train_loader, total=len(train_loader))
        for data in train_iterator:
            if args['poly_train']:
                optimizer.param_groups[0]['lr'] = 2 * args['lr'] * \
                                                  (1 - float(curr_iter) / (args['epoch_num'] * len(train_loader))) \
                                                  ** args['lr_decay']
                optimizer.param_groups[1]['lr'] = args['lr'] * \
                                                  (1 - float(curr_iter) / (args['epoch_num'] * len(train_loader))) \
                                                  ** args['lr_decay']

            inputs, labels = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda(device_ids[0])
            labels = Variable(labels).cuda(device_ids[0])

            optimizer.zero_grad()

            predict_4, predict_3, predict_2, predict_1, predict_0, predict_g, predict_f = net(inputs)

            loss_4 = bce_logit(predict_4, labels)
            loss_3 = bce_logit(predict_3, labels)
            loss_2 = bce_logit(predict_2, labels)
            loss_1 = bce_logit(predict_1, labels)
            loss_0 = bce_logit(predict_0, labels)
            loss_g = bce_logit(predict_g, labels)
            loss_f = bce_logit(predict_f, labels)

            loss = loss_4 + loss_3 + loss_2 + loss_1 + loss_0 + loss_g + loss_f

            loss.backward()

            optimizer.step()

            loss_record.update(loss.data, batch_size)
            loss_4_record.update(loss_4.data, batch_size)
            loss_3_record.update(loss_3.data, batch_size)
            loss_2_record.update(loss_2.data, batch_size)
            loss_1_record.update(loss_1.data, batch_size)
            loss_0_record.update(loss_0.data, batch_size)
            loss_g_record.update(loss_g.data, batch_size)
            loss_f_record.update(loss_f.data, batch_size)

            if curr_iter % 50 == 0:
                writer.add_scalar('loss', loss, curr_iter)
                writer.add_scalar('loss_4', loss_4, curr_iter)
                writer.add_scalar('loss_3', loss_3, curr_iter)
                writer.add_scalar('loss_2', loss_2, curr_iter)
                writer.add_scalar('loss_1', loss_1, curr_iter)
                writer.add_scalar('loss_0', loss_0, curr_iter)
                writer.add_scalar('loss_g', loss_g, curr_iter)
                writer.add_scalar('loss_f', loss_f, curr_iter)

            log = '[Epoch: %2d], [Iter: %5d], [Sum: %.5f], [L4: %.5f], [L3: %.5f], [L2: %.5f], [L1: %.5f] ' \
                  '[L0: %.5f], [Lg: %.5f], [Lf: %.5f]' % \
                  (epoch, curr_iter, loss_record.avg, loss_4_record.avg, loss_3_record.avg, loss_2_record.avg,
                   loss_1_record.avg, loss_0_record.avg, loss_g_record.avg, loss_f_record.avg)
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
