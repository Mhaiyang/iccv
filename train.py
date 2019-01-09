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

import joint_transforms
from config import msd_training_root, msd_testing_root
from config import backbone_path
from dataset import ImageFolder
from misc import AvgMeter, check_mkdir
from model.fuse import FUSE

cudnn.benchmark = True

device_ids = [0]
# device_ids = [2, 3, 4, 5]
# device_ids = [0, 1]

ckpt_path = './ckpt'
exp_name = 'DSC'

# batch size of 8 with resolution of 416*416 is exactly OK for the GTX 1080Ti GPU
args = {
    'iter_num': 12000,
    'train_batch_size': 8,
    'val_batch_size': 8,
    'last_iter': 0,
    'lr': 1e-8,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'scale': 416,
    'add_graph': False,
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
bce = nn.BCELoss().cuda(device_ids[0])
bce_logit = nn.BCEWithLogitsLoss().cuda(device_ids[0])


def main():
    print(args)

    net = DSC().cuda(device_ids[0]).train()
    if args['add_graph']:
        writer.add_graph(net, input_to_model=torch.rand(
            args['train_batch_size'], 3, args['scale'], args['scale']).cuda(device_ids[0]))
    net = nn.DataParallel(net, device_ids=device_ids)

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])

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
    curr_iter = args['last_iter']
    while True:
        train_loss_record, \
        loss_f3_record, loss_f2_record, loss_f1_record, loss_f0_record, \
        loss_b3_record, loss_b2_record, loss_b1_record, loss_b0_record, \
        loss_fb_record, loss_e_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), \
                         AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

        for i, data in enumerate(train_loader):
            if args['poly_train']:
                optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                    ) ** args['lr_decay']
                optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']

            inputs, labels, edges = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda(device_ids[0])
            labels = Variable(labels).cuda(device_ids[0])
            edges = Variable(edges).cuda(device_ids[0])

            optimizer.zero_grad()

            predict_f3, predict_f2, predict_f1, predict_f0, \
            predict_b3, predict_b2, predict_b1, predict_b0, predict_e, predict_fb = net(inputs)

            loss_f3 = bce_logit(predict_f3, labels)
            loss_f2 = bce_logit(predict_f2, labels)
            loss_f1 = bce_logit(predict_f1, labels)
            loss_f0 = bce_logit(predict_f0, labels)

            loss_b3 = bce(1 - torch.sigmoid(predict_b3), labels)
            loss_b2 = bce(1 - torch.sigmoid(predict_b2), labels)
            loss_b1 = bce(1 - torch.sigmoid(predict_b1), labels)
            loss_b0 = bce(1 - torch.sigmoid(predict_b0), labels)

            loss_e = bce_logit(predict_e, edges)

            loss_fb = bce_logit(predict_fb, labels)

            loss = loss_f3 + loss_f2 + loss_f1 + loss_f0 + \
                   loss_b3 + loss_b2 + loss_b1 + loss_b0 + 10*loss_e + 8*loss_fb

            loss.backward()

            optimizer.step()

            train_loss_record.update(loss.data, batch_size)
            loss_f3_record.update(loss_f3.data, batch_size)
            loss_f2_record.update(loss_f2.data, batch_size)
            loss_f1_record.update(loss_f1.data, batch_size)
            loss_f0_record.update(loss_f0.data, batch_size)
            loss_b3_record.update(loss_b3.data, batch_size)
            loss_b2_record.update(loss_b2.data, batch_size)
            loss_b1_record.update(loss_b1.data, batch_size)
            loss_b0_record.update(loss_b0.data, batch_size)
            loss_e_record.update(loss_e.data, batch_size)
            loss_fb_record.update(loss_fb.data, batch_size)

            if curr_iter % 10 == 0:
                writer.add_scalar('Total loss', loss, curr_iter)
                writer.add_scalar('f3 loss', loss_f3, curr_iter)
                writer.add_scalar('f2 loss', loss_f2, curr_iter)
                writer.add_scalar('f1 loss', loss_f1, curr_iter)
                writer.add_scalar('f0 loss', loss_f0, curr_iter)
                writer.add_scalar('b3 loss', loss_b3, curr_iter)
                writer.add_scalar('b2 loss', loss_b2, curr_iter)
                writer.add_scalar('b1 loss', loss_b1, curr_iter)
                writer.add_scalar('b0 loss', loss_b0, curr_iter)
                writer.add_scalar('e loss', loss_e, curr_iter)
                writer.add_scalar('fb loss', loss_fb, curr_iter)

            curr_iter += 1

            log = '[iter %d], [sum %.5f],  [f3 %.5f], [f2 %.5f], [f1 %.5f], [f0 %.5f] ' \
                  '[b3 %.5f], [b2 %.5f], [b1 %.5f], [b0 %.5f], [e %.5f], [fb %.5f], [lr %.13f]' % \
                  (curr_iter, train_loss_record.avg,
                   loss_f3_record.avg, loss_f2_record.avg, loss_f1_record.avg, loss_f0_record.avg,
                   loss_b3_record.avg, loss_b2_record.avg, loss_b1_record.avg, loss_b0_record.avg,
                   loss_e_record.avg, loss_fb_record.avg, optimizer.param_groups[1]['lr'])
            print(log)
            open(log_path, 'a').write(log + '\n')

            # if curr_iter % 5 == 0:
            #     net.eval()
            #     validate(net, curr_iter, 128)
            #     net.train()

            if curr_iter >= args['iter_num']:
                net.cpu()
                torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                # torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, '%d_optim.pth' % curr_iter))
                print("Optimization Have Done!")
                return


def validate(net, curr_iter, total_iter):
    loss_record = AvgMeter()
    for x in range(total_iter):
        for y, data in enumerate(val_loader):
            inputs, labels, _ = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda(device_ids[0])
            labels = Variable(labels).cuda(device_ids[0])

            _, predict = net(inputs)
            loss = bce(predict, labels)
            loss_record.update(loss.data, batch_size)
    print("Validation Loss: {}".format(loss_record.avg))
    writer.add_scalar('val loss', loss_record.avg, curr_iter)


if __name__ == '__main__':
    main()
