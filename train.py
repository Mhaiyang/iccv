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
from model.edge_cbam_x_ccl_wl import EDGE_CBAM_X_CCL

cudnn.benchmark = True

# device_ids = [6, 7]
# device_ids = [2, 3, 4, 5]
device_ids = [1, 0]

ckpt_path = './ckpt'
exp_name = 'EDGE_CBAM_X_CCL'

# batch size of 8 with resolution of 416*416 is exactly OK for the GTX 1080Ti GPU
args = {
    'epoch_num': 60,
    'train_batch_size': 8,
    'val_batch_size': 8,
    'last_epoch': 0,
    'lr': 1e-4,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'scale': 512,
    'save_point': [40, 50],
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


class WL(nn.Module):
    def __init__(self):
        super(WL, self).__init__()

    def forward(self, pred, truth):
        # n c h w
        N_p = torch.tensor(torch.sum(torch.sum(truth, -1), -1), dtype=torch.float).unsqueeze(-1).unsqueeze(-1).expand_as(truth)
        N = torch.tensor(torch.numel(truth[0, :, :, :]), dtype=torch.float).unsqueeze(-1).unsqueeze(-1).expand_as(N_p)
        N_n = N - N_p

        pred_p = torch.where(pred.cpu() >= 0.5, torch.tensor(1.), torch.tensor(2.))
        TP_mask = torch.where(pred_p == truth.cpu(), torch.tensor(1.), torch.tensor(0.))
        TP = torch.tensor(torch.sum(torch.sum(TP_mask, -1), -1), dtype=torch.float).unsqueeze(-1).unsqueeze(-1).expand_as(truth)

        pred_n = torch.where(pred.cpu() < 0.5, torch.tensor(1.), torch.tensor(2.))
        TN_mask = torch.where(pred_n == (1 - truth.cpu()), torch.tensor(1.), torch.tensor(0.))
        TN = torch.tensor(torch.sum(torch.sum(TN_mask, -1), -1), dtype=torch.float).unsqueeze(-1).unsqueeze(-1).expand_as(truth)

        L1 = -(N_n / N) * (truth.cpu() * torch.log(pred.cpu())) - (N_p / N) * ((1 - truth.cpu()) * torch.log(1 - pred.cpu()))
        L2 = -(1 - TP / N_p) * truth.cpu() * torch.log(pred.cpu()) - (1 - TN / N_n) * (1 - truth.cpu()) * torch.log(1 - pred.cpu())

        return L1.mean() + L2.mean()


class EL(nn.Module):
    def __init__(self):
        super(EL, self).__init__()

    def forward(self, pred, truth):

        L = -10 * truth.cpu() * torch.log(pred.cpu()) - (1 - truth.cpu()) * torch.log(1 - pred.cpu())

        return L.mean()


wl = WL().cuda(device_ids[0])
el = EL().cuda(device_ids[0])


def main():
    print(args)

    net = EDGE_CBAM_X_CCL().cuda(device_ids[0]).train()
    if args['add_graph']:
        writer.add_graph(net, input_to_model=torch.rand(
            args['train_batch_size'], 3, args['scale'], args['scale']).cuda(device_ids[0]))
    net = nn.DataParallel(net, device_ids=device_ids)

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
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
    curr_iter = 1

    for epoch in range(args['last_epoch'] + 1, args['last_epoch'] + 1 + args['epoch_num']):
        loss_f4_record, loss_f3_record, loss_f2_record, loss_f1_record, \
        loss_b4_record, loss_b3_record, loss_b2_record, loss_b1_record, \
        loss_e_record, loss_fb_record, loss_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), \
                                                     AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), \
                                                     AvgMeter(), AvgMeter(), AvgMeter()

        train_iterator = tqdm(train_loader, total=len(train_loader))
        for data in train_iterator:
            if args['poly_train']:
                base_lr = args['lr'] * (1 - float(curr_iter) / (args['epoch_num'] * len(train_loader))) ** args['lr_decay']
                optimizer.param_groups[0]['lr'] = 2 * base_lr
                optimizer.param_groups[1]['lr'] = 1 * base_lr

            inputs, labels, edges = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda(device_ids[0])
            labels = Variable(labels).cuda(device_ids[0])
            edges = Variable(edges).cuda(device_ids[0])

            optimizer.zero_grad()

            predict_f4, predict_f3, predict_f2, predict_f1, \
            predict_b4, predict_b3, predict_b2, predict_b1, predict_e, predict_fb = net(inputs)

            loss_f4 = wl(predict_f4, labels)
            loss_f3 = wl(predict_f3, labels)
            loss_f2 = wl(predict_f2, labels)
            loss_f1 = wl(predict_f1, labels)

            # loss_b4 = wl(1 - torch.sigmoid(predict_b4), labels)
            # loss_b3 = wl(1 - torch.sigmoid(predict_b3), labels)
            # loss_b2 = wl(1 - torch.sigmoid(predict_b2), labels)
            # loss_b1 = wl(1 - torch.sigmoid(predict_b1), labels)

            loss_b4 = wl(1 - predict_b4, labels)
            loss_b3 = wl(1 - predict_b3, labels)
            loss_b2 = wl(1 - predict_b2, labels)
            loss_b1 = wl(1 - predict_b1, labels)

            loss_e = el(predict_e, edges)

            loss_fb = wl(predict_fb, labels)

            loss = loss_f4 + loss_f3 + loss_f2 + loss_f1 + \
                   loss_b4 + loss_b3 + loss_b2 + loss_b1 + loss_e + 8 * loss_fb

            loss.backward()

            optimizer.step()

            loss_record.update(loss.data, batch_size)
            loss_f4_record.update(loss_f4.data, batch_size)
            loss_f3_record.update(loss_f3.data, batch_size)
            loss_f2_record.update(loss_f2.data, batch_size)
            loss_f1_record.update(loss_f1.data, batch_size)
            loss_b4_record.update(loss_b4.data, batch_size)
            loss_b3_record.update(loss_b3.data, batch_size)
            loss_b2_record.update(loss_b2.data, batch_size)
            loss_b1_record.update(loss_b1.data, batch_size)
            loss_e_record.update(loss_e.data, batch_size)
            loss_fb_record.update(loss_fb.data, batch_size)

            if curr_iter % 50 == 0:
                writer.add_scalar('Total loss', loss, curr_iter)
                writer.add_scalar('f4 loss', loss_f4, curr_iter)
                writer.add_scalar('f3 loss', loss_f3, curr_iter)
                writer.add_scalar('f2 loss', loss_f2, curr_iter)
                writer.add_scalar('f1 loss', loss_f1, curr_iter)
                writer.add_scalar('b4 loss', loss_b4, curr_iter)
                writer.add_scalar('b3 loss', loss_b3, curr_iter)
                writer.add_scalar('b2 loss', loss_b2, curr_iter)
                writer.add_scalar('b1 loss', loss_b1, curr_iter)
                writer.add_scalar('e loss', loss_e, curr_iter)
                writer.add_scalar('fb loss', loss_fb, curr_iter)

            log = '[%3d], [f4 %.5f], [f3 %.5f], [f2 %.5f], [f1 %.5f] ' \
                  '[b4 %.5f], [b3 %.5f], [b2 %.5f], [b1 %.5f], [e %.5f], [fb %.5f], [lr %.6f]' % \
                  (epoch,
                   loss_f4_record.avg, loss_f3_record.avg, loss_f2_record.avg, loss_f1_record.avg,
                   loss_b4_record.avg, loss_b3_record.avg, loss_b2_record.avg, loss_b1_record.avg,
                   loss_e_record.avg, loss_fb_record.avg, base_lr)
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
