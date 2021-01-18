"""
 @Time    : 1/13/21 20:04
 @Author  : TaylorMei
 @Email   : mhy666@mail.dlut.edu.cn

 @Project : iccv
 @File    : infer.py
 @Function:

"""
import time
import datetime
import sys

sys.path.append("..")

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from collections import OrderedDict

from config import *
from misc import *
from mirrornet_plus_resnet50 import MirrorNet_Plus_ResNet50

# from model.base_dce_wof import Base_dce_wof

torch.manual_seed(2020)
device_ids = [1]
torch.cuda.set_device(device_ids[0])

results_path = './results'
# results_path = '/home/iccd/sod/results_intermediate_ca'
# results_path = '/home/iccd/sod/results_intermediate_sa'
check_mkdir(results_path)
# ckpt_path = '/media/iccd/disk2/tip_mirror_ckpt'
ckpt_path = './ckpt'
# exp_name = 'MirrorNet_NAC_SL_resnet50'
# exp_name = 'MirrorNet_NAC_resnet50_bie_four_ms_poly_v12'
exp_name = 'MirrorNet_Plus_sod_resnet50_4'
args = {
    'snapshot': '120',
    'scale': 384,
    'crf': False,
    'save_results': True,  # whether to save the resulting masks
    'if_eval': False
}

print(torch.__version__)

img_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_pil = transforms.ToPILImage()

to_test = OrderedDict([
    ('SOD', sod_path),
    ('ECSSD', ecssd_path),
    ('DUT-OMRON', dutomron_path),
    ('PASCAL-S', pascals_path),
    ('HKU-IS', hkuis_path),
    # ('HKU-IS-TEST', hkuis_test_path),
    ('DUTS-TE', dutste_path),
    # ('MSD', msd9_test_path)
])

results = OrderedDict()


def main():
    net = MirrorNet_Plus_ResNet50(backbone_path).cuda(device_ids[0])

    if len(args['snapshot']) > 0:
        print('Load snapshot {} for testing'.format(args['snapshot']))
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        print('Load {} succeed!'.format(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))

    net.eval()
    with torch.no_grad():
        start = time.time()
        for name, root in to_test.items():

            start_each = time.time()
            image_path = os.path.join(root, 'image')
            mask_path = os.path.join(root, 'mask')

            precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
            mae_record = AvgMeter()

            if args['save_results']:
                check_mkdir(os.path.join(results_path, exp_name, args['snapshot'], '%s' % (name)))

            img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('g')]
            for idx, img_name in enumerate(img_list):
                # print('predicting for {}: {:>4d} / {}'.format(name, idx + 1, len(img_list)))

                if name == 'HKU-IS':
                    img = Image.open(os.path.join(image_path, img_name + '.png')).convert('RGB')
                else:
                    img = Image.open(os.path.join(image_path, img_name + '.jpg')).convert('RGB')

                w, h = img.size
                img_var = Variable(img_transform(img).unsqueeze(0)).cuda(device_ids[0])
                # prediction = net(img_var)
                # _, _, prediction = net(img_var)
                _, _, _, prediction = net(img_var)
                # _, _, _, _, prediction = net(img_var)

                prediction = np.array(transforms.Resize((h, w))(to_pil(prediction.data.squeeze(0).cpu())))
                # c = prediction.shape[1]
                # prediction = np.array(transforms.Resize((int(c/4), c))(to_pil(prediction.data.transpose(1, 3).squeeze(0).cpu())))

                if args['crf']:
                    prediction = crf_refine(np.array(img.convert('RGB')), prediction)

                if args['save_results']:
                    Image.fromarray(prediction).convert('RGB').save(
                        os.path.join(results_path, exp_name, args['snapshot'],
                                     '%s' % (name), img_name + '.png'))

                if args['if_eval']:
                    gt = np.array(Image.open(os.path.join(mask_path, img_name + '.png')).convert('L'))
                    precision, recall, mae = cal_precision_recall_mae(prediction, gt)
                    for pidx, pdata in enumerate(zip(precision, recall)):
                        p, r = pdata
                        precision_record[pidx].update(p)
                        recall_record[pidx].update(r)
                    mae_record.update(mae)

            if args['if_eval']:
                fmeasure = cal_fmeasure([precord.avg for precord in precision_record],
                                        [rrecord.avg for rrecord in recall_record])
                results[name] = OrderedDict([('F', "%.3f" % fmeasure), ('MAE', "%.3f" % mae_record.avg)])
            print("{}'s average Time Is : {:.2f}".format(name, (time.time() - start_each) / len(img_list)))
    if args['if_eval']:
        print('test results:')
        print(results)

    end = time.time()
    print("Total Testing Time: {}".format(str(datetime.timedelta(seconds=int(end - start)))))


if __name__ == '__main__':
    main()

