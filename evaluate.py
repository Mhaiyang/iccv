"""
  @Time    : 2019-1-2 01:38
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com

  @Project : iccv
  @File    : evaluate.py
  @Function:

"""
import os
import numpy as np
import skimage.io
from misc import *
from config import msd_testing_root

ckpt_path = 'ckpt'

exp_name = 'EDGE_CBAM_X_CCL'
args = {
    'snapshot': '100',
    'type': 0
}


ROOT_DIR = os.getcwd()
IMAGE_DIR = os.path.join(msd_testing_root, "image")
MASK_DIR = os.path.join(msd_testing_root, "mask")
PREDICT_DIR = os.path.join(ROOT_DIR, ckpt_path, exp_name, '%s_%s' % (exp_name, args['snapshot']))
# PREDICT_DIR = os.path.join(ROOT_DIR, "other_methods", "PSPNet_20")

if args['type'] != 0:
    type_path = os.path.join("/home/iccd/data/types", str(args['type']))
    typelist = os.listdir(type_path)
    testlist = os.listdir(IMAGE_DIR)
    imglist = list(set(typelist) & set(testlist))
else:
    imglist = os.listdir(IMAGE_DIR)

print("Total {} test images".format(len(imglist)))

ACC = []
IOU = []
F = []
MAE = []
BER = []
NUM = []

for i, imgname in enumerate(imglist):

    print("###############  {}   ###############".format(i+1))
    ###########################################################################
    ################  Quantitative Evaluation for Single Image ################
    ###########################################################################

    # gt_mask = evaluation.get_mask(imgname, MASK_DIR)
    gt_mask = get_mask_directly(imgname, MASK_DIR)
    predict_mask = get_predict_mask(imgname, PREDICT_DIR)

    acc = accuracy_mirror(predict_mask, gt_mask)
    iou = compute_iou(predict_mask, gt_mask)
    f = f_score(predict_mask, gt_mask)
    mae = compute_mae(predict_mask, gt_mask)
    ber = compute_ber(predict_mask, gt_mask)

    print("acc : {}".format(acc))
    print("iou : {}".format(iou))
    print("f : {}".format(f))
    print("mae : {}".format(mae))
    print("ber : {}".format(ber))

    ACC.append(acc)
    IOU.append(iou)
    F.append(f)
    MAE.append(mae)
    BER.append(ber)

    num = imgname.split("_")[0]
    NUM.append(int(num))

mean_ACC = 100 * sum(ACC)/len(ACC)
mean_IOU = 100 * sum(IOU)/len(IOU)
mean_F = sum(F)/len(F)
mean_MAE = sum(MAE)/len(MAE)
mean_BER = 100 * sum(BER)/len(BER)

print(len(ACC))
print(len(IOU))
print(len(F))
print(len(MAE))
print(len(BER))

data_write(os.path.join('./excel', '%s_%s.xlsx' % (exp_name, args['snapshot'])), [NUM, [100*x for x in ACC],
            [100*x for x in IOU], F, MAE, [100*x for x in BER]])

print("{}, \n{:20} {:.2f} \n{:20} {:.2f} \n{:20} {:.3f} \n{:20} {:.3f} \n{:20} {:.2f}\n".
      format(PREDICT_DIR, "mean_ACC", mean_ACC, "mean_IOU", mean_IOU, "mean_F", mean_F,
             "mean_MAE", mean_MAE, "mean_BER", mean_BER))




