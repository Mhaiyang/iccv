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

args = {
    'exp_name': 'msd8_MHY1',
    'type': 0
}


ROOT_DIR = os.getcwd()
IMAGE_DIR = os.path.join(msd_testing_root, "image")
MASK_DIR = os.path.join(msd_testing_root, "mask")
# PREDICT_DIR = os.path.join(ROOT_DIR, ckpt_path, exp_name, '%s_%s' % (exp_name, args['snapshot']))
PREDICT_DIR = "/home/iccd/iccv/ckpt/TAYLOR5/TAYLOR5_160_crf/f1"
# PREDICT_DIR = "/home/iccd/iccv/ckpt/TAYLOR5_NOC/TAYLOR5_NOC_140/f1"
# PREDICT_DIR = "/home/iccd/iccv/msd9_results/msd9_R3Net-4-test512"
# PREDICT_DIR = "/home/iccd/R3Net/ckpt/R3Net/(R3Net) HKU-IS_6000"

if args['type'] != 0:
    type_path = os.path.join("/home/iccd/data/2019", str(args['type']))
    typelist = os.listdir(type_path)
    testlist = os.listdir(IMAGE_DIR)
    imglist = list(set(typelist) & set(testlist))
else:
    imglist = os.listdir(IMAGE_DIR)

print("Total {} test images".format(len(imglist)))

ACC = []
IOU = []
precision_record, recall_record = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
MAE = []
BER = []
NUM = []

for i, imgname in enumerate(imglist):

    print("###############  {}   ###############".format(i+1))
    ###########################################################################
    ################  Quantitative Evaluation for Single Image ################
    ###########################################################################

    # gt_mask = evaluation.get_mask(imgname, MASK_DIR)
    gt_mask = get_gt_mask(imgname, MASK_DIR)
    predict_mask_normalized = get_normalized_predict_mask(imgname, PREDICT_DIR)
    predict_mask_binary = get_binary_predict_mask(imgname, PREDICT_DIR)

    acc = accuracy_mirror(predict_mask_binary, gt_mask)
    iou = compute_iou(predict_mask_binary, gt_mask)
    precision, recall = cal_precision_recall((predict_mask_normalized * 255).astype(np.uint8),
                                             (gt_mask * 255).astype(np.uint8))
    for idx, data in enumerate(zip(precision, recall)):
        p, r = data
        precision_record[idx].update(p)
        recall_record[idx].update(r)
    mae = compute_mae(predict_mask_normalized, gt_mask)
    ber = compute_ber(predict_mask_binary, gt_mask)

    print("acc : {}".format(acc))
    print("iou : {}".format(iou))
    print("mae : {}".format(mae))
    print("ber : {}".format(ber))

    ACC.append(acc)
    IOU.append(iou)
    MAE.append(mae)
    BER.append(ber)

    num = imgname.split(".")[0]
    NUM.append(num)

mean_ACC = sum(ACC)/len(ACC)
mean_IOU = 100 * sum(IOU)/len(IOU)
F = cal_fmeasure([precord.avg for precord in precision_record], [rrecord.avg for rrecord in recall_record])
mean_MAE = sum(MAE)/len(MAE)
mean_BER = 100 * sum(BER)/len(BER)

print(len(ACC))
print(len(IOU))
print(len(MAE))
print(len(BER))

data_write(os.path.join('./excel', '%s.xlsx' % (args['exp_name'])), [NUM, ACC,
            [100*x for x in IOU], MAE, [100*x for x in BER]])

print("{}, \n{:20} {:.2f} \n{:20} {:.3f} \n{:20} {:.3f} \n{:20} {:.3f} \n{:20} {:.2f}\n".
      format(PREDICT_DIR, "mean_IOU", mean_IOU, "mean_ACC", mean_ACC, "F", F,
             "mean_MAE", mean_MAE, "mean_BER", mean_BER))




