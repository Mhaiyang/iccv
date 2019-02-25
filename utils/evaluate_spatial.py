"""
 @Time    : 202/19/19 15:47
 @Author  : TaylorMei
 @Email   : mhy845879017@gmail.com
 
 @Project : iccv
 @File    : evaluate_spatial.py
 @Function:
 
"""
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
import skimage.transform
from misc import *
from config import msd_testing_root

ckpt_path = 'ckpt'

exp_name = 'MHY1_12_1e-3'
args = {
    'snapshot': '80',
    'type': 0
}


ROOT_DIR = os.getcwd()
IMAGE_DIR = os.path.join(msd_testing_root, "image")
MASK_DIR = os.path.join(msd_testing_root, "mask")
# PREDICT_DIR = os.path.join(ROOT_DIR, ckpt_path, exp_name, '%s_%s' % (exp_name, args['snapshot']))
PREDICT_DIR = "/home/iccd/iccv/utils/msd7_train_normalized.png"

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
# F = []
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
    height = gt_mask.shape[0]
    width = gt_mask.shape[1]
    predict_mask = skimage.io.imread(PREDICT_DIR)
    predict_mask = skimage.transform.resize(predict_mask, [height, width], 0)
    predict_mask = predict_mask.astype(np.float32)
    predict_mask_binary = np.where(predict_mask >= 0.5, 1, 0).astype(np.float32)

    acc = accuracy_image(predict_mask_binary, gt_mask)
    iou = compute_iou(predict_mask_binary, gt_mask)
    # f = f_score(predict_mask, gt_mask)
    mae = compute_mae(predict_mask, gt_mask)
    ber = compute_ber(predict_mask_binary, gt_mask)

    print("acc : {}".format(acc))
    print("iou : {}".format(iou))
    # print("f : {}".format(f))
    print("mae : {}".format(mae))
    print("ber : {}".format(ber))

    ACC.append(acc)
    IOU.append(iou)
    # F.append(f)
    MAE.append(mae)
    BER.append(ber)

    num = imgname.split("_")[0]
    NUM.append(int(num))

mean_ACC = sum(ACC)/len(ACC)
mean_IOU = 100 * sum(IOU)/len(IOU)
# mean_F = sum(F)/len(F)
mean_MAE = sum(MAE)/len(MAE)
mean_BER = 100 * sum(BER)/len(BER)

print(len(ACC))
print(len(IOU))
# print(len(F))
print(len(MAE))
print(len(BER))


print("{}, \n{:20} {:.3f} \n{:20} {:.2f} \n{:20} {:.3f} \n{:20} {:.2f}\n".
      format(PREDICT_DIR, "mean_ACC", mean_ACC, "mean_IOU", mean_IOU,
             "mean_MAE", mean_MAE, "mean_BER", mean_BER))




