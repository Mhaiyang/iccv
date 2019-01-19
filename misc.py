import numpy as np
import os
from PIL import Image
import skimage.io
import skimage.transform
import xlwt

import pydensecrf.densecrf as dcrf


class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def crf_refine(img, annos):
    assert img.dtype == np.uint8
    assert annos.dtype == np.uint8
    assert img.shape[:2] == annos.shape

    # img and annos should be np array with data type uint8

    EPSILON = 1e-8

    M = 2  # salient or not
    tau = 1.05
    # Setup the CRF model
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

    anno_norm = annos / 255.

    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * _sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))

    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')
    res = infer[1, :]

    res = res * 255
    res = res.reshape(img.shape[:2])
    return res.astype('uint8')


################################################################
######################## Evaluation ############################
################################################################
def data_write(file_path, datas):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(sheetname="sheet1", cell_overwrite_ok=True)

    j = 0
    for data in datas:
        for i in range(len(data)):
            sheet1.write(i, j, data[i])
        j = j + 1

    f.save(file_path)


def get_mask_directly(imgname, MASK_DIR):
    filestr = imgname.split(".")[0]
    mask_folder = MASK_DIR
    mask_path = mask_folder + "/" + filestr + ".png"
    mask = skimage.io.imread(mask_path)
    mask = np.where(mask == 255, 1, 0).astype(np.uint8)

    return mask


def get_predict_mask(imgname, PREDICT_MASK_DIR):
    """Get mask by specified single image name"""
    filestr = imgname.split(".")[0]
    mask_folder = PREDICT_MASK_DIR
    mask_path = mask_folder + "/" + filestr + "_b.png"
    if not os.path.exists(mask_path):
        print("{} has no label8.png")
    mask = skimage.io.imread(mask_path)
    mask = np.where(mask < 127, 1, 0).astype(np.uint8)

    return mask


def accuracy_mirror(predict_mask, gt_mask):
    """
    sum_i(n_ii) / sum_i(t_i)
    :param predict_mask:
    :param gt_mask:
    :return:
    """

    check_size(predict_mask, gt_mask)

    N_p = np.sum(gt_mask)
    N_n = np.sum(np.logical_not(gt_mask))
    if N_p + N_n != 640 * 512:
        raise Exception("Check if mask shape is correct!")

    TP = np.sum(np.logical_and(predict_mask, gt_mask))
    TN = np.sum(np.logical_and(np.logical_not(predict_mask), np.logical_not(gt_mask)))

    accuracy_ = TP/N_p

    return accuracy_


def compute_iou(predict_mask, gt_mask):
    """
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    Here, n_cl = 1 as we have only one class (mirror).
    :param predict_mask:
    :param gt_mask:
    :return:
    """

    check_size(predict_mask, gt_mask)

    if np.sum(predict_mask) == 0 or np.sum(gt_mask) == 0:
        IoU = 0

    n_ii = np.sum(np.logical_and(predict_mask, gt_mask))
    t_i = np.sum(gt_mask)
    n_ij = np.sum(predict_mask)

    iou_ = n_ii / (t_i + n_ij - n_ii)

    return iou_


def f_score(predict_mask, gt_mask):

    check_size(predict_mask, gt_mask)

    N_p = np.sum(gt_mask)
    N_n = np.sum(np.logical_not(gt_mask))
    if N_p + N_n != 640 * 512:
        raise Exception("Check if mask shape is correct!")

    TP = np.sum(np.logical_and(predict_mask, gt_mask))
    TN = np.sum(np.logical_and(np.logical_not(predict_mask), np.logical_not(gt_mask)))

    precision = TP/N_p
    if np.sum(predict_mask):
        recall = TP/np.sum(predict_mask)
    else:
        recall = 0
    alpha = 0.3

    if precision == 0 and recall == 0:
        f_score_ = 0
    else:
        f_score_ = (1.3*precision*recall)/(0.3*precision+recall)

    return f_score_

def compute_mae(predict_mask, gt_mask):

    check_size(predict_mask, gt_mask)

    N_p = np.sum(gt_mask)
    N_n = np.sum(np.logical_not(gt_mask))
    if N_p + N_n != 640 * 512:
        raise Exception("Check if mask shape is correct!")

    mae_ = np.mean(abs(predict_mask.astype(np.int) - gt_mask.astype(np.int)))

    return mae_


def compute_ber(predict_mask, gt_mask):
    """
    BER: balance error rate.
    :param predict_mask:
    :param gt_mask:
    :return:
    """
    check_size(predict_mask, gt_mask)

    N_p = np.sum(gt_mask)
    N_n = np.sum(np.logical_not(gt_mask))
    if N_p + N_n != 640 * 512:
        raise Exception("Check if mask shape is correct!")

    TP = np.sum(np.logical_and(predict_mask, gt_mask))
    TN = np.sum(np.logical_and(np.logical_not(predict_mask), np.logical_not(gt_mask)))

    ber_ = 1 - (1 / 2) * ((TP / N_p) + (TN / N_n))

    return ber_


def segm_size(segm):
    try:
        height = segm.shape[0]
        width  = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


'''
Exceptions
'''
class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
