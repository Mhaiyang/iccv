import time
import datetime
import os
import sys
sys.path.append("../..")
from PIL import Image
from collections import OrderedDict
from misc import *
from config import ecssd_path, hkuis_path, hkuis_test_path, pascals_path, sod_path, dutste_path, dutomron_path

method_list = [
    # 'C2SNet',
    # 'RAS',
    # 'PAGRN',
    # 'DGRL',
    # 'R3Net',
    'BMPM',
    # 'PiCANet-R',
    # 'DSS',
    # 'BASNet',
    # 'CPD',
    # 'PAGE-Net',
    # 'AFNet',
    # 'BANet',
    # 'GCPANet',
    # 'F3Net',
    # 'MINet-R',
    # 'ITSD',
    # 'GDNet-B-S'
]

for order, method in enumerate(method_list):

    results_path = os.path.join('/media/iccd/disk1/18', method)

    to_test = OrderedDict([
                           ('SOD', sod_path),
                           ('PASCAL-S', pascals_path),
                           ('DUT-OMRON', dutomron_path),
                           ('ECSSD', ecssd_path),
                           ('HKU-IS', hkuis_path),
                           # ('HKU-IS-TEST', hkuis_test_path),
                           ('DUTS-TE', dutste_path),
                           ])

    print(results_path)
    # for key in to_test:
    #     print("{:12} {}".format(key, to_test[key]))

    results = OrderedDict()

    start = time.time()
    for name, root in to_test.items():
        prediction_path = os.path.join(results_path, name)
        gt_path = os.path.join(root, 'mask')

        precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]

        img_list = [os.path.splitext(f)[0] for f in os.listdir(gt_path) if f.endswith('.png')]
        # img_list = [os.path.splitext(f)[0] for f in os.listdir(prediction_path) if f.endswith('.png')]
        for idx, img_name in enumerate(img_list):
            # print('evaluating for %s: %d / %d      %s' % (name, idx + 1, len(img_list), img_name + '.png'))

            prediction = np.array(Image.open(os.path.join(prediction_path, img_name + '.jpg')).convert('L'))
            gt = np.array(Image.open(os.path.join(gt_path, img_name + '.png')).convert('L'))

            precision, recall, _ = cal_precision_recall_mae(prediction, gt)
            for idx, data in enumerate(zip(precision, recall)):
                p, r = data
                precision_record[idx].update(p)
                recall_record[idx].update(r)

        fmeasure = cal_fmeasure([precord.avg for precord in precision_record],
                                [rrecord.avg for rrecord in recall_record])

        results[name] = {('F', "%.3f" % fmeasure)}

        t_256 = [t / 255.0 for t in range(256)]
        r_256 = [rrecord.avg for rrecord in recall_record]
        p_256 = [precord.avg for precord in precision_record]
        f_256 = [(1 + 0.3) * p_each * r_each / (0.3 * p_each + r_each) for p_each, r_each in zip(p_256, r_256)]

        data = [[t, r, p, f] for t, r, p, f in zip(t_256, r_256, p_256, f_256)]

        save_root = os.path.join('/home/iccd/cvpr/sod/plot/data_18', name)
        check_mkdir(save_root)
        save_path = os.path.join(save_root, method + '_trpf.txt')
        file = open(save_path, 'a')
        file.seek(0)
        file.truncate()
        for i in range(len(data)):
            s = str(data[i]).replace('[', '').replace(']', '')
            s = s.replace("'", '').replace(',', '    ') + '\n'
            file.write(s)
        file.close()

    end = time.time()
    print("Time: {}".format(str(datetime.timedelta(seconds=int(end - start)))))
    for key in results:
        print("{:12} {}".format(key, results[key]))

