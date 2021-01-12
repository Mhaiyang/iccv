"""
 @Time    : 202/17/19 16:20
 @Author  : TaylorMei
 @Email   : mhy845879017@gmail.com
 
 @Project : iccv
 @File    : 1_json_to_dataset.py
 @Function:
 
"""
import os

json_path = '/home/iccd/data/msd9_c/test/correction/cor_json'

json_list = os.listdir(json_path)

for i, json_name in enumerate(json_list):
    print(i, json_name)

    full_path = os.path.join(json_path, json_name)

    os.system('labelme_json_to_dataset %s' % (full_path))
