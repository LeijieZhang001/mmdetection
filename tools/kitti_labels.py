'''
----------------------------------------------------------
  @File:     kitti_label.py
  @Brief:    trans kitti labels
  @Author:   Leijie.Zhang
  @Created:  18:20/9/17/2019
  @Modified: 18:20/9/17/2019
----------------------------------------------------------
'''

import numpy as np

import mmcv

def main():

    root_dir = '/space/data/kitti/training/label_2'
    new_root_dir = '/space/data/kitti/training/label_2_front'
    mmcv.mkdir_or_exist(new_root_dir)

    ann_file = '/space/data/kitti/ImageSets/trainval.txt'

    pc_ids = mmcv.list_from_file(ann_file)
    for pc_id in pc_ids:
        with open(root_dir+'/{}.txt'.format(pc_id), 'r') as f:
            lines = f.readlines()

        with open(new_root_dir+'/{}.txt'.format(pc_id), 'w') as f:
            for ix, line in enumerate(lines):
                obj = line.strip().split(' ')
                if float(obj[13]) >= 0:
                    f.writelines(line)

if __name__ == '__main__':
    main()
