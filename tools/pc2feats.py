'''
----------------------------------------------------------
  @File:     pc2feats.py
  @Brief:    convert point cloud to feats
  @Author:   Leijie.Zhang
  @Created:  16:51/9/19/2019
  @Modified: 16:51/9/19/2019
----------------------------------------------------------
'''

import numpy as np
import argparse
import math

import mmcv
from mmdet.datasets.pc_utils import *

def load_pc(file_path, front_only=True):
    pc = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    if front_only:
        pc = pc[pc[:,0] >= 0, :]

    return pc

def main():
    root_dir = '/space/data/kitti/training/'
    ids = mmcv.list_from_file(root_dir+'../ImageSets/trainval.txt')

    feats_dir = root_dir+'front_feats_10x496x432/'
    mmcv.mkdir_or_exist(feats_dir)

    for i, id in enumerate(ids):
        file_path = root_dir + 'velodyne/' + id + '.bin'

        pc = load_pc(file_path)
        feats = feats_encoder(pc)
        feats.tofile(feats_dir + '{}.bin'.format(id))
        print(i)



if __name__ == '__main__':
    main()
