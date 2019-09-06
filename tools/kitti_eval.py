'''
----------------------------------------------------------
  @File:     kitti_eval.py
  @Brief:    dump results to kitti format and do kitti eval with dev tools
  @Author:   Leijie.Zhang
  @Created:  11:23/9/6/2019
  @Modified: 11:23/9/6/2019
----------------------------------------------------------
'''

import os
from argparse import ArgumentParser

import mmcv
import numpy as np

#  Kitti Format
#  default     Values    Name      Description
#       ----------------------------------------------------------------------------
#  invalid       1    type         Describes the type of object: 'Car', 'Van', 'Truck',
#                                  'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
#                                  'Misc' or 'DontCare'
#     0          1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
#                                  truncated refers to the object leaving image boundaries
#     0          1    occluded     Integer (0,1,2,3) indicating occlusion state:
#                                  0 = fully visible, 1 = partly occluded
#                                  2 = largely occluded, 3 = unknown
#    -10         1    alpha        Observation angle of object, ranging [-pi..pi]
#    -1          4    bbox         2D bounding box of object in the image (0-based index):
#                                  contains left, top, right, bottom pixel coordinates
#   -1000        3    dimensions   3D object dimensions: height, width, length (in meters)
#   -1000        3    location     3D object location x,y,z in camera coordinates (in meters)
#   -1000        1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
#   -1000        1    score        Only for results: Float, indicating confidence in
#                                  detection, needed for p/r curves, higher is better.

CLASSES = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}

def kitti_eval(result_file, cvt_result, cfg):
    det_results = mmcv.load(result_file)
    filenames = mmcv.list_from_file(cfg.data.test.ann_file)
    assert len(filenames) == len(det_results), "test files must be equal"
    for filename, dets in zip(filenames, det_results):
        with open(cvt_result+'/data/{}.txt'.format(filename), 'w') as f:
            # per class
            for cls, det_cls in enumerate(dets):
                if det_cls.any():
                    assert det_cls.shape[1] == 5, 'det_cls shape should be 5'
                    for det in det_cls:
                        line = [CLASSES[cls], 0, 0, -10]
                        line.extend(det[:-1])
                        line.extend([-1000, -1000, -1000, -1000, -1000, -1000, -1000, det[-1]])
                        f.writelines([str(i)+' ' for i in line])
                        f.write('\n')

    gt_dir = cfg.data.test.img_prefix + '/../label_2'
    print('gt_dir: ', gt_dir)
    print('result_dir: ', cvt_result)
    eval_fun = './tools/kitti_devkit_object/kitti_eval/build/eval_3d ' + gt_dir + ' ' + cvt_result
    os.system(eval_fun)

def main():
    parser = ArgumentParser(description='Kitti Evaluation')
    parser.add_argument('result', help='pkl result file path')
    parser.add_argument('config', help='config file path')
    parser.add_argument('cvt_result', help='convert pkl file to txt of kitti format')
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    kitti_eval(args.result, args.cvt_result, cfg)

if __name__ == '__main__':
    main()
