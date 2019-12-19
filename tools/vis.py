'''
----------------------------------------------------------
  @File:     vis.py
  @Brief:    point cloud & bbox visulization
  @Author:   Leijie.Zhang
  @Created:  17:20/9/17/2019
  @Modified: 17:20/9/17/2019
----------------------------------------------------------
'''

import numpy as np
import argparse
import math
import os.path as osp

import cv2

import mmcv
from mmdet.core.dddet import bbox3d_cam2lidar, Calib, bbox3d_to_corners3d_lidar
from mmdet.models.detectors.pc_utils import *
from mmdet.datasets.pc_utils import *

CLASSES = ('car')

def get_kitti_object_level(obj):
    height = float(obj[7]) - float(obj[5]) + 1
    trucation = float(obj[1])
    occlusion = float(obj[2])
    if height >= 40 and trucation <= 0.15 and occlusion <= 0:
        return 1
    elif height >= 25 and trucation <= 0.3 and occlusion <= 1:
        return 2
    elif height >= 25 and trucation <= 0.5 and occlusion <= 2:
        return 3
    else:
        return 4

def load_kitti_anns(label_path, calib_path, gt_truth=True):
    calib = Calib(calib_path)
    with open(label_path, 'r') as f:
        lines = f.readlines()

    im_bboxes = []
    bboxes_3d = []
    scores = []
    for ix, line in enumerate(lines):
        obj = line.strip().split(' ')

        # 0-based coordinates
        cls = obj[0].lower().strip()
        if cls not in CLASSES:
            continue
        im_bbox = [float(obj[4]), float(obj[5]), float(obj[6]), float(obj[7])]
        ## ry, l, h, w, tx, ty, tz
        #bbox_3d = [float(obj[14]), float(obj[10]), float(obj[8]), float(obj[9]),
        #           float(obj[11]), float(obj[12]), float(obj[13])]
        # 7: tx, ty, tz, w, l, h, ry
        bbox_3d = [float(obj[11]), float(obj[12]), float(obj[13]), float(obj[9]),
                    float(obj[10]), float(obj[8]), float(obj[14])]
        if len(obj) == 16:
            score = [float(obj[15])]
        else:
            score = None

        # ignore objects with undetermined difficult level
        if gt_truth:
            level = get_kitti_object_level(obj)
        else:
            level = 0

        if level > 3:
            continue
        else:
            im_bboxes.append(im_bbox)
            bboxes_3d.append(bbox_3d)
            scores.append(score)

    if not im_bboxes:
        im_bboxes = np.zeros((0, 4))
        bboxes_3d = np.zeros((0, 7))
        scores = np.zeros((0,1))
    else:
        im_bboxes = np.array(im_bboxes, ndmin=2)
        bboxes_3d = np.array(bboxes_3d, ndmin=2)
        scores = np.array(scores, ndmin=2)
        # 7: tx, ty, tz, w, l, h, ry
        # ==>
        # 7: tx, ty, tz, w, l, h, rz
        bboxes_3d = bbox3d_cam2lidar(bboxes_3d.reshape((-1, 7)), calib.Tr_cam_to_velo)

    return {'bboxes_3d' : bboxes_3d.astype(np.float32),
            'im_bboxes' : im_bboxes.astype(np.float32),
            'scores' : scores.astype(np.float32)}

def load_pc(file_path, front_only=True):
    pc = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    if front_only:
        pc = pc[pc[:,0] >= 0, :]

    return pc

def feats_encoder(pc, pc_range=np.array([0, -39.68, -3, 69.12, 39.68, 1]), resolution=np.array([0.16, 0.16, 0.4])):
    """
    input:  N x 4 array
    output: c x h x w
    """
    whc = (pc_range[3:] - pc_range[:3]) / resolution
    whc = [int(i) for i in whc]
    width, height, channel = whc
    feats = np.zeros((channel, height, width), dtype=np.float32)
    for point in pc:
        if point[0] >= pc_range[0] and point[0] < pc_range[3] and point[1] >= pc_range[1] and point[1] < pc_range[4] and point[2] >= pc_range[2] and point[2] < pc_range[5]:
            w, h, c = (point[:3] - pc_range[:3]) / resolution
            feats[math.floor(c), math.floor(h), math.floor(w)] = 1

    return feats

def get_intensity(pc, pc_range=np.array([0, -39.68, -3, 69.12, 39.68, 1]), resolution=np.array([0.16, 0.16, 0.4])):
    """
    input:  N x 4 array
    output: c x h x w
    """
    whc = (pc_range[3:] - pc_range[:3]) / resolution
    whc = [int(i) for i in whc]
    width, height, channel = whc
    feats = np.zeros((channel, height, width), dtype=np.float32)
    max_height = -100 * np.ones((height, width), dtype=np.float32)
    intensity = np.zeros((height, width), dtype=np.float32)
    for point in pc:
        if point[0] >= pc_range[0] and point[0] < pc_range[3] and point[1] >= pc_range[1] and point[1] < pc_range[4] and point[2] >= pc_range[2] and point[2] < pc_range[5]:
            w, h, c = (point[:3] - pc_range[:3]) / resolution
            feats[math.floor(c), math.floor(h), math.floor(w)] = 1
            if point[2] >= max_height[math.floor(h), math.floor(w)]:
                intensity[math.floor(h), math.floor(w)] = point[3]
                max_height[math.floor(h), math.floor(w)] = point[2]

    return intensity, max_height

def intensity2map(intensity):
    show_map = np.zeros((intensity.shape[0], intensity.shape[1], 3), dtype=np.uint8)
    show_map[..., 0] = intensity * 0
    show_map[..., 1] = intensity * 255
    show_map[..., 2] = intensity * 120

    return show_map

def height2map(height):
    show_map = np.zeros((height.shape[0], height.shape[1], 3), dtype=np.uint8)
    max_height = height.max()
    min_height = height.min()
    height = (height-min_height)/(max_height-min_height)

    show_map[..., 0] = height * 0
    show_map[..., 1] = height * 255
    show_map[..., 2] = height * 120

    return show_map

def plot_corners3d(show_map, corners_3d, scalar, pc_range=np.array([0, -39.68, -3, 69.12, 39.68, 1]), resolution=np.array([0.16, 0.16, 0.4])):
    '''
    Input:
        corners_3d: Nx24 array, (x0 ... x7 y0 ... y7 z0 ... z7) in lidar coordinate
    '''
    for corner_3d in corners_3d:
        points = []
        for i in range(4):
            point = np.array([corner_3d[i], corner_3d[i+8], corner_3d[i+16]])
            # w, h => x, y
            w, h, c = (point - pc_range[:3]) / resolution
            points.append([int(w), int(h)])
        cv2.line(show_map, tuple(points[0]), tuple(points[1]), scalar, 1)
        cv2.line(show_map, tuple(points[1]), tuple(points[2]), scalar, 1)
        cv2.line(show_map, tuple(points[2]), tuple(points[3]), scalar, 1)
        cv2.line(show_map, tuple(points[3]), tuple(points[0]), scalar, 1)
    return show_map

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    # parser.add_argument('config', help='test config file path')
    # parser.add_argument('checkpoint', help='checkpoint file')
    # parser.add_argument('res_dir', help='output result file')
    # parser.add_argument('--lidar2cam', action='store_true', help='convert results from lidar coords to camera coords')
    # parser.add_argument(
    #     '--eval',
    #     type=str,
    #     nargs='+',
    #     choices=['bbox3d'],
    #     help='eval types')
    # parser.add_argument('--show', action='store_true', help='show results')
    # parser.add_argument(
    #     '--launcher',
    #     choices=['none', 'pytorch', 'slurm', 'mpi'],
    #     default='none',
    #     help='job launcher')
    args = parser.parse_args()
    return args

def gen_anchor():
    anchor_generator = AnchorGeneratorStride(sizes=[1.6, 3.9, 1.56],
                                             anchor_strides=[0.32, 0.32, 0.0],
                                             anchor_offsets=[0.16, -39.52, -1.78],
                                             rotations=[0, 1.57],
                                             match_threshold=0.6,
                                             unmatch_threshold=0.45,
                                             class_name='Car')
    anchor_map = [5, 248, 216]
    anchors = anchor_generator.generate(anchor_map)
    anchors = anchors[0,...]
    print('anchors shape: ', anchors.shape)
    return anchors

def plot_bbox3d(show_map, boxes_3d_in, scalar=(0,0,255)):
    boxes_3d = boxes_3d_in.copy()
    boxes_3d_bk = boxes_3d.copy()
    boxes_3d[:,4:] = boxes_3d_bk[:,:3]
    boxes_3d[:,3] = boxes_3d_bk[:,5]
    boxes_3d[:,2] = boxes_3d_bk[:,3]
    boxes_3d[:,1] = boxes_3d_bk[:,4]
    boxes_3d[:,0] = boxes_3d_bk[:,6]
    # corners_3d: Nx24 array, (x0 ... x7 y0 ... y7 z0 ... z7) in lidar coordinate
    corners_3d = bbox3d_to_corners3d_lidar(boxes_3d)
    bv = plot_corners3d(show_map, corners_3d, scalar)
    return bv

def main():
    args = parse_args()

    root_dir = '/space/data/kitti/training/'
    ids = mmcv.list_from_file(root_dir+'../ImageSets/val.txt')

    for id in ids:
        label_path = root_dir + 'label_2_front/' + id + '.txt'
        calib_path = root_dir + 'calib/' + id + '.txt'
        file_path = root_dir + 'velodyne/' + id + '.bin'

        pc = load_pc(file_path)
        #feats = feats_encoder(pc)
        #debug_feats(feats)
        intensity, max_height = get_intensity(pc)
        show_map = intensity2map(intensity)
        #show_map = height2map(max_height)

        anns = load_kitti_anns(label_path, calib_path)
        boxes_3d = anns['bboxes_3d']
        bv = plot_bbox3d(show_map, boxes_3d)

        im_bboxes = anns['im_bboxes']
        im = mmcv.imread(root_dir+'image_2/{}.png'.format(id))
        for bbox in im_bboxes:
            bbox = bbox.astype(np.int32)
            cv2.rectangle(im, tuple(bbox[:2]), tuple(bbox[2:]), (0,0,255), 1)

        # test result
        score_path = '/space/work_dirs/pillar_kitti_focalloss/results/data/' + '{}.txt'.format(id)
        if osp.exists(score_path):
            anns = load_kitti_anns(score_path, calib_path, gt_truth=False)
            boxes_3d = anns['bboxes_3d']
            scores = anns['scores']
            print(scores)
            boxes_3d = boxes_3d[scores[:,0] > 0.5,:]
            bv = plot_bbox3d(bv, boxes_3d, scalar=(55,177,0))

        # ## anchor test
        # anchors = gen_anchor().reshape((-1,7))
        # ## find the targets
        # box_encoding_fn = GroundBox3dCoder().encode
        # print('bboxes_3d: ', boxes_3d)
        # print('anchors: ', anchors[1000:1004, :])
        # res = get_anchor_target(anchors.reshape((-1, 7)), boxes_3d, similarity_fn, box_encoding_fn, neg_pos=100)
        # labels = res['labels'].reshape(-1).astype(np.long)
        # print(labels)
        # pos_inds = (labels > 0).nonzero()
        # neg_inds = (labels == 0).nonzero()
        # num_pos_samples = len(pos_inds[0])
        # num_neg_samples = len(neg_inds[0])
        # print('num_pos_samples: ', num_pos_samples)
        # print('num_neg_samples: ', num_neg_samples)
        # boxes_3d = anchors[labels>=0, :]
        # bv = plot_bbox3d(bv, boxes_3d, scalar=(125,92,0))

        show_map = np.zeros((im.shape[0]+bv.shape[0], im.shape[1], 3), dtype=np.uint8)
        show_map[:im.shape[0], :im.shape[1], :] = im
        show_map[im.shape[0]:, :bv.shape[1], :] = bv

        print(id)
        cv2.imshow('hh', show_map)
        cv2.waitKey()

    print('finished.')

if __name__ == '__main__':
    main()
