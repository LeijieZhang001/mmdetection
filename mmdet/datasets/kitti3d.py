'''
----------------------------------------------------------
  @File:     kitti3d.py
  @Brief:    kitti3d dataset wrapper
  @Author:   Leijie.Zhang
  @Created:  18:20/9/6/2019
  @Modified: 18:20/9/122/2019
----------------------------------------------------------
'''

import os.path as osp

import mmcv
import numpy as np

from .point_cloud import PointCloudDataset
from .registry import DATASETS
from mmdet.core.dddet import bbox3d_cam2lidar, Calib

@DATASETS.register_module
class Kitti3dDataset(PointCloudDataset):

    #CLASSES = ('car','pedestrian','cyclist', 'van', 'person_sitting', 'truck', 'tram', 'misc', 'dontcare')
    CLASSES = ('car')

    #def __init__(self, **kwargs):
    #    super(CustomDataset, self).__init__(**kwargs)

    def load_annotations(self, ann_file):
        if isinstance(self.CLASSES, (list, tuple)):
            self.cat2label = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}
        else:
            self.cat2label = {self.CLASSES: 1}
        pc_infos = []
        pc_ids = mmcv.list_from_file(ann_file)
        for pc_id in pc_ids:
            filename = '{}.bin'.format(pc_id)
            pc_infos.append(dict(id=pc_id, filename=filename))
        return pc_infos

    def get_ann_info(self, idx):
        pc_id = self.pc_infos[idx]['id']
        label_path = osp.join(self.pc_prefix, '../label_2_front',
                              '{}.txt'.format(pc_id))
        calib_path = osp.join(self.pc_prefix, '../calib', '{}.txt'.format(pc_id))
        ann = self.load_kitti_anns(label_path, calib_path)

        ann = dict(
            bboxes_3d = ann['bboxes_3d'],
            bboxes=ann['im_bboxes'],
            labels=ann['labels'],
            calib=ann['calib'],
            bboxes_ignore=ann['bboxes_ignore'],
            labels_ignore=ann['labels_ignore'])
        return ann

    def load_kitti_anns(self, label_path, calib_path):
        calib = Calib(calib_path)

        with open(label_path, 'r') as f:
            lines = f.readlines()

        truncated = []
        occluded = []
        gt_alphas = []
        labels = []
        im_bboxes = []
        bboxes_3d = []
        calibs = []

        bboxes_ignore = []
        labels_ignore = []

        for ix, line in enumerate(lines):
            obj = line.strip().split(' ')

            # 0-based coordinates
            cls = obj[0].lower().strip()
            if cls not in self.CLASSES:
                continue
            label = self.cat2label[cls]
            im_bbox = [float(obj[4]), float(obj[5]), float(obj[6]), float(obj[7])]
            ## ry, l, h, w, tx, ty, tz
            #bbox_3d = [float(obj[14]), float(obj[10]), float(obj[8]), float(obj[9]),
            #           float(obj[11]), float(obj[12]), float(obj[13])]
            # 7: tx, ty, tz, w, l, h, ry
            bbox_3d = [float(obj[11]), float(obj[12]), float(obj[13]), float(obj[9]),
                       float(obj[10]), float(obj[8]), float(obj[14])]

            # filter objs with z < 0 in cam corrds
            if bbox_3d[2] < 0:
                continue

            # ignore objects with undetermined difficult level
            level = self.get_kitti_object_level(obj)
            if level > 3:
                bboxes_ignore.append(im_bbox)
                labels_ignore.append(label)
            else:
                truncated.append(float(obj[1]))
                occluded.append(float(obj[2]))
                gt_alphas.append(float(obj[3]))
                im_bboxes.append(im_bbox)
                labels.append(label)
                bboxes_3d.append(bbox_3d)


        if not im_bboxes:
            im_bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
            bboxes_3d = np.zeros((0, 7))
            truncated = np.zeros((0, ))
            occluded = np.zeros((0, ))
            gt_alphas = np.zeros((0, ))
        else:
            im_bboxes = np.array(im_bboxes, ndmin=2)
            labels = np.array(labels)
            bboxes_3d = np.array(bboxes_3d, ndmin=2)
            # 7: tx, ty, tz, w, l, h, ry
            # ==>
            # 7: tx, ty, tz, w, l, h, rz
            bboxes_3d = bbox3d_cam2lidar(bboxes_3d.reshape((-1, 7)), calib.Tr_cam_to_velo)
            truncated = np.array(truncated)
            occluded = np.array(occluded)
            gt_alphas = np.array(gt_alphas)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin = 2)
            labels_ignore = np.array(labels_ignore)

        return {'bboxes_3d' : bboxes_3d.astype(np.float32),
                'im_bboxes' : im_bboxes.astype(np.float32),
                'gt_alphas': gt_alphas.astype(np.float32),
                'labels' : labels.astype(np.int64),
                'calib': calib,
                'truncated': truncated.astype(np.float32),
                'occluded': occluded.astype(np.float32),
                'bboxes_ignore': bboxes_ignore.astype(np.float32),
                'labels_ignore': labels_ignore.astype(np.int64)}

    def get_kitti_object_level(self, obj):
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

