import os.path as osp

import mmcv
import numpy as np

from .custom import CustomDataset
from .registry import DATASETS

@DATASETS.register_module
class KittiDataset(CustomDataset):

    #CLASSES = ('car','pedestrian','cyclist', 'van', 'person_sitting', 'truck', 'tram', 'misc', 'dontcare')
    CLASSES = ('car', 'pedestrian', 'cyclist')

    #def __init__(self, **kwargs):
    #    super(CustomDataset, self).__init__(**kwargs)

    def load_annotations(self, ann_file):
        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}
        img_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        img = mmcv.imread(self.img_prefix+'/{}.png'.format(img_ids[0]))
        height, width = img.shape[:-1]
        for img_id in img_ids:
            filename = '{}.png'.format(img_id)
            img_infos.append(dict(id=img_id, filename=filename, width=width, height=height))
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        label_path = osp.join(self.img_prefix, '../label_2',
                              '{}.txt'.format(img_id))
        ann = self.load_kitti_anns(label_path)

        ann = dict(
            bboxes=ann['im_bboxes'],
            labels=ann['labels'],
            bboxes_ignore=ann['bboxes_ignore'],
            labels_ignore=ann['labels_ignore'])
        return ann

    def load_kitti_anns(self, label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()

        truncated = []
        occluded = []
        gt_alphas = []
        labels = []
        im_bboxes = []
        bboxes_3d = []

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
            # ry, l, h, w, tx, ty, tz
            bbox_3d = [float(obj[14]), float(obj[10]), float(obj[8]), float(obj[9]),
                       float(obj[11]), float(obj[12]), float(obj[13])]

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

