'''
----------------------------------------------------------
  @File:     point_cloud.py
  @Brief:    point cloud dataset
  @Author:   Leijie.Zhang
  @Created:  15:35/9/6/2019
  @Modified: 15:35/9/20/2019
----------------------------------------------------------
'''

import os.path as osp
import warnings

import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from .utils import to_tensor
from .registry import DATASETS
from .pc_utils import feats_encoder

@DATASETS.register_module
class PointCloudDataset(Dataset):
    """Custom dataset for point cloud 3D detection.

    Annotation format:
    [
        {
            'filename': 'a.bin',
            'num': points number,
            'ele': elements of each point,
            'ann': {
                'bboxes': <np.ndarray> (n, 7),    # kitti format, tx ty tz w l h ry
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4),
                'labels_ignore': <np.ndarray> (k) (optional field)
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    CLASSES = None

    def __init__(self,
                 ann_file,
                 pc_prefix,
                 with_label=True,
                 handcraft_feats=False,
                 load_feats=False,
                 pc_range=None,
                 resolution=None,
                 skip_pc_without_anno=True,
                 test_mode=False):
        # prefix of point cloud path
        self.pc_prefix = pc_prefix
        self.load_feats = load_feats

        # load annotations (and proposals)
        self.pc_infos = self.load_annotations(ann_file)

        # point cloud feature encoder
        self.handcraft_feats = handcraft_feats
        if self.handcraft_feats:
            self.pc_range = pc_range
            self.resolution = resolution

        self.with_label = with_label
        self.skip_pc_without_anno = skip_pc_without_anno
        # in test mode or not
        self.test_mode = test_mode

        # set group flag for the sampler
        # if not self.test_mode:
        self._set_group_flag()

    def __len__(self):
        return len(self.pc_infos)

    def load_annotations(self, ann_file):
        return mmcv.load(ann_file)

    def _set_group_flag(self):
        """Set flag with 0 for only one group
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        # if self.test_mode:
        #     return self.prepare_test_pc(idx)
        while True:
            data = self.prepare_train_pc(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_pc(self, idx):
        # # load point cloud. coordinate system: x: forward, y: left, z: up
        # pc = np.fromfile(osp.join(self.pc_prefix, self.pc_infos[idx]['filename']), dtype=np.float32).reshape(-1, 4)
        # # filter pc with x < 0 in lidar coords
        # pc = pc[pc[:,0] >= 0, :]
        # self.pc_infos[idx]['num'], self.pc_infos[idx]['ele'] = pc.shape

        pc_info = self.pc_infos[idx]

        ann = self.get_ann_info(idx)
        gt_bboxes = ann['bboxes_3d']
        gt_labels = ann['labels']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0 and self.skip_pc_without_anno:
            warnings.warn('Skip the point cloud "%s" that has no valid gt bbox' %
                          osp.join(self.pc_prefix, pc_info['filename']))
            return None

        # pts_shape = (pc_info['num'], pc_info['ele'])
        pc_meta = dict(id=pc_info['id'])
        #    pts_shape=pts_shape)

        if self.handcraft_feats:
            if self.load_feats:
                feats = np.fromfile(osp.join(self.pc_prefix, pc_info['filename']), dtype=np.float32).reshape(10, 496, 432)
            else:
                feats = feats_encoder(pc, np.array(self.pc_range), np.array(self.resolution))


            pc_meta['feats_shape'] = feats.shape

            data = dict(
                feats=DC(to_tensor(feats), stack=True),
                pc_meta=DC(pc_meta, cpu_only=True),
                gt_bboxes=DC(to_tensor(gt_bboxes)))
        else:
            if self.load_feats:
                pillars = np.fromfile(osp.join(self.pc_prefix, 'pillars_'+pc_info['filename']), dtype=np.float16).reshape((-1, 100, 9)).astype(np.float32)
                coords = np.fromfile(osp.join(self.pc_prefix, 'coords_'+pc_info['filename']), dtype=np.int16).reshape((-1, 2)).astype(np.int32)
                assert pillars.shape[0] == coords.shape[0], 'pillars & coords size mismatch.'
            else:
                raise NotImplementedError

            data = dict(
                feats=DC(to_tensor(pillars)),
                coords=DC(to_tensor(coords)),
                pc_meta=DC(pc_meta, cpu_only=True),
                gt_bboxes=DC(to_tensor(gt_bboxes)))

        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        return data
