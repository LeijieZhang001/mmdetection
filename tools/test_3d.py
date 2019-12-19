'''
----------------------------------------------------------
  @File:     rpn_det.py
  @Brief:    update from second.pytorch repo
  @Author:   Leijie.Zhang
  @Created:  17:20/9/7/2019
  @Modified: 17:20/9/12/2019
----------------------------------------------------------
'''

import argparse
import os
import os.path as osp

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint

from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.core import bbox3d_lidar2cam, Calib

CLASSES = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}

def single_gpu_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result)

        batch_size = len(data['img'].data) if 'img' in data.keys() else len(data['gt_labels'].data[0])
        for _ in range(batch_size):
            prog_bar.update()
    return results

def write_results(ress, res_dir, lidar2cam, cfg):
    calib_dir = cfg.data.test.pc_prefix + '/../calib'

    res_dir += '/results/data'
    mmcv.mkdir_or_exist(res_dir)
    for i, res in enumerate(ress):
        # forward_test: batch_size should be 1
        res = res[0]
        with open(res_dir+'/{}.txt'.format(res['id']), 'w') as f:
            bbox3d = res['box3d_preds'].cpu().numpy()  # N x 7
            labels = res['label_preds'].cpu().numpy()  # N x 1
            scores = res['scores'].cpu().numpy()       # N x 1
            # per class
            for cls, bbox, score in zip(labels, bbox3d, scores):
                if bbox.any() and score > 0.1:
                    assert bbox.shape[0] == 7, 'bbox shape should be 7'
                    if lidar2cam:
                        calib = Calib(calib_dir+'/{}.txt'.format(res['id']))
                        bbox = bbox3d_lidar2cam(bbox.reshape((1, 7)), calib.Tr_velo_to_cam)
                        bbox = bbox.reshape((bbox.shape[1],))
                    line = [CLASSES[cls], 0, 0, -10]
                    line.extend([-1, -1, -1, -1])
                    line.extend([bbox[5], bbox[3], bbox[4], bbox[0], bbox[1], bbox[2], bbox[6], score])
                    #line.extend([-1000, -1000, -1000, -1000, -1000, -1000, -1000, 0.7])
                    f.writelines([str(i)+' ' for i in line])
                    f.write('\n')
            print('writing ', i)

def kitti_eval(res_dir, cfg):
    gt_dir = cfg.data.test.pc_prefix + '/../label_2_front'
    res_dir += '/results'
    print('gt_dir: ', gt_dir)
    print('result_dir: ', res_dir)
    eval_fun = './tools/kitti_devkit_object/kitti_eval/build/eval_3d ' + gt_dir + ' ' + res_dir
    os.system(eval_fun)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('res_dir', help='output result file')
    parser.add_argument('--lidar2cam', action='store_true', help='convert results from lidar coords to camera coords')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['bbox3d'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)

    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        raise NotImplementedError('distributed testing not support.')

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    pkl_res = args.res_dir+'/results.pkl'
    if osp.exists(pkl_res):
        outputs = mmcv.load(pkl_res)
    else:
        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_test(model, data_loader, args.show)
        else:
            raise NotImplementedError('distributed testing not support.')

        mmcv.dump(outputs, pkl_res)

    write_results(outputs, args.res_dir, args.lidar2cam, cfg)

    kitti_eval(args.res_dir, cfg)

if __name__ == '__main__':
    main()
