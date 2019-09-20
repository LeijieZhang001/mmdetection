'''
----------------------------------------------------------
  @File:     pc2ppillars.py
  @Brief:    convert point cloud to point pillars
  @Author:   Leijie.Zhang
  @Created:  21:51/9/19/2019
  @Modified: 21:51/9/19/2019
----------------------------------------------------------
'''

import numpy as np
import argparse
import math
import numba as nb

import mmcv

def load_pc(file_path, front_only=True):
    pc = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    if front_only:
        pc = pc[pc[:,0] >= 0, :]

    return pc

@nb.jit()
def gen_pillars(pc, N=100, pc_range=np.array([0, -39.68, -3, 69.12, 39.68, 1]), resolution=np.array([0.16, 0.16, 0.4])):
    """
    input:  N x 4 array
    output:
        pillars: PxNx9
        coords: Px2  # h,w == y,x
    """
    whc = (pc_range[3:] - pc_range[:3]) / resolution
    whc = [int(i) for i in whc]
    width, height, channel = whc

    hrange = np.arange(pc_range[4]-resolution[1]/2, pc_range[1], -resolution[1])
    wrange = np.arange(pc_range[0]+resolution[0]/2, pc_range[3], resolution[0])
    pillar_center = np.meshgrid(hrange, wrange, indexing='ij')
    pillar_center = np.stack(pillar_center).transpose((1,2,0))   # h x w x 2,  => y,x
    # print(pillar_center.shape)
    # print(pillar_center)
    assert pillar_center.shape[:2] == (height, width), 'pillar_center size not match.'

    pts_count = np.zeros((height, width), dtype=np.float32)
    pts_sum = np.zeros((height, width, 3), dtype=np.float32)  # x,y,z

    pillars = np.zeros((height, width, N, 9), dtype=np.float32)

    for point in pc:
        if point[0] >= pc_range[0] and point[0] < pc_range[3] and point[1] >= pc_range[1] and point[1] < pc_range[4] and point[2] >= pc_range[2] and point[2] < pc_range[5]:
            w, h, c = (point[:3] - pc_range[:3]) / resolution
            w = math.floor(w)
            h = math.floor(h)
            c = math.floor(c)
            pts_sum[h, w, 0] += point[0]
            pts_sum[h, w, 1] += point[1]
            pts_sum[h, w, 2] += point[2]
            pts_count[h, w] += 1

    pts_sum[...,0] = pts_sum[...,0] / pts_count
    pts_sum[...,1] = pts_sum[...,1] / pts_count
    pts_sum[...,2] = pts_sum[...,2] / pts_count
    pts_count = np.zeros((height, width), dtype=np.float32)

    for point in pc:
        if point[0] >= pc_range[0] and point[0] < pc_range[3] and point[1] >= pc_range[1] and point[1] < pc_range[4] and point[2] >= pc_range[2] and point[2] < pc_range[5]:
            w, h, c = (point[:3] - pc_range[:3]) / resolution
            w = math.floor(w)
            h = math.floor(h)
            c = math.floor(c)
            count = int(pts_count[h, w])
            if count >= 100:
                ## random sampling
                if np.random.randint(100) > 50:
                    #print('random...')
                    count = np.random.randint(N-1)
                else:
                    continue

            pillars[h, w, count, :4] = point
            pillars[h, w, count, 4] = point[1] - pillar_center[h, w, 0]
            pillars[h, w, count, 5] = point[0] - pillar_center[h, w, 1]
            pillars[h, w, count, 6] = point[0] - pts_sum[h, w, 0]
            pillars[h, w, count, 7] = point[1] - pts_sum[h, w, 1]
            pillars[h, w, count, 8] = point[2] - pts_sum[h, w, 2]
            pts_count[h, w] += 1

    res_pillars = []  # P x N x 9
    coords = [] # P x 2
    for h in range(height):
        for w in range(width):
            if int(pts_count[h,w]) > 0:
                res_pillars.append(pillars[h,w,...])
                coords.append(np.array([h,w]))

    return np.stack(res_pillars), np.stack(coords)

def main():
    root_dir = '/space/data/kitti/training/'
    ids = mmcv.list_from_file(root_dir+'../ImageSets/trainval.txt')

    pillars_dir = root_dir+'pointpillars_496x432_Px100x9_np16/'
    mmcv.mkdir_or_exist(pillars_dir)

    for i, id in enumerate(ids):
        if i < 3800:
            continue

        file_path = root_dir + 'velodyne/' + id + '.bin'

        # pc = load_pc(file_path)
        # pillars, coords = gen_pillars(pc)
        # print('pillars shape: ', pillars.shape)
        # print('coords shape: ', coords.shape)
        pillars = np.fromfile(root_dir+'pointpillars_496x432_Px100x9/pillars_{}.bin'.format(id), dtype=np.float32).reshape((-1,100,9))
        coords = np.fromfile(root_dir+'pointpillars_496x432_Px100x9/coords_{}.bin'.format(id), dtype=np.int64).reshape((-1,2))
        assert pillars.shape[0] == coords.shape[0], 'size mismatch'

        pillars.astype(np.float16).tofile(pillars_dir + 'pillars_{}.bin'.format(id))
        coords.astype(np.int16).tofile(pillars_dir + 'coords_{}.bin'.format(id))


        print(i)

if __name__ == '__main__':
    main()
