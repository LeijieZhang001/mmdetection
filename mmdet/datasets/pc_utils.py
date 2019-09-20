'''
----------------------------------------------------------
  @File:     pc_utils.py
  @Brief:    point cloud utils
  @Author:   Leijie.Zhang
  @Created:  17:20/9/9/2019
  @Modified: 17:20/9/9/2019
----------------------------------------------------------
'''

import math
import numpy as np
import cv2

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

def debug_feats(feats, channel=None, show=True):
    assert len(feats.shape) == 3, 'feats dim should be 3'
    for c in range(feats.shape[0]):
        if channel is not None and channel != c:
            continue
        img = np.zeros((feats.shape[1], feats.shape[2], 3), dtype = np.uint8)
        feat = feats[c,...]
        img[feat > 0, :] = (0, 120, 169)
        if show:
            cv2.imshow("hh", img)
            cv2.waitKey()
        else:
            return img

def main():
    pc = np.fromfile('/space/data/kitti/training/velodyne_reduced/002990.bin', dtype = np.float32).reshape(-1, 4)
    feats = feats_encoder(pc)
    debug_feats(feats)

if __name__ == '__main__':
    main()
