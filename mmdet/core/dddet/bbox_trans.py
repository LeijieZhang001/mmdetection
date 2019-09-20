'''
----------------------------------------------------------
  @File:     bbox_trans.py
  @Brief:    update from data.py with kitti
  @Author:   Leijie.Zhang
  @Created:  17:20/9/12/2019
  @Modified: 17:20/9/12/2019
----------------------------------------------------------
'''

import numpy as np
from scipy import linalg

def corners3d_transform(corners_3d, T):
    """
    Input
      corners_3d: Nx24 array, (x0 .. x7 y0 ... y7 z0 ... z7) in world coordinate
      Tr:         4x4 transformation matrix
    """
    if len(corners_3d.shape) == 1:
        corners_3d = corners_3d.reshape(-1, 24)
    assert corners_3d.shape[1] == 24

    corners_3d_new = np.zeros(corners_3d.shape, dtype=np.float32)
    for i in range(corners_3d.shape[0]):
        corners = np.vstack((corners_3d[i, :].reshape(3, 8), np.ones((1, 8))))
        corners_3d_new[i, :] = T.dot(corners)[:3, :].ravel()

    return corners_3d_new

def transform3d(pts, T):
    """
    Input
      pts: Nx3 array storing 3D points (x y z)
      T:   4x4 transformation matrix
    """
    assert pts.shape[1] == 3
    pts_new = T.dot(np.vstack((pts.T, np.ones((1, pts.shape[0])))))[:3, :].T

    return pts_new

def bbox3d_to_corners3d_cam(boxes_3d):
    """
    Input
      boxes_3d: Nx7 array, (ry l h w tx ty tz) in world coordinate
    Return:
      corners_3d: Nx24 array, (x0 ... x7 y0 ... y7 z0 ... z7) in world coordinate
    """
    if len(boxes_3d.shape) == 1:
        boxes_3d = boxes_3d.reshape(1, 7)
    assert boxes_3d.shape[1] == 7

    # convert to (x0 ... x7 y0 ... y7 z0 ... z7)
    corners_3d = np.zeros((boxes_3d.shape[0], 24), dtype=np.float32)
    for i in range(boxes_3d.shape[0]):
        ry, l, h, w = boxes_3d[i, :4]
        R = np.array([[np.cos(ry), 0, np.sin(ry)],
                      [0, 1, 0],
                      [-np.sin(ry), 0, np.cos(ry)]])

        x_corners = l * 0.5 * np.array([1,1,-1,-1,1,1,-1,-1], dtype=np.float32)
        y_corners = h * 0.5 * np.array([1,1,1,1,-1,-1,-1,-1], dtype=np.float32)
        z_corners = w * 0.5 * np.array([1,-1,-1,1,1,-1,-1,1], dtype=np.float32)
        corners = np.dot(R, np.vstack((x_corners, y_corners, z_corners)))
        # corners: 3x8 array
        corners +=  boxes_3d[i, 4:7].reshape(3, 1)
        corners_3d[i, :] = corners.ravel()

    return corners_3d

def bbox3d_to_corners3d_lidar(boxes_3d):
    """
    Input
      boxes_3d: Nx7 array, (rz l w h tx ty tz) in lidar coordinate
    Return:
      corners_3d: Nx24 array, (x0 ... x7 y0 ... y7 z0 ... z7) in lidar coordinate
    """
    if len(boxes_3d.shape) == 1:
        boxes_3d = boxes_3d.reshape(1, 7)
    assert boxes_3d.shape[1] == 7

    # convert to (x0 ... x7 y0 ... y7 z0 ... z7)
    corners_3d = np.zeros((boxes_3d.shape[0], 24), dtype=np.float32)
    for i in range(boxes_3d.shape[0]):
        rz, l, w, h = boxes_3d[i, :4]
        R = np.array([[np.cos(rz), -np.sin(rz), 0],
                      [np.sin(rz), np.cos(rz), 0],
                      [0, 0, 1]])

        x_corners = l * 0.5 * np.array([1,1,-1,-1,1,1,-1,-1], dtype=np.float32)
        y_corners = w * 0.5 * np.array([1,-1,-1,1,1,-1,-1,1], dtype=np.float32)
        z_corners = h * 0.5 * np.array([-1,-1,-1,-1,1,1,1,1], dtype=np.float32)
        corners = np.dot(R, np.vstack((x_corners, y_corners, z_corners)))
        # corners: 3x8 array
        corners +=  boxes_3d[i, 4:7].reshape(3, 1)
        corners_3d[i, :] = corners.ravel()

    return corners_3d

def bbox3d_cam2lidar(boxes_3d, T):
    """
    INPUT
        #boxes_3d:        (ry, l, h, w, tx, ty, tz) in camera coordinates
        boxes_3d:        (tx, ty, tz, w, l, h, ry) in camera coordinates
                         ry:    yaw angle (around Y-axis) in camera coordinates
                                          -pi/2  (+Z)
                                             |
                                   -pi/pi ------- 0  (+X)
                                             |
                                            pi/2
        T:     3x4 or 4x4 ndarray, transformation matrix from camera coordinates to LIDAR coordinates
    OUTPUT
        #boxes_3d_lidar:  (rz, l, w, h, tx, ty, tz) in LIDAR coordinates
        boxes_3d_lidar:  (tx, ty, tz, w, l, h, rz) in LIDAR coordinates
                         rz:    yaw angle (around Z-axis) in LIDAR coordinates
                                             0  (+X)
                                             |
                                (+Y) pi/2 ------- -pi/2
                                             |
                                          -pi/pi
    """
    boxes_3d_bk = boxes_3d.copy()
    boxes_3d[:,4:] = boxes_3d_bk[:,:3]
    boxes_3d[:,3] = boxes_3d_bk[:,3]
    boxes_3d[:,2] = boxes_3d_bk[:,5]
    boxes_3d[:,1] = boxes_3d_bk[:,4]
    boxes_3d[:,0] = boxes_3d_bk[:,6]

    boxes_3d_lidar = np.zeros(boxes_3d.shape)
    corners_3d = bbox3d_to_corners3d_cam(boxes_3d)
    corners_3d_lidar = corners3d_transform(corners_3d, T)
    # the 1st corner corresponds to head
    p01 = 0.5 * (corners_3d_lidar[:, [0,8]] + corners_3d_lidar[:, [1,9]])
    ctrs = 0.25 * (corners_3d_lidar[:, [0,8]] + corners_3d_lidar[:, [1,9]] + \
                   corners_3d_lidar[:, [2,10]] + corners_3d_lidar[:, [3,11]])
    deltas = p01 - ctrs
    rz = np.arctan2(deltas[:, 1], deltas[:, 0])   # delta_y, delta_x
    boxes_3d_lidar[:, 0] = rz
    # l, h, w -> l, w, h
    boxes_3d_lidar[:, 1:4] = boxes_3d[:, [1,3,2]]
    # tx, ty, tz
    boxes_3d_lidar[:, 4:] = transform3d(boxes_3d[:, 4:], T)

    boxes_3d_lidar_bk = boxes_3d_lidar.copy()
    boxes_3d_lidar[:,:3] = boxes_3d_lidar_bk[:,4:]
    boxes_3d_lidar[:,3] = boxes_3d_lidar_bk[:,2]
    boxes_3d_lidar[:,4] = boxes_3d_lidar_bk[:,1]
    boxes_3d_lidar[:,5] = boxes_3d_lidar_bk[:,3]
    boxes_3d_lidar[:,6] = boxes_3d_lidar_bk[:,0]
    return boxes_3d_lidar

def bbox3d_lidar2cam(boxes_3d, T):
    """
    INPUT
        #boxes_3d:       (rz, l, w, h, tx, ty, tz) in LIDAR coordinates
        boxes_3d:        (tx, ty, tz, w, l, h, rz) in LIDAR coordinates
                         rz:    yaw angle (around Z-axis) in LIDAR coordinates
                                             0  (+X)
                                             |
                                (+Y) pi/2 ------- -pi/2
                                             |
                                          -pi/pi
        T:     3x4 or 4x4 ndarray, transformation matrix from LIDAR coordinates to camera coordinates
    OUTPUT
        #boxes_3d_cam:   (ry, l, h, w, tx, ty, tz) in camera coordinates
        boxes_3d:        (tx, ty, tz, w, l, h, ry) in camera coordinates
                         ry:    yaw angle (around Y-axis) in camera coordinates
                                          -pi/2  (+Z)
                                             |
                                   -pi/pi ------- 0  (+X)
                                             |
                                            pi/2
    """
    boxes_3d_bk = boxes_3d.copy()
    boxes_3d[:,4:] = boxes_3d_bk[:,:3]
    boxes_3d[:,3] = boxes_3d_bk[:,5]
    boxes_3d[:,2] = boxes_3d_bk[:,3]
    boxes_3d[:,1] = boxes_3d_bk[:,4]
    boxes_3d[:,0] = boxes_3d_bk[:,6]

    boxes_3d_cam = np.zeros(boxes_3d.shape)
    corners_3d = bbox3d_to_corners3d_lidar(boxes_3d)
    corners_3d_cam = corners3d_transform(corners_3d, T)
    # the 1st corner corresponds to head
    p01 = 0.5 * (corners_3d_cam[:, [0,16]] + corners_3d_cam[:, [1,17]])
    ctrs = 0.25 * (corners_3d_cam[:, [0,16]] + corners_3d_cam[:, [1,17]] + \
                   corners_3d_cam[:, [2,18]] + corners_3d_cam[:, [3,19]])
    deltas = p01 - ctrs
    ry = -np.arctan2(deltas[:, 1], deltas[:, 0])   # delta_z, delta_x
    boxes_3d_cam[:, 0] = ry
    # l, w, h -> l, h, w
    boxes_3d_cam[:, 1:4] = boxes_3d[:, [1,3,2]]
    # tx, ty, tz
    boxes_3d_cam[:, 4:] = transform3d(boxes_3d[:, 4:], T)

    boxes_3d_cam_bk = boxes_3d_cam.copy()
    boxes_3d_cam[:,:3] = boxes_3d_cam_bk[:,4:]
    boxes_3d_cam[:,3] = boxes_3d_cam_bk[:,3]
    boxes_3d_cam[:,4] = boxes_3d_cam_bk[:,1]
    boxes_3d_cam[:,5] = boxes_3d_cam_bk[:,2]
    boxes_3d_cam[:,6] = boxes_3d_cam_bk[:,0]
    return boxes_3d_cam

class Camera(object):
    """ Class for representing pin-hole cameras. """

    def __init__(self,P):
        """ Initialize P = K[R|t] camera model. """
        self.P = P
        self.K, self.R, self.t = self.factor()

    @property
    def f(self):
        return self.K[0, 0]

    @property
    def cx(self):
        return self.K[0, 2]

    @property
    def cy(self):
        return self.K[1, 2]

    @property
    def tx(self):
        return self.t[0,0]

    @property
    def ty(self):
        return self.t[1,0]

    def factor(self):
        """ Factorize the camera matrix into K,R,t as P = K[R|t]. """

        # factor first 3*3 part
        K, R = linalg.rq(self.P[:,:3])

        # make diagonal of K positive
        T = np.diag(np.sign(np.diag(K)))
        if linalg.det(T) < 0:
            T[1,1] *= -1

        K = np.dot(K, T)
        R = np.dot(T, R) # T is its own inverse
        t = np.dot(linalg.inv(K), self.P[:,3])

        return K, R, t

class Calib(object):
    def __init__(self, calib_path):
        calib = {}
        self.is_tracking_dataset = False
        with open(calib_path, 'r') as cf:
            for line in cf:
                fields = line.split()
                if len(fields) is 0:
                    continue
                key = fields[0][:-1]
                if key == 'Tr_velo_ca' or key == 'Tr_imu_vel' or key == 'R_rec':
                    self.is_tracking_dataset = True
                val = np.asmatrix(fields[1:]).astype(np.float32).reshape(3, -1)
                calib[key] = val
        #print("cf: ", cf)
        # KITTI tracking dataset calib file is extremely annoying for this inconsistent data format!!
        if self.is_tracking_dataset:
            calib['Tr_velo_to_cam'] = np.vstack((calib['Tr_velo_ca'],
                                                      [0, 0, 0, 1]))
            calib['R0_rect'] = np.hstack((calib['R_rec'],
                                               np.zeros((3, 1))))
            calib['R0_rect'] = np.vstack((calib['R0_rect'],
                                               [0, 0, 0, 1]))
            calib['Tr_velo_to_rect'] = calib['R0_rect'] * calib['Tr_velo_to_cam']
            calib['Tr_rect_to_velo'] = np.linalg.inv(calib['Tr_velo_to_rect'])
            calib['P_velo_to_left'] = calib['P2'] * calib['Tr_velo_to_rect']
        else:
            calib['Tr_velo_to_cam'] = np.vstack((calib['Tr_velo_to_cam'],
                                                      [0, 0, 0, 1]))
            calib['Tr_cam_to_velo'] = np.linalg.inv(calib['Tr_velo_to_cam'])
            calib['R0_rect'] = np.hstack((calib['R0_rect'],
                                               np.zeros((3, 1))))
            calib['R0_rect'] = np.vstack((calib['R0_rect'],
                                               [0, 0, 0, 1]))
            calib['Tr_velo_to_rect'] = calib['R0_rect'] * calib['Tr_velo_to_cam']
            calib['Tr_rect_to_velo'] = np.linalg.inv(calib['Tr_velo_to_rect'])
            calib['P_velo_to_left'] = calib['P2'] * calib['Tr_velo_to_rect']

        calib['P_left'] = calib['P2']
        calib['P_right'] = calib['P3']
        self.calib = calib
        self.lcam = Camera(self.P2)
        self.rcam = Camera(self.P3)
        self.baseline = abs(self.lcam.tx - self.rcam.tx)

    def to_dict(self):
        return self.calib

    @property
    def Tr_velo_to_cam(self):
        return self.calib['Tr_velo_to_cam']

    @property
    def Tr_cam_to_velo(self):
        return self.calib['Tr_cam_to_velo']

    @property
    def Tr_velo_to_rect(self):
        return self.calib['Tr_velo_to_rect']

    @property
    def Tr_rect_to_velo(self):
        return self.calib['Tr_rect_to_velo']

    @property
    def R0_rect(self):
        return self.calib['R0_rect']

    @property
    def P2(self):
        return self.calib['P2']

    @property
    def P3(self):
        return self.calib['P3']

    @property
    def P_velo_to_left(self):
        return self.calib['P_velo_to_left']
