#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File: kinematics.py
@Auth: Huiqiao
@Date: 2022/7/6
"""
import math
import numpy as np
from isaacgym.torch_utils import *
import torch


class Kinematics:
    # default param of a1 (meter)
    def __init__(self, body_length=0.85, body_wide=0.094, hip_length=0.08505, thigh_length=0.2, calf_length=0.2):
        self.body_length = body_length
        self.body_wide = body_wide
        self.hip_length = hip_length
        self.thigh_length = thigh_length
        self.calf_length = calf_length

    def inverse_kinematics(self, base_pose_w, base_rot_w, foot_pos_w):
        '''
            Input
            base_pose_w: base position in world frame
            base_rot_w: base rotation in world frame (fixed frame)
            foot_pos_w: foot position in world frame (fr, fl, rr, rl)
            Output
            [fr_ang, fl_ang, rr_ang, rl_ang]: all joint angles
        '''
        foot_fr_w = foot_pos_w[:3]
        foot_fl_w = foot_pos_w[3:6]
        foot_rr_w = foot_pos_w[6:9]
        foot_rl_w = foot_pos_w[9:]

        t_wa = self.trans_matrix_ab(base_pose_w, base_rot_w)

        # base frame
        fr_link_pos = np.array([self.body_length, -self.body_wide / 2, 0])
        fl_link_pos = np.array([self.body_length, self.body_wide / 2, 0])
        rr_link_pos = np.array([0, -self.body_wide / 2, 0])
        rl_link_pos = np.array([0, self.body_wide / 2, 0])

        r_hip = [0, np.pi / 2, 0]
        t_ab_fr = self.trans_matrix_ab(fr_link_pos, r_hip)
        t_ab_fl = self.trans_matrix_ab(fl_link_pos, r_hip)
        t_ab_rr = self.trans_matrix_ab(rr_link_pos, r_hip)
        t_ab_rl = self.trans_matrix_ab(rl_link_pos, r_hip)

        fr_foot_hip = t_wa @ np.append(foot_fr_w, 1)
        fr_foot_hip = (t_ab_fr @ fr_foot_hip)[:-1]
        fl_foot_hip = t_wa @ np.append(foot_fl_w, 1)
        fl_foot_hip = (t_ab_fl @ fl_foot_hip)[:-1]
        rr_foot_hip = t_wa @ np.append(foot_rr_w, 1)
        rr_foot_hip = (t_ab_rr @ rr_foot_hip)[:-1]
        rl_foot_hip = t_wa @ np.append(foot_rl_w, 1)
        rl_foot_hip = (t_ab_rl @ rl_foot_hip)[:-1]

        # inverse kinematics TODO
        l_org_r = [self.hip_length, self.thigh_length, self.calf_length]
        l_org_l = [-self.hip_length, self.thigh_length, self.calf_length]

        fr_ang = np.array(self.inverse(fr_foot_hip, l_org_r))
        fl_ang = np.array(self.inverse(fl_foot_hip, l_org_l))
        rr_ang = np.array(self.inverse(rr_foot_hip, l_org_r))
        rl_ang = np.array(self.inverse(rl_foot_hip, l_org_l))

        fr_ang = np.array([fr_ang[0], -fr_ang[1], -fr_ang[2]])
        fl_ang = np.array([fl_ang[0], -fl_ang[1], -fl_ang[2]])
        rr_ang = np.array([rr_ang[0], -rr_ang[1], -rr_ang[2]])
        rl_ang = np.array([rl_ang[0], -rl_ang[1], -rl_ang[2]])

        return np.hstack([fr_ang, fl_ang, rr_ang, rl_ang])

    def inverse(self, p, l):
        '''
            Input
            p: foot end position in hip frame
            l: joint local coordinates
            Output
            [t1, t2, t3]: joint angles
        '''
        x1 = l[0] / (p[0] ** 2 + p[1] ** 2) ** 0.5
        t1 = np.arctan2(x1, (max(1 - x1 ** 2, 0.)) ** 0.5) + np.arctan2(p[1], p[0])

        r = (max(p[0] ** 2 + p[1] ** 2 - l[0] ** 2, 0.)) ** 0.5
        n = (p[2] ** 2 + r ** 2 + l[1] ** 2 - l[2] ** 2) / (2 * l[1])
        x2 = n / (max(p[2] ** 2 + r ** 2, 0.)) ** 0.5
        t2 = np.arctan2(x2, (max(1 - x2 ** 2, 0.)) ** 0.5) - np.arctan2(r, p[2])

        k = (p[2] ** 2 + r ** 2 - l[1] ** 2 - l[2] ** 2) / (2 * l[1] * l[2])
        t3 = np.arctan2((max(1 - k ** 2, 0.)) ** 0.5, k)
        return [t1, t2, t3]

    @staticmethod
    def rot_matrix_ba(t):
        r = np.array([[np.cos(t[2]) * np.cos(t[1]),
                       np.cos(t[2]) * np.sin(t[1]) * np.sin(t[0]) - np.sin(t[2]) * np.cos(t[0]),
                       np.cos(t[2]) * np.sin(t[1]) * np.cos(t[0]) + np.sin(t[2]) * np.sin(t[0])],
                      [np.sin(t[2]) * np.cos(t[1]),
                       np.sin(t[2]) * np.sin(t[1]) * np.sin(t[0]) + np.cos(t[2]) * np.cos(t[0]),
                       np.sin(t[2]) * np.sin(t[1]) * np.cos(t[0]) - np.cos(t[2]) * np.sin(t[0])],
                      [-np.sin(t[1]), np.cos(t[1]) * np.sin(t[0]), np.cos(t[1]) * np.cos(t[0])]])
        return r

    @staticmethod
    def rot_matrix_ab(t):
        r = np.array([[np.cos(t[2]) * np.cos(t[1]),
                       np.cos(t[2]) * np.sin(t[1]) * np.sin(t[0]) - np.sin(t[2]) * np.cos(t[0]),
                       np.cos(t[2]) * np.sin(t[1]) * np.cos(t[0]) + np.sin(t[2]) * np.sin(t[0])],
                      [np.sin(t[2]) * np.cos(t[1]),
                       np.sin(t[2]) * np.sin(t[1]) * np.sin(t[0]) + np.cos(t[2]) * np.cos(t[0]),
                       np.sin(t[2]) * np.sin(t[1]) * np.cos(t[0]) - np.cos(t[2]) * np.sin(t[0])],
                      [-np.sin(t[1]), np.cos(t[1]) * np.sin(t[0]), np.cos(t[1]) * np.cos(t[0])]])
        return r.T

    @staticmethod
    def trans_matrix_ba(m, t):
        r = np.array([[np.cos(t[2]) * np.cos(t[1]),
                       np.cos(t[2]) * np.sin(t[1]) * np.sin(t[0]) - np.sin(t[2]) * np.cos(t[0]),
                       np.cos(t[2]) * np.sin(t[1]) * np.cos(t[0]) + np.sin(t[2]) * np.sin(t[0])],
                      [np.sin(t[2]) * np.cos(t[1]),
                       np.sin(t[2]) * np.sin(t[1]) * np.sin(t[0]) + np.cos(t[2]) * np.cos(t[0]),
                       np.sin(t[2]) * np.sin(t[1]) * np.cos(t[0]) - np.cos(t[2]) * np.sin(t[0])],
                      [-np.sin(t[1]), np.cos(t[1]) * np.sin(t[0]), np.cos(t[1]) * np.cos(t[0])]])
        trans = np.hstack([r, np.array(m)[:, np.newaxis]])
        trans = np.vstack([trans, np.array([[0, 0, 0, 1]])])
        return trans

    @staticmethod
    def trans_matrix_ab(m, t):
        r = np.array([[np.cos(t[2]) * np.cos(t[1]),
                       np.cos(t[2]) * np.sin(t[1]) * np.sin(t[0]) - np.sin(t[2]) * np.cos(t[0]),
                       np.cos(t[2]) * np.sin(t[1]) * np.cos(t[0]) + np.sin(t[2]) * np.sin(t[0])],
                      [np.sin(t[2]) * np.cos(t[1]),
                       np.sin(t[2]) * np.sin(t[1]) * np.sin(t[0]) + np.cos(t[2]) * np.cos(t[0]),
                       np.sin(t[2]) * np.sin(t[1]) * np.cos(t[0]) - np.cos(t[2]) * np.sin(t[0])],
                      [-np.sin(t[1]), np.cos(t[1]) * np.sin(t[0]), np.cos(t[1]) * np.cos(t[0])]])
        trans = np.hstack([r.T, -np.dot(r.T, np.array(m))[:, np.newaxis]])
        trans = np.vstack([trans, np.array([[0, 0, 0, 1]])])
        return trans

    # Checks if a matrix is a valid rotation matrix.
    @staticmethod
    def isRotationMatrix(R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    def rotationMatrixToEulerAngles(self, R):
        assert (self.isRotationMatrix(R))
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        return [x, y, z]

    @staticmethod
    def quaternion2rpy(q):
        x, y, z, w = q[0], q[1], q[2], q[3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
        return [roll, pitch, yaw]

    @staticmethod
    def rpy2quaternion(r):
        roll, pitch, yaw = r[0], r[1], r[2]
        x = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        y = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        z = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        w = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return [x, y, z, w]

    @staticmethod
    def quaternion2axis_angle(q):
        x, y, z, w = q[0], q[1], q[2], q[3]
        angle = 2 * np.arccos(w)
        s = np.sqrt(1 - w * w)
        if s < 0.001:
            x = x
            y = y
            z = z
        else:
            x = x / s
            y = y / s
            z = z / s
        return [x, y, z, angle]


if __name__ == '__main__':
    kinematic = Kinematics()
    r_org = [0.0, -1.62406, 0.0]
    q_org = [0.0, -0.72568, 0.0, 0.68803]

    q = kinematic.rpy2quaternion(r_org)
    r = kinematic.quaternion2rpy(q)

    r_axis_angle = kinematic.quaternion2axis_angle(q)

    r_org = torch.tensor(r_org)
    q_org = torch.tensor(q_org).reshape(1, -1)
    q2 = quat_from_euler_xyz(r_org[0], r_org[1], r_org[2])
    r2 = torch.vstack(get_euler_xyz(q2.reshape(1, -1))).T
    aa = 1

    # body_length = 0.366
    # body_wide = 0.094
    # hip_length = 0.08505
    # thigh_length = 0.2
    # calf_length = 0.2
    # # body_trans = np.array([0, 0.1, (3**0.5)*(thigh_length + calf_length)/2])
    # body_trans = np.array([0, 0, thigh_length + calf_length - 0.0001])
    # body_rot = np.array([0, 0, 0])
    # fr_x = body_length
    # fr_y = hip_length + body_wide/2
    # fr_z = 0
    # fr = np.array([fr_x, -fr_y, fr_z])
    # fl = np.array([fr_x, fr_y, fr_z])
    # rr = np.array([0, -fr_y, fr_z])
    # rl = np.array([0, fr_y, fr_z])
    # foot_pos = np.hstack([fr, fl, rr, rl])
    # ang = kinematic.inverse_kinematics(body_trans, body_rot, foot_pos)
    # aa = 1
