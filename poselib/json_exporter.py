#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File: json_exporter.py
@Auth: Huiqiao
@Date: 2022/7/12
"""
import json

import numpy as np
from isaacgym.torch_utils import *
from poselib.skeleton.skeleton3d import SkeletonMotion
from kinematics import Kinematics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def differential(d, dt=1/60):
    diff_data = []
    diff = 0
    for i in range(1, d.shape[0]):
        diff = (d[i, :] - d[i-1, :]) / dt
        diff_data.append(diff)
    diff_data.append(diff)
    return diff_data

# ['trunk', 'FR_hip', 'FR_thigh', 'FR_calf', 'FR_foot', 'FL_hip', 'FL_thigh', 'FL_calf', 'FL_foot', 'RR_hip',
# 'RR_thigh', 'RR_calf', 'RR_foot', 'RL_hip', 'RL_thigh', 'RL_calf', 'RL_foot']
json_dir = "data/load_config.json"
with open(json_dir, "r") as f:
    motion_json = json.load(f)
    file_name = motion_json["file_name"]
    clip = motion_json["clip"]
    remarks = motion_json["remarks"]

root_dir = "data/amp_hardware_a1/{}/".format(file_name)
file_name = file_name + '_amp_{}_{}_{}'.format(clip[0], clip[1], remarks)
input_file = root_dir + '{}.npy'.format(file_name)
target_motion = SkeletonMotion.from_file(input_file)
skeleton = target_motion.skeleton_tree

dt = 1/target_motion.fps

# root_pos, root_rot, joint_pos, foot_pos, lin_vel, ang_vel, joint_vel, foot_vel  order: [FR, FL, RR, RL]
global_trans = target_motion.global_transformation  # (len, 17, 7)

# root_pos: (len, 3), root_rot: (len, 4)
root_pos = global_trans[:, skeleton.node_names.index('trunk'), -3:].numpy()  # (len, 3)
root_rot = global_trans[:, skeleton.node_names.index('trunk'), :-3].numpy()  # (len, 4)

local_rot = target_motion.local_rotation  # (len, 17, 4)  (i, j, k, w)

local_rot = local_rot.reshape(-1, 4)
kinematic = Kinematics()
joint_pos_euler = []
for i in range(local_rot.shape[0]):
    axis_ang = np.array(kinematic.quaternion2axis_angle(local_rot[i, :]))
    joint_ang = axis_ang[:3] * axis_ang[-1]
    joint_pos_euler.append(joint_ang)
joint_pos_euler = np.array(joint_pos_euler).reshape((-1, 17, 3))

# joint_pos: (len, 12)
joint_pos = np.vstack([joint_pos_euler[:, 1, 0], joint_pos_euler[:, 2, 1], joint_pos_euler[:, 3, 1],
                       joint_pos_euler[:, 5, 0], joint_pos_euler[:, 6, 1], joint_pos_euler[:, 7, 1],
                       joint_pos_euler[:, 9, 0], joint_pos_euler[:, 10, 1], joint_pos_euler[:, 11, 1],
                       joint_pos_euler[:, 13, 0], joint_pos_euler[:, 14, 1], joint_pos_euler[:, 15, 1]]).T

foot_pos_fr = global_trans[:, skeleton.node_names.index('FR_foot'), -3:].numpy()
foot_pos_fl = global_trans[:, skeleton.node_names.index('FL_foot'), -3:].numpy()
foot_pos_rr = global_trans[:, skeleton.node_names.index('RR_foot'), -3:].numpy()
foot_pos_rl = global_trans[:, skeleton.node_names.index('RL_foot'), -3:].numpy()
# foot_pos: (len, 12)
foot_pos = np.hstack([foot_pos_fr, foot_pos_fl, foot_pos_rr, foot_pos_rl])

# lin_vel: (len, 3), ang_vel: (len, 3)
lin_vel = target_motion.global_root_velocity.numpy()
ang_vel = target_motion.global_root_angular_velocity.numpy()

# joint_vel: (len, 12)
joint_vel = np.array(differential(joint_pos, dt))

# foot_vel: (len, 12)
foot_vel = np.array(differential(foot_pos, dt))

# # joint_vel = np.ones_like(joint_vel)
# # foot_vel = np.ones_like(foot_vel)
# for i in range(joint_vel.shape[0]):
#     if np.sum(joint_vel[i, :]) == 0:
#         if i == 0:
#             joint_vel[i, :] = joint_vel[i+1, :]
#         elif i == joint_vel.shape[0] - 1:
#             joint_vel[i, :] = joint_vel[-3, :]
#             joint_vel[i-1, :] = joint_vel[-3, :]
#         else:
#             joint_vel[i, :] = joint_vel[i-1, :]
#
# for i in range(foot_vel.shape[0]):
#     if np.sum(foot_vel[i, :]) == 0:
#         if i == 0:
#             foot_vel[i, :] = foot_vel[i+1, :]
#         elif i == foot_vel.shape[0] - 1:
#             foot_vel[i, :] = foot_vel[-3, :]
#             foot_vel[i-1, :] = foot_vel[-3, :]
#         else:
#             foot_vel[i, :] = foot_vel[i-1, :]


# motion_data: (len, 61)
motion_data = np.hstack([root_pos, root_rot, joint_pos, foot_pos, lin_vel, ang_vel, joint_vel, foot_vel])

# # fr, fl, rr, rl
# plt.ion()
# plt.figure("3D Scatter", facecolor="lightgray")
# ax3d = plt.gca(projection="3d")  # 创建三维坐标
# for i in range(root_pos.shape[0]):
#     plt.cla()
#     ax3d.scatter(root_pos[i, 0], root_pos[i, 1], root_pos[i, 2], s=100)
#     ax3d.scatter(foot_pos[i, 0], foot_pos[i, 1], foot_pos[i, 2], s=100)
#     ax3d.scatter(foot_pos[i, 3], foot_pos[i, 4], foot_pos[i, 5], s=100)
#     ax3d.scatter(foot_pos[i, 6], foot_pos[i, 7], foot_pos[i, 8], s=100)
#     ax3d.scatter(foot_pos[i, 9], foot_pos[i, 10], foot_pos[i, 11], s=100)
#     plt.pause(0.5)
# plt.ioff()
# plt.show()

mocap_data = {}
mocap_data["LoopMode"] = "Wrap"
mocap_data["FrameDuration"] = dt
mocap_data["EnableCycleOffsetPosition"] = True
mocap_data["EnableCycleOffsetRotation"] = True
mocap_data["MotionWeight"] = 0.5

motion_data = [list(d) for d in motion_data]
mocap_data["Frames"] = motion_data

output_file = root_dir + "{}.json".format(file_name)
with open(output_file, 'w') as f:
    json.dump(mocap_data, f)
print('done!')
