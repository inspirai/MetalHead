#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File: key_points_retargetting.py
@Auth: Huiqiao
@Date: 2022/7/6
"""
import copy
import json
import numpy as np
from isaacgym.torch_utils import *
from kinematics import Kinematics
from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
from poselib.skeleton.backend.fbx.fbx_read_wrapper import fbx_to_array
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# get key points
# transforms:(len, 27, 4, 4)
# ['Hips', 'Spine', 'Spine1', 'Neck', 'Head', 'Head_End', 'LeftShoulder', 'LeftArm',
#  'LeftForeArm', 'LeftHand', 'LeftHand_End', 'RightShoulder', 'RightArm', 'RightForeArm',
#  'RightHand', 'RightHand_End', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftFoot_End',
#  'RightUpLeg', 'RightLeg', 'RightFoot', 'RightFoot_End', 'Tail', 'Tail1', 'Tail1_End']

# ['trunk', 'FR_hip', 'FR_thigh', 'FR_calf', 'FR_foot', 'FL_hip', 'FL_thigh', 'FL_calf', 'FL_foot', 'RR_hip',
# 'RR_thigh', 'RR_calf', 'RR_foot', 'RL_hip', 'RL_thigh', 'RL_calf', 'RL_foot']

# source file path
json_dir = "data/load_config.json"
with open(json_dir, "r") as f:
    motion_json = json.load(f)
    file_name = motion_json["file_name"]
    clip = motion_json["clip"]
    remarks = motion_json["remarks"]

root_dir = "data/amp_hardware_a1/{}/".format(file_name)
output_file = '{}_amp_{}_{}_{}'.format(file_name, clip[0], clip[1], remarks)

# target robot parameters (meter)
body_length = 0.366
body_wide = 0.094
hip_length = 0.08505
thigh_length = 0.2
calf_length = 0.2
kinematic = Kinematics(body_length=0.366, body_wide=0.094, hip_length=0.08505, thigh_length=0.2, calf_length=0.2)

# load and visualize source motion sequence
source_motion = SkeletonMotion.from_file(root_dir + '{}.npy'.format(file_name))  # generate from fbx_importer

# zoom according to source t-pose and target t-pose
source_tpose = SkeletonState.from_file('data/dog_tpose.npy')
# plot_skeleton_state(source_tpose)
target_tpose = SkeletonState.from_file('data/amp_a1_tpose.npy')
# plot_skeleton_state(target_tpose)
skeleton_s = source_tpose.skeleton_tree
skeleton_t = target_tpose.skeleton_tree
source_length = torch.abs(source_tpose.global_translation[skeleton_s.index('LeftHand_End'), 0] -
                          source_tpose.global_translation[skeleton_s.index('LeftFoot_End'), 0])
source_wide = torch.abs(source_tpose.global_translation[skeleton_s.index('LeftHand_End'), 1] -
                        source_tpose.global_translation[skeleton_s.index('RightHand_End'), 1])
source_height = torch.abs(source_tpose.global_translation[skeleton_s.index('Hips'), 2] -
                          source_tpose.global_translation[skeleton_s.index('LeftFoot_End'), 2])

target_length = torch.abs(target_tpose.global_translation[skeleton_t.index('FL_foot'), 0] -
                          target_tpose.global_translation[skeleton_t.index('RL_foot'), 0])
target_wide = torch.abs(target_tpose.global_translation[skeleton_t.index('FL_foot'), 1] -
                        target_tpose.global_translation[skeleton_t.index('FR_foot'), 1])
target_height = torch.abs(target_tpose.global_translation[skeleton_t.index('trunk'), 2] -
                          target_tpose.global_translation[skeleton_t.index('FL_foot'), 2])

zoom_x = (target_length / source_length) * 1.  # height
zoom_y = target_wide / source_wide
zoom_z = target_height / source_height

skeleton = source_motion.skeleton_tree
local_translation = skeleton.local_translation  # (27, 3)
root_translation = source_motion.root_translation  # (len, 3)

print('zoom_x: {}, zoom_y: {}, zoom_z: {}'.format(zoom_x, zoom_y, zoom_z))
for i in range(local_translation.shape[0]):
    if i == skeleton.node_names.index('LeftFoot_End') or i == skeleton.node_names.index('RightFoot_End'):
        local_translation[i, 0] *= zoom_y
        local_translation[i, 1] *= zoom_x
        local_translation[i, 2] *= zoom_z
        continue
    local_translation[i, 0] *= zoom_x
    local_translation[i, 1] *= zoom_y
    local_translation[i, 2] *= zoom_z
root_translation[:, 0] *= zoom_x
root_translation[:, 1] *= zoom_y
root_translation[:, 2] *= zoom_z

# global_root_translation 在source_motion.root_translation改变后自动更新
# global_root_translation = root_translation
global_trans = source_motion.global_transformation  # (len， 27, 7)
global_root_translation = global_trans[:, skeleton.node_names.index("Hips"), -3:]  # (len, 7)

# global_left_shoulder_trans = global_trans[:, skeleton.node_names.index("LeftShoulder"), -3:]
# global_right_shoulder_trans = global_trans[:, skeleton.node_names.index("RightShoulder"), -3:]  # (len, 3)
# global_left_up_leg_trans = global_trans[:, skeleton.node_names.index("LeftUpLeg"), -3:]  # (len, 3)
# global_right_up_leg_trans = global_trans[:, skeleton.node_names.index("RightUpLeg"), -3:]  # (len, 3)

global_left_hand_end_trans = global_trans[:, skeleton.node_names.index("LeftHand_End"), -3:]  # (len, 3)
global_right_hand_end_trans = global_trans[:, skeleton.node_names.index("RightHand_End"), -3:]  # (len, 3)
global_left_foot_end_trans = global_trans[:, skeleton.node_names.index("LeftFoot_End"), -3:]  # (len, 3)
global_right_foot_end_trans = global_trans[:, skeleton.node_names.index("RightFoot_End"), -3:]  # (len, 3)

global_root_rotation_quat = global_trans[:, skeleton.node_names.index("Hips"), :4]
# TODO: get_euler_xyz和kinematic.quart_to_rpy结果不一样？
# global_root_rotation_euler = torch.vstack(get_euler_xyz(global_root_rotation_quat)).T.numpy()

global_root_rotation_quat = global_root_rotation_quat.numpy()
global_root_rotation_euler = [kinematic.quaternion2rpy(q) for q in
                              global_root_rotation_quat]
global_root_rotation_euler = np.array(global_root_rotation_euler)

# plt.scatter(list(range(global_root_rotation_euler.shape[0])), global_root_rotation_euler[:, 0], label='x')
# plt.scatter(list(range(global_root_rotation_euler.shape[0])), global_root_rotation_euler[:, 1], label='y')
# plt.scatter(list(range(global_root_rotation_euler.shape[0])), global_root_rotation_euler[:, 2], label='z')
# plt.legend()
# plt.show()

# plt.figure("3D Scatter", facecolor="lightgray")
# ax3d = plt.gca(projection="3d")  # 创建三维坐标
# gt = source_motion.global_transformation[0, ...]
# gt[:, -3] *= zoom_x
# gt[:, -2] *= zoom_y
# gt[:, -1] *= zoom_z
# for i in range(27):
#     ax3d.scatter(gt[i, -3], gt[i, -2], gt[i, -1])
# plt.show()

# inverse kinematic
g_root_trans = global_root_translation.numpy()[clip[0]:clip[1], :]  # (len, 3)
g_root_rot_euler = global_root_rotation_euler[clip[0]:clip[1], :]  # (len, 3)
g_lh_trans = global_left_hand_end_trans.numpy()[clip[0]:clip[1], :]  # (len, 3)
g_rh_trans = global_right_hand_end_trans.numpy()[clip[0]:clip[1], :]  # (len, 3)
g_lf_trans = global_left_foot_end_trans.numpy()[clip[0]:clip[1], :]  # (len, 3)
g_rf_trans = global_right_foot_end_trans.numpy()[clip[0]:clip[1], :]  # (len, 3)

for i in range(g_root_rot_euler.shape[0]):
    if g_root_rot_euler[i, 2] <= np.pi / 2:
        g_root_rot_euler[i, 2] = 2*np.pi + g_root_rot_euler[i, 2]
g_root_rot_euler[:, 2] = g_root_rot_euler[:, 2] - np.pi

g_root_rot_euler[:, 0] = np.pi / 2 - g_root_rot_euler[:, 0]
for i in range(g_root_rot_euler.shape[0]):
    for j in range(3):
        if g_root_rot_euler[i, j] >= np.pi:
            g_root_rot_euler[i, j] = g_root_rot_euler[i, j] - 2 * np.pi
        elif g_root_rot_euler[i, j] <= -np.pi:
            g_root_rot_euler[i, j] = 2 * np.pi + g_root_rot_euler[i, j]
g_root_rot_euler[:, 0] = -g_root_rot_euler[:, 0]
g_root_rot_euler[:, 1] = -g_root_rot_euler[:, 1]

# plt.scatter(list(range(g_root_rot_euler.shape[0])), g_root_rot_euler[:, 0], label='x')
# plt.scatter(list(range(g_root_rot_euler.shape[0])), g_root_rot_euler[:, 1], label='y')
# plt.scatter(list(range(g_root_rot_euler.shape[0])), g_root_rot_euler[:, 2], label='z')
# plt.legend()
# plt.show()

# TODO: 旋转到x轴朝向
rot_matrix = kinematic.rot_matrix_ba([0, 0, -g_root_rot_euler[0, -1]])
g_root_trans = np.array([rot_matrix@t for t in g_root_trans])
g_root_rot_euler[:, -1] -= g_root_rot_euler[0, -1]
g_lh_trans = np.array([rot_matrix@t for t in g_lh_trans])
g_rh_trans = np.array([rot_matrix@t for t in g_rh_trans])
g_lf_trans = np.array([rot_matrix@t for t in g_lf_trans])
g_rf_trans = np.array([rot_matrix@t for t in g_rf_trans])

g_foot_trans = np.hstack([g_rh_trans, g_lh_trans, g_rf_trans, g_lf_trans])
# # fr, fl, rr, rl
# plt.ion()
# plt.figure("3D Scatter", facecolor="lightgray")
# ax3d = plt.gca(projection="3d")  # 创建三维坐标
# for i in range(g_foot_trans.shape[0]):
#     plt.cla()
#     ax3d.scatter(g_root_trans[i, 0], g_root_trans[i, 1], g_root_trans[i, 2], s=100)
#     ax3d.scatter(g_foot_trans[i, 0], g_foot_trans[i, 1], g_foot_trans[i, 2], s=100)
#     ax3d.scatter(g_foot_trans[i, 3], g_foot_trans[i, 4], g_foot_trans[i, 5], s=100)
#     ax3d.scatter(g_foot_trans[i, 6], g_foot_trans[i, 7], g_foot_trans[i, 8], s=100)
#     ax3d.scatter(g_foot_trans[i, 9], g_foot_trans[i, 10], g_foot_trans[i, 11], s=100)
#     plt.pause(0.1)
# plt.ioff()
# plt.show()

# TODO: 重心位置向后调整
root_new_local = np.array([-body_length / 3.5, 0, 0, 1])
g_root_trans = np.array([list((kinematic.trans_matrix_ba(m, t) @ root_new_local)[:3])
                         for m, t in zip(g_root_trans, g_root_rot_euler)])

# 计算关节转角
joint_rot_euler = np.zeros((g_root_trans.shape[0], 12))
for i in range(g_root_trans.shape[0]):
    ang = kinematic.inverse_kinematics(g_root_trans[i, :], g_root_rot_euler[i, :], g_foot_trans[i, :])
    joint_rot_euler[i, :] = ang
if np.isnan(joint_rot_euler).any():
    raise ValueError('Angle can not be nan!')

# TODO 关节转角约束
hip_limit = [-0.30, 0.30]  # 0.8 in a1.urdf
thigh_limit = [-1.04, 4.18]
calf_limit = [-2.69, -0.91]
for i in range(joint_rot_euler.shape[0]):
    for j in range(joint_rot_euler.shape[1]):
        if j==0 or j==3 or j==6 or j==9:
            joint_rot_euler[i, j] = np.clip(joint_rot_euler[i, j], hip_limit[0], hip_limit[1])
        elif j==1 or j==4 or j==7 or j==10:
            joint_rot_euler[i, j] = np.clip(joint_rot_euler[i, j], thigh_limit[0], thigh_limit[1])
        elif j==2 or j==5 or j==8 or j==11:
            joint_rot_euler[i, j] = np.clip(joint_rot_euler[i, j], calf_limit[0], calf_limit[1])

# local_rotation = source_motion.local_rotation
# skeleton_state = SkeletonState.from_rotation_and_root_translation(source_motion.skeleton_tree, local_rotation,
#                                                                   root_translation, is_local=True)
# source_motion = SkeletonMotion.from_skeleton_state(skeleton_state, fps=source_motion.fps)
# plot_skeleton_motion_interactive(source_motion)

# ['trunk', 'FR_hip', 'FR_thigh', 'FR_calf', 'FR_foot', 'FL_hip', 'FL_thigh', 'FL_calf', 'FL_foot', 'RR_hip',
# 'RR_thigh', 'RR_calf', 'RR_foot', 'RL_hip', 'RL_thigh', 'RL_calf', 'RL_foot']
# target motion
g_root_rot_euler = torch.from_numpy(g_root_rot_euler)
g_root_rot_quat = quat_from_euler_xyz(g_root_rot_euler[:, 0], g_root_rot_euler[:, 1], g_root_rot_euler[:, 2])  # (len. 4)

joint_rot_euler = torch.from_numpy(joint_rot_euler)  # (190, 12)
skeleton_target = target_tpose.skeleton_tree
local_rotation = torch.zeros((joint_rot_euler.shape[0], 17, 4))  # (len, 17, 4)

local_rotation[:, :, -1] = 1
for i in range(joint_rot_euler.shape[0]):
    joint_rot_quat = []
    for j in range(joint_rot_euler.shape[1]):
        if j == 0 or j == 3 or j == 6 or j == 9:
            # joint_quat = quat_from_euler_xyz(joint_rot_euler[i, j], torch.tensor(0), torch.tensor(0))
            joint_quat = kinematic.rpy2quaternion([joint_rot_euler[i, j], 0, 0])
        else:
            # joint_quat = quat_from_euler_xyz(torch.tensor(0), joint_rot_euler[i, j], torch.tensor(0))
            joint_quat = kinematic.rpy2quaternion([0, joint_rot_euler[i, j], 0])
        joint_quat = torch.from_numpy(np.array(joint_quat))
        joint_rot_quat.append(joint_quat)
    local_rotation[i, 0, :] = g_root_rot_quat[i, :]
    local_rotation[i, 1, :] = joint_rot_quat[0]
    local_rotation[i, 2, :] = joint_rot_quat[1]
    local_rotation[i, 3, :] = joint_rot_quat[2]
    local_rotation[i, 5, :] = joint_rot_quat[3]
    local_rotation[i, 6, :] = joint_rot_quat[4]
    local_rotation[i, 7, :] = joint_rot_quat[5]
    local_rotation[i, 9, :] = joint_rot_quat[6]
    local_rotation[i, 10, :] = joint_rot_quat[7]
    local_rotation[i, 11, :] = joint_rot_quat[8]
    local_rotation[i, 13, :] = joint_rot_quat[9]
    local_rotation[i, 14, :] = joint_rot_quat[10]
    local_rotation[i, 15, :] = joint_rot_quat[11]

# local_rotation = source_motion.local_rotation  # (len, 27, 4)
# 需要把root的translation变换到质心位置
root_new_local = np.array([body_length / 2, 0, 0, 1])
root_trans_w = np.array([list((kinematic.trans_matrix_ba(m, t) @ root_new_local)[:3])
                         for m, t in zip(g_root_trans, g_root_rot_euler)])

# 放置质心到原点
root_translation = torch.from_numpy(root_trans_w)
root_translation[:, 0] = root_translation[:, 0] - root_translation[0, 0]
root_translation[:, 1] = root_translation[:, 1] - root_translation[0, 1]
# aa = torch.min(global_trans[0, :, 2])
# root_translation[:, 2] = root_translation[:, 2] - torch.min(global_trans[0, :, 2])

# foot above the ground
root_height_offset = 0.02
root_translation += root_height_offset
skeleton_state = SkeletonState.from_rotation_and_root_translation(target_tpose.skeleton_tree, local_rotation,
                                                                  root_translation, is_local=True)
target_motion = SkeletonMotion.from_skeleton_state(skeleton_state, fps=source_motion.fps)
target_motion.to_file(root_dir + '{}.npy'.format(output_file))
target_motion = SkeletonMotion.from_file(root_dir + '{}.npy'.format(output_file))
plot_skeleton_motion_interactive(target_motion)
