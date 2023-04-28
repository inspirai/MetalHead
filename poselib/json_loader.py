#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File: json_loader.py
@Auth: Huiqiao
@Date: 2022/7/13
"""
import json
import numpy as np
from isaacgym.torch_utils import *
from kinematics import Kinematics
from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
from poselib.skeleton.backend.fbx.fbx_read_wrapper import fbx_to_array
import torch
from kinematics import Kinematics

body_length = 0.366
body_wide = 0.094
hip_length = 0.08505
thigh_length = 0.2
calf_length = 0.2
kinematic = Kinematics(body_length=0.366, body_wide=0.094, hip_length=0.08505, thigh_length=0.2, calf_length=0.2)

a1_tpose = SkeletonState.from_file('data/amp_a1_tpose.npy')
skeleton = a1_tpose.skeleton_tree

# data_target = 'data/mocap_motions/gallop_jump0.json'
# data_target = 'data/amp_hardware_a1/walk_jump1/walk_jump1_amp_291_324_.json'
# data_target = 'data/amp_hardware_a1_org/pace0.txt'
# data_target = '/home/trna/Documents/AMP/AMP_git/amp_for_hardware/datasets/mocap_motions_jump_cmd/gallop_jump0_amp_242_273_jump_forward.json'
data_target = 'data/mocap_motions/trot_forward0.json'

# root_pos, root_rot, joint_pos, foot_pos, lin_vel, ang_vel, joint_vel, foot_vel  order: [FR, FL, RR, RL]
with open(data_target, "r") as f:
    motion_json = json.load(f)
    motion_s = np.array(motion_json['Frames'])  # (len, 61)
root_pos = motion_s[:, :3]  # 3
root_rot = motion_s[:, 3:7]  # 4
joint_pos = motion_s[:, 7:19]  # 12
foot_pos = motion_s[:, 19:31]  # 12
lin_vel = motion_s[:, 31:34]  # 3
ang_vel = motion_s[:, 34:37]  # 3
joint_vel = motion_s[:, 37:49]  # 12
foot_vel = motion_s[:, 49:]  # 12

# lin_vel_x_min = min(lin_vel[:, 0])
# lin_vel_x_max = max(lin_vel[:, 0])
# lin_vel_y_min = min(lin_vel[:, 1])
# lin_vel_y_max = max(lin_vel[:, 1])

local_rotation = torch.zeros((joint_pos.shape[0], 17, 4))  # (len, 17, 4)
local_rotation[:, :, -1] = 1
for i in range(joint_pos.shape[0]):
    joint_rot_quat = []
    for j in range(joint_pos.shape[1]):
        if j == 0 or j == 3 or j == 6 or j == 9:
            joint_quat = kinematic.rpy2quaternion([joint_pos[i, j], 0, 0])
        else:
            joint_quat = kinematic.rpy2quaternion([0, joint_pos[i, j], 0])
        joint_quat = torch.from_numpy(np.array(joint_quat))
        joint_rot_quat.append(joint_quat)
    local_rotation[i, 0, :] = torch.tensor(root_rot[i, :])
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

root_translation = torch.from_numpy(root_pos)
skeleton_state = SkeletonState.from_rotation_and_root_translation(skeleton, local_rotation,
                                                                  root_translation, is_local=True)
target_motion = SkeletonMotion.from_skeleton_state(skeleton_state, fps=60)
plot_skeleton_motion_interactive(target_motion)









