# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np

from isaacgym.torch_utils import *
from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from poselib.visualization.common import plot_skeleton_state
import torch
"""
This scripts imports a MJCF XML file and converts the skeleton into a SkeletonTree format.
It then generates a zero rotation pose, and adjusts the pose into a T-Pose.
"""

# ['trunk', 'FR_hip', 'FR_thigh', 'FR_calf', 'FR_foot', 'FL_hip', 'FL_thigh', 'FL_calf', 'FL_foot', 'RR_hip', 'RR_thigh',
#  'RR_calf', 'RR_foot', 'RL_hip', 'RL_thigh', 'RL_calf', 'RL_foot']

# import MJCF file
xml_path = "../../../../assets/mjcf/a1.xml"
skeleton = SkeletonTree.from_mjcf(xml_path)

# generate zero rotation pose
zero_pose = SkeletonState.zero_pose(skeleton)

# plot_skeleton_state(zero_pose)
# 关于四元数: https://www.qiujiawei.com/understanding-quaternions/
# adjust pose into a T Pose
local_rotation = zero_pose.local_rotation  # (15, 4)
# local_rotation[skeleton.node_names.index("FR_hip")] = quat_mul(
#     quat_from_angle_axis(angle=torch.tensor([20]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True),
#     local_rotation[skeleton.node_names.index("FR_hip")]
# )

joint_offset = 20
local_rotation[skeleton.node_names.index("FR_thigh")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([joint_offset]), axis=torch.tensor([0.0, 1.0, 0.0]), degree=True),
    local_rotation[skeleton.node_names.index("FR_thigh")]
)
local_rotation[skeleton.node_names.index("FR_calf")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([-joint_offset*2]), axis=torch.tensor([0.0, 1.0, 0.0]), degree=True),
    local_rotation[skeleton.node_names.index("FR_calf")]
)

local_rotation[skeleton.node_names.index("FL_thigh")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([joint_offset]), axis=torch.tensor([0.0, 1.0, 0.0]), degree=True),
    local_rotation[skeleton.node_names.index("FL_thigh")]
)
local_rotation[skeleton.node_names.index("FL_calf")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([-joint_offset*2]), axis=torch.tensor([0.0, 1.0, 0.0]), degree=True),
    local_rotation[skeleton.node_names.index("FL_calf")]
)

local_rotation[skeleton.node_names.index("RR_thigh")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([joint_offset]), axis=torch.tensor([0.0, 1.0, 0.0]), degree=True),
    local_rotation[skeleton.node_names.index("RR_thigh")]
)
local_rotation[skeleton.node_names.index("RR_calf")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([-joint_offset*2]), axis=torch.tensor([0.0, 1.0, 0.0]), degree=True),
    local_rotation[skeleton.node_names.index("RR_calf")]
)

local_rotation[skeleton.node_names.index("RL_thigh")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([joint_offset]), axis=torch.tensor([0.0, 1.0, 0.0]), degree=True),
    local_rotation[skeleton.node_names.index("RL_thigh")]
)
local_rotation[skeleton.node_names.index("RL_calf")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([-joint_offset*2]), axis=torch.tensor([0.0, 1.0, 0.0]), degree=True),
    local_rotation[skeleton.node_names.index("RL_calf")]
)

translation = zero_pose.root_translation
global_trans = zero_pose.global_translation
translation[2] -= torch.min(global_trans[:, 2])

# save and visualize T-pose
zero_pose.to_file("data/amp_a1_tpose.npy")

target_tpose = SkeletonState.from_file("data/amp_a1_tpose.npy")  # 大概高0.3864
plot_skeleton_state(target_tpose)
