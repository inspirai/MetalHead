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

import copy
from isaacgym.torch_utils import *
from poselib.core.rotation3d import *

from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion

import torch
"""
This scripts imports a MJCF XML file and converts the skeleton into a SkeletonTree format.
It then generates a zero rotation pose, and adjusts the pose into a T-Pose.
"""

# ['Hips', 'Spine', 'Spine1', 'Neck', 'Head', 'Head_End', 'LeftShoulder', 'LeftArm',
#  'LeftForeArm', 'LeftHand', 'LeftHand_End', 'RightShoulder', 'RightArm', 'RightForeArm',
#  'RightHand', 'RightHand_End', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftFoot_End',
#  'RightUpLeg', 'RightLeg', 'RightFoot', 'RightFoot_End', 'Tail', 'Tail1', 'Tail1_End']

# # import MJCF file
# xml_path = "../../../../assets/mjcf/amp_humanoid.xml"
# skeleton = SkeletonTree.from_mjcf(xml_path)

# import fbx file
# fbx_file = "data/amp_hardware_a1/D1_ex05_KAN02_001.fbx"

motion = SkeletonMotion.from_file("data/amp_hardware_a1/walk_around1.npy")

# # import fbx file - make sure to provide a valid joint name for root_joint
# motion = SkeletonMotion.from_fbx(
#     fbx_file_path=fbx_file,
#     root_joint="Hips",  # Hips
#     fps=60
# )
skeleton = motion.skeleton_tree

local_translation = skeleton.local_translation  # (15, 4)
local_translation[skeleton.node_names.index("LeftShoulder")] += torch.tensor([0, 0, 2])
local_translation = skeleton.local_translation  # (15, 4)
local_translation[skeleton.node_names.index("RightShoulder")] -= torch.tensor([0, 0, 2])

local_translation = skeleton.local_translation  # (15, 4)
local_translation[skeleton.node_names.index("LeftUpLeg")] += torch.tensor([0, 0, 2])
local_translation = skeleton.local_translation  # (15, 4)
local_translation[skeleton.node_names.index("RightUpLeg")] -= torch.tensor([0, 0, 2])

# generate zero rotation pose
zero_pose = SkeletonState.zero_pose(skeleton)

# plot_skeleton_state(zero_pose)
# 关于四元数: https://www.qiujiawei.com/understanding-quaternions/
# adjust pose into a T Pose

local_rotation = zero_pose.local_rotation  # (15, 4)
local_rotation[skeleton.node_names.index("Spine")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([180.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True),
    local_rotation[skeleton.node_names.index("Spine")]
)

local_rotation[skeleton.node_names.index("LeftFoot")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True),
    local_rotation[skeleton.node_names.index("LeftFoot")]
)

local_rotation[skeleton.node_names.index("RightFoot")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True),
    local_rotation[skeleton.node_names.index("RightFoot")]
)

local_rotation[skeleton.node_names.index("LeftUpLeg")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([-90.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True),
    local_rotation[skeleton.node_names.index("LeftUpLeg")]
)

local_rotation[skeleton.node_names.index("RightUpLeg")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([-90.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True),
    local_rotation[skeleton.node_names.index("RightUpLeg")]
)

local_rotation[skeleton.node_names.index("LeftUpLeg")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([-20.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True),
    local_rotation[skeleton.node_names.index("LeftUpLeg")]
)

local_rotation[skeleton.node_names.index("RightUpLeg")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([-20.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True),
    local_rotation[skeleton.node_names.index("RightUpLeg")]
)

local_rotation[skeleton.node_names.index("LeftUpLeg")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([-5.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True),
    local_rotation[skeleton.node_names.index("LeftUpLeg")]
)

local_rotation[skeleton.node_names.index("RightUpLeg")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([5.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True),
    local_rotation[skeleton.node_names.index("RightUpLeg")]
)

local_rotation[skeleton.node_names.index("LeftLeg")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([40.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True),
    local_rotation[skeleton.node_names.index("LeftLeg")]
)

local_rotation[skeleton.node_names.index("RightLeg")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([40.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True),
    local_rotation[skeleton.node_names.index("RightLeg")]
)

local_rotation[skeleton.node_names.index("LeftShoulder")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True),
    local_rotation[skeleton.node_names.index("LeftShoulder")]
)

local_rotation[skeleton.node_names.index("RightShoulder")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True),
    local_rotation[skeleton.node_names.index("RightShoulder")]
)

local_rotation[skeleton.node_names.index("LeftShoulder")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([-45.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True),
    local_rotation[skeleton.node_names.index("LeftShoulder")]
)

local_rotation[skeleton.node_names.index("RightShoulder")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([-45.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True),
    local_rotation[skeleton.node_names.index("RightShoulder")]
)

local_rotation[skeleton.node_names.index("LeftShoulder")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([5.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True),
    local_rotation[skeleton.node_names.index("LeftShoulder")]
)

local_rotation[skeleton.node_names.index("RightShoulder")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([-5.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True),
    local_rotation[skeleton.node_names.index("RightShoulder")]
)

local_rotation[skeleton.node_names.index("LeftArm")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([60.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True),
    local_rotation[skeleton.node_names.index("LeftArm")]
)

local_rotation[skeleton.node_names.index("RightArm")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([60.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True),
    local_rotation[skeleton.node_names.index("RightArm")]
)

local_rotation[skeleton.node_names.index("LeftForeArm")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([-40.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True),
    local_rotation[skeleton.node_names.index("LeftForeArm")]
)

local_rotation[skeleton.node_names.index("RightForeArm")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([-40.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True),
    local_rotation[skeleton.node_names.index("RightForeArm")]
)

local_rotation[skeleton.node_names.index("Hips")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True),
    local_rotation[skeleton.node_names.index("Hips")]
)

local_rotation[skeleton.node_names.index("Hips")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([180.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True),
    local_rotation[skeleton.node_names.index("Hips")]
)

translation = zero_pose.root_translation
global_trans = zero_pose.global_translation
translation[2] -= torch.min(global_trans[:, 2])

# save and visualize T-pose
zero_pose.to_file("data/dog_tpose.npy")

zero_pose = SkeletonState.from_file("data/dog_tpose.npy")  # 大概高49.6
plot_skeleton_state(zero_pose)
