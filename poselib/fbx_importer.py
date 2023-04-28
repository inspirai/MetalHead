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

import re
import os
import json
import copy
from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive

# ['Hips', 'Spine', 'Spine1', 'Neck', 'Head', 'Head_End', 'LeftShoulder', 'LeftArm',
#  'LeftForeArm', 'LeftHand', 'LeftHand_End', 'RightShoulder', 'RightArm', 'RightForeArm',
#  'RightHand', 'RightHand_End', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftFoot_End',
#  'RightUpLeg', 'RightLeg', 'RightFoot', 'RightFoot_End', 'Tail', 'Tail1', 'Tail1_End']

# source fbx file path
json_dir = "data/load_config.json"
with open(json_dir, "r") as f:
    motion_json = json.load(f)
    fbx_file_name = motion_json["file_name"]

root_dir = "data/amp_hardware_a1/{}/".format(fbx_file_name)
fbx_file = root_dir + "{}.fbx".format(fbx_file_name)

# import fbx file - make sure to provide a valid joint name for root_joint
motion = SkeletonMotion.from_fbx(
    fbx_file_path=fbx_file,
    fps=60
)

skeleton = motion.skeleton_tree

# 规范node name格式
for i in range(len(skeleton.node_names)):
    rgx = r'^.+:(.+)$'
    skeleton.node_names[i] = re.search(rgx, skeleton.node_names[i]).group(1)

local_rotation = motion.local_rotation
for i in range(local_rotation.shape[0]):
    local_rotation[i, skeleton.node_names.index("Hips"), :] = quat_mul(
        quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True),
        local_rotation[i, skeleton.node_names.index("Hips"), :]
    )

local_rotation = motion.local_rotation
for i in range(local_rotation.shape[0]):
    local_rotation[i, skeleton.node_names.index("Hips"), :] = quat_mul(
        quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True),
        local_rotation[i, skeleton.node_names.index("Hips"), :]
    )

root_translation = motion.root_translation
root_translation_copy = copy.deepcopy(root_translation[:, 2])
root_translation[:, 2] = root_translation[:, 0]
root_translation[:, 0] = root_translation_copy

root_translation = motion.root_translation
root_translation_copy = copy.deepcopy(root_translation[:, 2])
root_translation[:, 2] = root_translation[:, 1]
root_translation[:, 1] = root_translation_copy

root_translation[:root_translation.shape[0], 0] -= copy.deepcopy(root_translation[0, 0])
root_translation[:root_translation.shape[0], 1] -= copy.deepcopy(root_translation[0, 1])
global_trans = motion.global_translation
root_translation[:, 2] -= global_trans[0, skeleton.node_names.index("RightFoot_End"), 2]

# TODO: AI4Animation中的数据除了第一帧，其余相邻两帧相同!
repeat_num = 2
local_rotation_out = []  # (len, 27, 4)
root_translation_out = []  # (len, 3)
for i in range(local_rotation.shape[0]):
    if i != 0 and i % repeat_num == 0:
        continue
    local_rotation_out.append(local_rotation[i, ...])
    root_translation_out.append(root_translation[i, ...])
local_rotation_out = torch.stack(local_rotation_out)
root_translation_out = torch.stack(root_translation_out)

new_sk_state = SkeletonState.from_rotation_and_root_translation(motion.skeleton_tree, local_rotation_out,
                                                                root_translation_out, is_local=True)
target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=motion.fps/repeat_num)

# save motion in npy format
target_motion.to_file(root_dir + "{}.npy".format(fbx_file_name))
target_motion = SkeletonMotion.from_file(root_dir + "{}.npy".format(fbx_file_name))
# visualize motion
plot_skeleton_motion_interactive(target_motion)
