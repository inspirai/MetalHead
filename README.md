
[![Watch the video](https://img.youtube.com/vi/IdzfE9rXoqY/maxresdefault.jpg)](https://youtu.be/IdzfE9rXoqY)
Click above image to watch the Video!

# MetalHead: Natural Locomotion, Jumping and Recovery of Quadruped Robot A1 with AMP

The majority of research on quadruped robots has yet to accomplish complete natural movements such as walking, running, jumping, and recovering from falls. This project MetalHead, which utilizes the AMP algorithm and meticulous engineering to successfully achieve these objectives.

This repository is based off of Alejandro Escontrela's [AMP_for_hardware](https://github.com/Alescontrela/AMP_for_hardware). All experiments are running with A1 robot from Unitree on [Isaac Gym](https://developer.nvidia.com/isaac-gym).



## Installation

Please just following [AMP_for_hardware](https://github.com/Alescontrela/AMP_for_hardware)'s installation instruction.


## How to Run
```
python legged_gym/scripts/play.py --task=a1_amp_jump_cmd --num_envs=64 --load_run=example
```
**Also, you can play with it!**
```
python legged_gym/scripts/play_v2.py --task=a1_amp_jump_cmd_play --num_envs=1 --load_run=example
```
Press w,a,s,d to change A1 speed for different gaits of walking and running, and press Space to make it jump! Have fun of it!

## How to Train
```
python legged_gym/scripts/train.py --task=a1_amp_jump_cmd --headless True
```
25000 iterations of training can be enough to show the performance.

## Changes compared to AMP_for_Hardware

Please see wikis https://github.com/inspirai/MetalHead/wiki for detail.(You can try to translate them into English with ChatGPT for convenient reading.)

You can also check this [ppt](https://docs.google.com/presentation/d/16BtMnja4JNx41ni6s1VpcMU_YQhpsE5X/edit?usp=sharing&ouid=100234233253970958121&rtpof=true&sd=true)

### Observation additions
EN:
- Added root_h, root_euler[:, :2], flat_local_key_pos to obs, representing the absolute height of the root, the rotation angle of the root, and the relative coordinates of the four foot ends in the root coordinate system, respectively
- Added jump_sig to obs, indicating whether the jump command is triggered
- policy_obs and critic_obs are consistent
- amp_policy_obs removes commands and jump_sig from the policy_obs

CH:
- obs增加`root_h`, `root_euler[:, :2]`, `flat_local_key_pos`, 分别表示root的绝对高度, root的转角以及四个足端在root坐标系下的相对坐标
- obs增加`jump_sig`, 表示是否触发jump command
- policy_obs和critic_obs一致
- amp_policy_obs在policy_obs基础上去掉`commands`以及`jump_sig`

### Action changes
EH:
- Changed action to position PD control, using `set_dof_position_target_tensor` API
- Policy inference frequency is 200 / 6 Hz, with a physical simulation frequency of 200 Hz and action repetition count of 6
CH:
- action改为位置PD控制，使用`set_dof_position_target_tensor` API
- policy inference频率200 / 6 Hz, 其中物理仿真200Hz, action重复次数为6次

### Reward additions
EN:
- Added _reward_jump_up for calculating task rewards
CH:
- 新增`_reward_jump_up`, 计算task奖励


### Random initialization
EN:
- recovery_init_prob = 0.15, with a 15% probability of random initialization, added _reset_root_states_rec function for random sampling in three Euler angle directions

CH:
- `recovery_init_prob = 0.15`, 以15%的概率随机初始化，新增`_reset_root_states_rec`函数，实现三个欧拉角方向上的随机采样

### Mocap data
EN:
- For command-based locomotion+jump, motion capture data includes gallop_forward0, gallop_forward1, jump0, jump1, jump2, trot_forward0, turn_left0, turn_left1, turn_right0, turn_right1, where trot is the same side two legs
- In the JSON file, the weight of the jump data is set to 1.5, and the others are 0.5

CH:
- 对于command-based locomotion+jump, 动捕数据有gallop_forward0, gallop_forward1, jump0, jump1, jump2, trot_forward0, turn_left0, turn_left1, turn_right0, turn_right1，其中，tort是同侧两条腿
- json文件中，jump数据的weight设置为1.5, 其余0.5

### Play camera tracking
EN:
- In MOVE_CAMERA mode, the camera follows the robot's root at a fixed angle, and the camera's position and yaw angle relative to the robot remain unchanged

CH:
- 在`MOVE_CAMERA`模式下, 摄像头以固定视角跟随机器人的root, 摄像头相对于机器人的position以及yaw角不变


### Some key parameters
EN:
- action_scale=0.75, too large or too small cannot achieve command jump
- all_stiffness = 80.0, all_damping=1.0, good PD parameters can facilitate simulation training, and more importantly, have a greater impact on the difficulty of sim2real transfer
- amp_task_reward_lerp = 0.3, controls the weight of task reward and style reward
- disc_grad_penalty = 0.01, smaller penalty is needed for high-dynamic mocap
- resampling_time = 2., episode_length_s=10., command sampling interval and episode length, in recovery_init mode, sampling interval has a greater impact on jump effect
- tracking_ang_vel = 0.1 * 1. / (.005 * 6), too small weight cannot follow angular velocity properly, maybe try heading tracking, which is more convenient in sim
- In random initialization mode, terminate_after_contacts_on is set to empty

CH:
- `action_scale=0.75`, 太大或者太小无法实现command jump
- `all_stiffness = 80.0`, `all_damping=1.0`, 一个好的PD参数能够方便仿真训练，更重要的是对sim2real的迁移难度影响较大
- `amp_task_reward_lerp = 0.3`, 控制task reward和style reward的权重
- `disc_grad_penalty = 0.01`, 在高动态的mocap需要较小的penalty
- `resampling_time = 2.`, `episode_length_s=10.`, command采样间隔以及回合长度, 在recovery_init模式下, 采样间隔对jump效果影响较大
- `tracking_ang_vel = 0.1 * 1. / (.005 * 6)`, 权重太小无法正常跟随角速度，或许可以尝试heading跟随，在sim中比较方便
- 在随机初始化模式下，`terminate_after_contacts_on`设置为空


## Highly Related Git
- [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- [legged_gym](https://github.com/leggedrobotics/legged_gym)
- [AMP_for_hardware](https://github.com/Alescontrela/AMP_for_hardware)


## Contact
If you have any questions or have a will to coorperate, please [contact](mailto:hqfu@smail.nju.edu.cn)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=inspirai/MetalHead&type=Date)](https://star-history.com/#inspirai/MetalHead&Date)


## Citing

If you use MetalHead in your research please use the following citation:

````
@misc{InspirAI,
  author = {Huiqiao Fu, Yutong Wu, Flood Sung},
  title = {MetalHead: Natural Locomotion, Jumping and Recovery of Quadruped Robot A1 with AMP},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/inspirai/MetalHead}},
}

