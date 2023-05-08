
# MetalHead: Natural Locomotion and Jumping of Quadruped Robot A1 with AMP

The core contribution of this repository is to make the quadruped robot A1 being able to do natural locomotion and jumping like the video below. This repository is based off of Alejandro Escontrela's [AMP_for_hardware](https://github.com/Alescontrela/AMP_for_hardware). All experiments are running on [Isaac Gym](https://developer.nvidia.com/isaac-gym).

[![Watch the video](https://img.youtube.com/vi/IdzfE9rXoqY/maxresdefault.jpg)](https://youtu.be/IdzfE9rXoqY)
Click above image to watch the Video!

### Installation ###

Please just following [AMP_for_hardware](https://github.com/Alescontrela/AMP_for_hardware)'s installation instruction.


## How to Run
```
python legged_gym/scripts/play.py --task=a1_amp_jump_cmd --num_envs=64 --load_run=example
```
Press w,a,s,d to change the speed, and press Space to make it jump!

## How to Train
```
python legged_gym/scripts/train.py --task=a1_amp_jump_cmd --headless True
```
25000 iterations of training can be enough to show the performance.

## observation 新增
- obs增加`root_h`, `root_euler[:, :2]`, `flat_local_key_pos`, 分别表示root的绝对高度, root的转角以及四个足端在root坐标系下的相对坐标
- obs增加`jump_sig`, 表示是否触发jump command
- policy_obs和critic_obs一致
- amp_policy_obs在policy_obs基础上去掉`commands`以及`jump_sig`

## action 改动
- action改为位置PD控制，使用`set_dof_position_target_tensor` API
- policy inference频率200 / 6 Hz, 其中物理仿真200Hz, action重复次数为6次

## reward 新增
- 新增`_reward_jump_up`, 计算task奖励

## 随机初始化
- `recovery_init_prob = 0.15`, 以15%的概率随机初始化，新增`_reset_root_states_rec`函数，实现三个欧拉角方向上的随机采样

## mocap数据
- 对于command-based locomotion+jump, 动捕数据有gallop_forward0, gallop_forward1, jump0, jump1, jump2, trot_forward0, turn_left0, turn_left1, turn_right0, turn_right1，其中，tort是同侧两条腿
- json文件中，jump数据的weight设置为1.5, 其余0.5

## play 视角跟随
- 在`MOVE_CAMERA`模式下, 摄像头以固定视角跟随机器人的root, 摄像头相对于机器人的position以及yaw角不变

## 一些关键参数
- `action_scale=0.75`, 太大或者太小无法实现command jump
- `all_stiffness = 80.0`, `all_damping=1.0`, 一个好的PD参数能够方便仿真训练，更重要的是对sim2real的迁移难度影响较大
- `amp_task_reward_lerp = 0.3`, 控制task reward和style reward的权重
- `disc_grad_penalty = 0.01`, 在高动态的mocap需要较小的penalty
- `resampling_time = 2.`, `episode_length_s=10.`, command采样间隔以及回合长度, 在recovery_init模式下, 采样间隔对jump效果影响较大
- `tracking_ang_vel = 0.1 * 1. / (.005 * 6)`, 权重太小无法正常跟随角速度，或许可以尝试heading跟随，在sim中比较方便
- 在随机初始化模式下，`terminate_after_contacts_on`设置为空

## 历史版本
- `amp_jump_cmd_v3.6`: `amp_task_reward_lerp`从0.3逐步降低到0.2，实现比较好的locomotion+jump
- `amp_jump_cmd_v3.4`: 导入a1 motion retargeting相关代码: `poselib`, 暂时取消recovery
- `amp_jump_cmd_v3.1`: 利用AMP以及动捕数据实现a1 jump+command+recovery, jump姿态不够理想
- `amp_jump_cmd_v1.8`: 利用AMP以及动捕数据实现a1 jump+command
- `amp_jump_v2.1`: 利用AMP以及动捕数据实现a1 jump
- `amp_base_version`: 利用AMP以及动捕数据实现a1 locomotion

## Highly Related Git
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- [legged_gym](https://github.com/leggedrobotics/legged_gym)
- [AMP_for_hardware](https://github.com/Alescontrela/AMP_for_hardware)