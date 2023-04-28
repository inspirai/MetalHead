## poselib

该`poselib` 是基于IssacGymEnvs中AMP Humanoid motion retargeting代码进行二次开发，完成了基于IK和Key points的a1 motion retargeting

## 原始动捕数据
使用AI4Animation提供的动捕数据: https://github.com/sebastianstarke/AI4Animation/tree/master/AI4Animation/SIGGRAPH_2018, 原始动捕数据为bvh格式，需要在windows下安装motion builder，然后将其导出为fbx格式。另外需要注意，AI4Animation中的数据除了第一帧外，其余帧均重复一次。

## a1 motion retargeting 过程
- 准备好fbx动捕文件放在`./data/amp_hardware_a1/fbx文件名/`目录下
- 在`load_config.json`中修改`file_name`(fbx文件名), `clip`(裁剪片段区间), `remarks`(输出文件名)
- 运行`fbx_importer.py`读取fbx文件, 生成npy文件, 去除相邻重复帧
- 运行`key_points_retargetting.py`生成retargeting后的npy文件
- 运行`json_exporter.py`导出训练所使用的json文件

## poselib目录下文件解释
- `fbx_importer.py` 导入fbx文件
- `key_points_retargetting.py` motion retargeting
- `json_exporter.py` 导出训练使用的json文件
- `json_loader.py` 导入某个json文件并可视化
- `ai4animation_dog_tpose.py` 生成source T-pose
- `generate_amp_a1_tpose.py` 生成target T-pose
- `kinematics.py` 运动学解算

## data目录下文件解释
- `load_config.json` 用于指定目标retargeting文件名以及导出文件名
- `amp_a1_tpose.npy` target T-pose
- `dog_tpose.npy` source T-pose
- `AI4Animation_fbx` 原始的fbx文件存放目录
- `amp_hardware_a1` motion retargeting读取以及导出的文件目录
- `amp_hardware_a1_org` 原始AMP for Hardware使用的动捕数据文件目录
- `mocap_motions`最终导出用于训练的的mocap片段文件目录

## visualization
- `poselib.visualization.common`: Functions used for visualizing skeletons interactively in `matplotlib`.
    - In SkeletonState visualization, use key `q` to quit window.
    - In interactive SkeletonMotion visualization, you can use the following key commands:
        - `w` - loop animation
        - `x` - play/pause animation
        - `z` - previous frame
        - `c` - next frame
        - `n` - quit window
