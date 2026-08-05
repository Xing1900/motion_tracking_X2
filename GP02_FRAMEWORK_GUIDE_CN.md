# MotionTracking 的 GP02 接入与代码说明

本文对应当前仓库中的 GP02 V3 接入，覆盖机器人资产、控制参数、动作数据、质量过滤、训练任务、奖励与终止条件，以及训练前检查。

## 1. 整体数据与训练链路

```text
AMASS / LaFAN 人体动作
        |
        v
GMR 将人体动作重定向到 GP02 24 关节
        |
        v
GMR PKL -> MotionTracking NPZ（统一到 50 Hz）
        |
        v
动作预处理与质量过滤
        |
        v
float16 memmap 数据集
        |
        v
GP02 MuJoCo 模型做前向运动学
        |
        v
MotionTracking 强化学习环境和策略训练
```

最终训练不是直接读取 AMASS 的人体参数，也不是在每一步读取大量 NPZ。训练读取的是预生成的内存映射数据；根位置、根姿态和 24 个关节角从磁盘按需读取，其余关键点位置和速度通过 GP02 模型的前向运动学计算。

## 2. GP02 机器人资产

### 2.1 文件位置

- `active_adaptation/assets/GP02/gp02_v3.xml`：MuJoCo 机器人模型。
- `active_adaptation/assets/GP02/meshes/`：模型引用的 25 个 STL 网格。
- `active_adaptation/assets/GP02/humanoid.py`：MotionTracking/MJLab 使用的机器人定义。
- `active_adaptation/assets/__init__.py`：把 `gp02_v3` 注册到全局机器人表。

当前训练资产复制自 `gmr/assets/gp02_v3/`。这是刻意选择：动作数据就是用该模型重定向出来的，所以训练中的前向运动学必须使用同一版本。UFO 中的 `assets/GP02.xml` 与它的哈希不同，不能在没有重新核对运动学和惯量的情况下直接替换。

### 2.2 自由度与关节顺序

当前 GP02 模型是 24 自由度：

- 双腿：12 个关节。
- 腰部：`waist_yaw_joint`、`waist_roll_joint`，共 2 个关节。
- 双臂：每侧肩 3、肘 1、腕偏航 1，共 10 个关节。

严格关节顺序定义在 `GP02_JOINT_ORDER`。它同时约束：

- MuJoCo 模型的关节顺序；
- 数据集 `joint_names`；
- 完整机器人状态、数据集和运动学计算的顺序；
- 左右对称数据增强的映射。

训练模型和动作数据仍保持完整 24 关节，但策略不必控制全部关节。当前已经采用“24 关节参考、22 关节控制”：腰部两个关节保留在模型和数据集中，但从策略动作空间中排除。

### 2.3 初始姿态和 PD 参数

稳定站立初始关节角全部为 0，根高度来自 MJCF，为 0.7328504 m。

当前仿真 PD 设置如下：

| 关节组 | KP | KD | 力矩上限 |
|---|---:|---:|---:|
| hip pitch | 100 | 4 | 139 Nm |
| hip roll | 100 | 3 | 88 Nm |
| hip yaw | 100 | 3 | 88 Nm |
| knee | 150 | 5 | 139 Nm |
| ankle pitch | 40 | 3 | 80 Nm |
| ankle roll | 30 | 2 | 80 Nm |
| waist yaw | 40 | 5 | 88 Nm |
| waist roll | 40 | 5 | 50 Nm |
| shoulder/elbow/wrist | 40 | 5 | 25 Nm |

腿和腰采用用户给出的稳定站立参数。手臂的 40/5 来自本机已有 GP02 控制配置，是保守训练初值，不代表已经完成实机辨识；部署前仍需核对电机减速比、驱动器力矩限制和实机 PD 接口。

### 2.4 左右对称映射

`JOINT_SYMMETRY_MAP` 描述关节镜像后的名称和符号，例如左右 hip pitch 互换且符号不变，hip roll 互换且符号翻转。`SPATIAL_SYMMETRY_MAP` 对刚体做同样映射。

它们供策略的对称损失、观测归一化和数据增强使用。预检要求所有 24 个关节都必须被关节对称表覆盖。

## 3. GP02 训练任务配置

### 3.1 文件位置

- `cfg/task/GP02/GP02.yaml`：机器人质量、环境数、控制周期和仿真周期等基础设置。
- `cfg/task/GP02/GP02_tracking.yaml`：完整动作跟踪任务。
- `cfg/exp/train.yaml`：第一阶段 teacher/base policy 训练。
- `cfg/exp/adapt.yaml`：第二阶段适配训练。
- `cfg/exp/finetune.yaml`：第三阶段 student 策略微调。

`GP02_tracking.yaml` 最初以 X2 中与机器人型号无关的全身跟踪配方为模板，但文件本身是自包含的，并没有在 Hydra 中继承 X2。GP02 专属的机器人、关键点、数据集、动作缩放、脚部奖励和终止条件都明确写在该文件中，因此不会因为修改 X2 配置而意外改变 GP02，也不会引用 X2 专属的头部或脚趾刚体。

### 3.2 基础仿真设置

- 并行环境数：2048。
- 策略控制周期：0.02 s，即 50 Hz。
- MuJoCo 物理周期：0.005 s，即每个策略动作执行 4 个物理子步。
- 最大 episode：1000 个策略步，约 20 s。

显存不足时可在命令行覆盖，例如 `task.num_envs=512`。环境数只影响并行采样速度和显存，不改变数据集内容。

### 3.3 动作输出

动作类型是关节位置目标，由 `active_adaptation/envs/mdp/action.py` 中的 `JointPosition` 处理。策略输出经过：

1. 各关节组的 action scaling；
2. 动作平滑系数 `alpha`；
3. 最多 2 步的随机控制延迟；
4. boot protection；
5. PD 执行器转换为关节力矩。

考虑到 GP02 腰部当前不成熟，策略动作空间排除了 `waist_yaw_joint` 和 `waist_roll_joint`，因此动作维度从 24 降为 22。位置控制器每个物理步仍向这两个腰关节发送默认目标 0 rad。这种方式保留了模型惯量、碰撞、完整数据集和 24 关节前向运动学，不需要重新生成 AMASS/LaFAN。

腰部同时从以下训练环节排除：

- 参考动作初始化，不会用数据集腰角初始化；
- 初始化关节角/速度噪声；
- 目标关节位置观测；
- 关节位置和速度跟踪奖励；
- 关节零位随机偏置；
- 策略动作输出。

身体关键点奖励仍包含 torso，因为策略需要在腰部固定的约束下，用腿和根部姿态尽量完成全身动作。这里的“锁定”是固定位置控制目标，而不是从 MJCF 删除关节；外力下仍可能出现由 PD 柔顺性导致的小幅偏移，这更接近实机位置控制器。

### 3.4 关键点跟踪

GP02 没有 X2/G1 的独立头部刚体，因此关键点使用：

- 上半身：torso、双肩 yaw、双腕 yaw；
- 下半身：双 hip yaw、双膝、双 ankle roll；
- 根部：pelvis。

配置在 `GP02_tracking.yaml` 的 `command.required_motion_body_patterns`、`keypoint_patterns`、`lower_keypoint_patterns` 和 `upper_keypoint_patterns`。

`active_adaptation/utils/fk_helper.py` 根据根位姿和 24 个关节角计算这些关键点的位置、姿态、线速度和角速度。

需要注意：GMR 旧的 `KinematicsModel` 没有解析 GP02 肘部 MJCF 的 `euler` 属性，所以 PKL/NPZ 中两只手腕的 `local_body_pos` 辅助字段存在偏差。最终 memmap 不保存该字段，训练时由上述 MotionTracking FK 根据正确的 MuJoCo 模型重新计算。预检会将 MotionTracking FK 与原生 MuJoCo 对照，当前最大位置误差约为 `1.5e-7 m`。

### 3.5 数据集混合

训练同时使用：

- `dataset/gp02_amass_all`，权重 0.9；
- `dataset/gp02_lafan_all`，权重 0.1。

这里的权重是“先选择哪个数据集”的概率，不是单个动作权重。LaFAN 数据量远小于 AMASS，设置 0.1 可以让走路、转身等动作被适度采样，又不会压过 AMASS 的动作多样性。

### 3.6 奖励

跟踪奖励主要包括：

- 根位置、根姿态、根线速度、根角速度；
- 关键点位置、姿态、线速度、角速度；
- 上半身和下半身关键点的独立奖励；
- 关节位置和关节速度跟踪。

运动质量奖励/惩罚包括：

- 存活奖励；
- 关节速度惩罚；
- 动作变化率惩罚；
- 参考脚腾空时间；
- 关节位置限制；
- 关节力矩限制。

GP02 的脚底碰撞球直接属于 ankle roll 刚体，没有独立 toe body，所以继承的“双刚体稠密脚部奖励”被禁用。普通脚腾空时间奖励仍然保留。

### 3.7 随机化

训练中会随机化：

- pelvis/torso 质心；
- 脚底摩擦和接触参数；
- 电机 KP、KD 和 armature；
- 关节零位偏差；
- 目标腿部关节偏差；
- 参考根位置漂移和高度偏差；
- 机器人根速度扰动；
- 重力扰动。

这些随机化用于提高仿真鲁棒性和 sim-to-real 能力。腰和手臂目标偏差当前为 0，避免本体尚未成熟时给上身加入过强随机目标。

### 3.8 终止条件

episode 主要在以下情况终止：

- pelvis、torso、手腕或脚踝高度相对参考动作偏差过大；
- 当前重力方向与参考根姿态差异过大；
- 动作正常播放完成。

失败后框架可以从此前动作的一小段位置重新初始化，而不是每次都从动作开头开始。

## 4. 数据转换和质量过滤

### 4.1 PKL 转 NPZ

代码：`scripts/data_process/gmr_pkl_to_motion_tracking_npz.py`

它把 GMR 输出转换为 MotionTracking 需要的字段，包括：

- 根位置和根四元数；
- 24 个关节角；
- 关节名和刚体名；
- 局部刚体位置；
- 50 Hz 帧率。

批量入口：`scripts/data_process/batch_convert_gmr_gp02_to_dataset.sh`。

### 4.2 NPZ 转训练 memmap

代码：`scripts/data_process/generate_dataset.py`

数据首先按每段最多 1000 帧切片，然后执行预处理：

- 首帧根部 XY 平移到原点；
- 所有刚体做同样的 XY 平移；
- 根据两只脚的最低点整体调整 Z，使动作接触地面。

通过过滤的内容以 float16 写入：

- `_tensordict/root_pos_w.memmap`；
- `_tensordict/root_quat_w.memmap`；
- `_tensordict/joint_pos.memmap`；
- `meta_motion.json`；
- `id_label.json`；
- `quality_report.json`。

### 4.3 所有数据都会执行的通用检查

- qpos、qvel、xpos 的维度和帧数必须一致；
- 所有数值必须是有限值，拒绝 NaN/Inf；
- 根四元数范数误差不能超过 0.1；
- 根线速度或角速度任一分量不能超过 10；
- 切片不能短于 250 帧，即 5 s；
- 所有刚体同时离地不能连续超过 1 s；
- 动作中的最高刚体高度必须超过 0.2 m。

### 4.4 `--amass-filter` 额外执行的检查

- 排除已知质量较差的动作族，例如部分 CMU、KIT 和形状轨迹动作；
- 读取 `scripts/data_process/label.txt` 中人工标注的坏片段；
- 以 `AMASS/` 后的相对路径加 `(segment_start, segment_end)` 精确匹配。

原代码使用旧机器的绝对路径，并忽略 start/end，导致 GP02 目录下的人工标签不能正确生效。现在已经改为机器无关的精确片段匹配，不会因为一个片段有问题而误删整个动作文件。

每次构建都会生成 `quality_report.json`，记录输入文件数、接受片段/帧数、各类拒绝原因和过滤阈值。

当前正式数据结果：

- AMASS：12,482 个训练片段，8,839,866 帧；
- LaFAN：518 个训练片段，498,580 帧；
- 两套数据都已完成逐值有限性扫描；
- AMASS 人工坏片段残留数为 0；
- 报告分别位于 `dataset/gp02_amass_all/quality_report.json` 和 `dataset/gp02_lafan_all/quality_report.json`。

切换前的数据没有删除，分别保存在 `dataset/gp02_amass_all.before_quality_fix_20260804` 和 `dataset/gp02_lafan_all.before_quality_report_20260804`。

## 5. 训练前检查

代码：`scripts/validate_gp02_setup.py`

普通检查：

```bash
.venv/bin/python scripts/validate_gp02_setup.py
```

扫描 memmap 中每一个数值：

```bash
.venv/bin/python scripts/validate_gp02_setup.py --full-data-scan
```

它会检查：

- MJCF 能否编译；
- 模型是否确实为 24 关节、24 执行器；
- 模型总质量是否与任务配置一致；
- 执行器和对称映射是否完整覆盖关节；
- 所有关键点/脚部/动作缩放正则表达式是否能匹配；
- 策略动作是否恰好为 22 维，且不包含两个锁定腰关节；
- AMASS 与 LaFAN 的关节名和顺序是否严格等于模型；
- 动作边界是否连续；
- 数据是否为空或包含 NaN/Inf；
- 质量报告是否与实际数据一致。

## 6. 训练命令

先检查配置：

```bash
cd /home/liuguoxing/Documents/motion_tracking
.venv/bin/python scripts/validate_gp02_setup.py --full-data-scan
```

第一阶段：

```bash
uv run torchrun --nproc_per_node=1 scripts/train.py \
  task=GP02/GP02_tracking +exp=train \
  wandb.project=gp02_motion_tracking
```

第二阶段需要把 `checkpoint_path` 指向第一阶段 checkpoint，并使用 `+exp=adapt`。第三阶段指向第二阶段 checkpoint，并使用 `+exp=finetune`。

调试时建议先用：

```bash
uv run torchrun --nproc_per_node=1 scripts/train.py \
  task=GP02/GP02_tracking +exp=train \
  task.num_envs=64 total_frames=1000000 \
  wandb.project=gp02_motion_tracking
```

确认环境创建、观测维度、奖励和 checkpoint 保存都正常后，再恢复正式环境数和总帧数。

### 6.1 GP02 第一阶段后台训练与自动恢复

第一阶段使用 `scripts/gp02_train_stage1.sh`，并由用户级 systemd 服务
`scripts/systemd/gp02_train_stage1.service` 托管。它只运行 `+exp=train`，不会自动进入
adapt 或 finetune。

服务启动时会递归查找 `outputs/gp02_stage1_train/` 中最新的 checkpoint。若找到，则恢复：

- teacher/student policy、critic 和优化器；
- 环境状态与 VecNorm；
- PPO 迭代号和累计环境帧；
- 原 WandB run ID。

训练进程异常退出时，systemd 等待 60 秒后重新启动脚本。第一阶段正常生成
`checkpoint_final.pt` 后脚本以成功状态退出，服务不会继续启动第二阶段。

常用命令：

```bash
systemctl --user status gp02_train_stage1.service
tail -f outputs/gp02_stage1_train.log

# 暂停；手动 stop 不会触发自动重启
systemctl --user stop gp02_train_stage1.service

# 从最新 checkpoint 继续
systemctl --user start gp02_train_stage1.service

# 停止并取消登录/开机后自动启动
systemctl --user disable --now gp02_train_stage1.service
```

## 7. 当前仍需实机信息确认的部分

- 双臂准确 KP/KD；
- 实机腕关节力矩限制；
- 腰部实际硬件关节究竟是 yaw/roll，还是控制接口中的 pitch/roll；
- 最终实机除腰部外是否还要固定手臂等关节；
- 电机侧 PD 是关节侧参数还是电机侧参数；
- 编码器零位、方向和减速比。

这些信息不影响当前“24 自由度模型、22 维策略动作”的仿真训练启动，但会直接影响策略导出和实机部署，不能仅靠训练模型猜测。
