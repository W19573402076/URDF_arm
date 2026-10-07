"""训练侧的全部常量，纯数据，不依赖 mujoco / torch / isaacgym。

数值逐条抄自 `../arm_rl/arm_reach_config.py`，改训练配置时这里要同步。
[可移植] 这个文件将来上真机时原样带走。
"""

import numpy as np

# ---------------------------------------------------------------- 关节
# 顺序恒为 Joint1..Joint6，两侧仿真的关节序一致，无需重排或取反
DEFAULT_DOF_POS = np.array([0.0, 0.0, 0.35, 0.0, 0.0, 0.0])
#                                              ↑ Joint3 的零位是它行程的下限，不是 0

# arm_reach_config.py:86-94，按名字子串匹配 Joint1..6 得到的
KP = np.array([80.0, 80.0, 80.0, 20.0, 20.0, 20.0])
KD = np.array([4.0, 4.0, 4.0, 1.0, 1.0, 1.0])
# 来自 URDF 的 <limit effort>，同时也是 MuJoCo 侧 jnt_actfrcrange 的值
TORQUE_LIMITS = np.array([20.0, 20.0, 20.0, 3.0, 3.0, 3.0])

ACTION_SCALE = 0.25

# ---------------------------------------------------------------- 观测
OBS_SCALE_DOF_POS = 1.0
OBS_SCALE_DOF_VEL = 0.1
OBS_SCALE_EEF_POS_ERR = 2.0
OBS_SCALE_EEF_POS = 1.0
OBS_SCALE_EEF_ROT = 1.0
NUM_OBS = 30

CLIP_OBS = 100.0          # 注意不是 ±1
CLIP_ACTION = 100.0       # 同上，等于不裁剪（网络输出可以明显超过 1）

# ---------------------------------------------------------------- 时基
PHYS_DT = 1.0 / 120.0     # arm_reach_config.py:218 (sim.dt)
DECIMATION = 4            # 控制 30 Hz
EPISODE_LENGTH_S = 5.0
# 终止判据是 episode_length_buf > max_episode_length（严格大于），
# 而 max_episode_length = ceil(5.0/(4/120)) = 150.0，所以是 151 步。
# 实测定格：Isaac 侧 env.reset() 之后又走了一步，所以实际 150 个策略步后终止。
MAX_EPISODE_LENGTH = 150.0

# ---------------------------------------------------------------- 目标采样
TARGET_JOINT_MARGIN = 0.1
TARGET_MIN_Z = 0.05
TARGET_MAX_RADIUS = 0.65
TARGET_MAX_TRIES = 8

# ---------------------------------------------------------------- 达标判据
SUCCESS_POS_TOL = 0.020   # m
SUCCESS_ORI_TOL = 0.100   # rad

# ---------------------------------------------------------------- 物理
GRAVITY = np.array([0.0, 0.0, -9.81])

# ---------------------------------------------------------------- 网络
ACTOR_HIDDEN_DIMS = (512, 256, 128)
ACTIVATION = "elu"
