"""动作 → 关节力矩。纯 numpy。

控制律（`legged_robot.py:365` + `arm_reach_env.py:148-151`）：

    action          = clip(网络输出, ±100)          # 注意是 ±100，不是 ±1
    target_q        = default_dof_pos + 0.25*action
    tau_pd          = kp*(target_q - q) - kd*qd
    tau             = clip( clip(tau_pd, ±lim) + gravity_ff, ±lim )
                      └────── 第一道 ──────┘   └── 第二道 ──┘

**双重裁剪缺一不可**：基类先把 PD 力矩裁一次，env 覆写再加上前馈后又裁一次。
前馈本身能把关节推到力矩上限，少了第二道就会超限。

`gravity_ff` 由 backend 提供（仿真侧走 qfrc_bias，真机侧走 RNEA 或干脆不补），
**只含重力、不含科氏项**，而且比当前控制步**滞后一步**（在 decimation 之后才更新）。
这两点都由 runner 保证，这里只管算。

[可移植] 上真机时原样带走。
"""

import numpy as np

import config as C


def clip_action(action):
    """网络输出裁剪。训练时 `clip_actions = 100.0`，实际上等于不裁。"""
    return np.clip(np.asarray(action, dtype=float), -C.CLIP_ACTION, C.CLIP_ACTION)


def compute_torques(action_clipped, q, qd, gravity_ff=None,
                    kp=None, kd=None, limits=None, default=None):
    """关节力矩。`action_clipped` 必须是已经过 `clip_action` 的值。

    q, qd 用**当前**状态（decimation 循环里每个物理子步都会重算一次），
    而 `action_clipped` 和 `gravity_ff` 在一个控制步内保持不变。
    """
    kp = C.KP if kp is None else kp
    kd = C.KD if kd is None else kd
    limits = C.TORQUE_LIMITS if limits is None else limits
    default = C.DEFAULT_DOF_POS if default is None else default

    q = np.asarray(q, dtype=float)
    qd = np.asarray(qd, dtype=float)

    target_q = default + C.ACTION_SCALE * action_clipped
    tau = kp * (target_q - q) - kd * qd
    tau = np.clip(tau, -limits, limits)                 # 第一道

    if gravity_ff is not None:
        tau = tau + np.asarray(gravity_ff, dtype=float)
        tau = np.clip(tau, -limits, limits)             # 第二道

    return tau


def target_joint_pos(action_clipped, default=None):
    """PD 的位置目标，调试时单独看看用。"""
    default = C.DEFAULT_DOF_POS if default is None else default
    return default + C.ACTION_SCALE * np.asarray(action_clipped, dtype=float)
