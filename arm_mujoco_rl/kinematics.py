"""四元数工具 + 观测构造 + 误差度量。纯 numpy。

**这个文件是整个移植里最容易翻车的地方**，改之前先读完下面的说明。

四元数顺序约定：本文件全链路统一用 **(x, y, z, w)** —— 也就是 Isaac Gym 的约定。
MuJoCo 的 `data.xquat` 是 (w,x,y,z)，**只在 backend 边界转一次**
（`wxyz_to_xyzw`），之后绝不再混。

[可移植] 上真机时原样带走。
"""

import numpy as np

import config as C


# ================================================================ 四元数 (xyzw)

def quat_mul(a, b):
    """Hamilton 积，xyzw。

    与 `isaacgym.torch_utils.quat_mul` 等价（那边用的是同一套展开式，
    数值上逐位一致 —— 已用 golden trace 对拍验证）。
    """
    ax, ay, az, aw = a[0], a[1], a[2], a[3]
    bx, by, bz, bw = b[0], b[1], b[2], b[3]
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def quat_conjugate(q):
    """xyzw 的共轭。与 isaacgym 的 `quat_conjugate` 一致（前三个分量取负）。"""
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_angle(q):
    """相对旋转的旋转角 [rad]。取绝对值再 acos —— q 和 -q 是同一个旋转，
    不取绝对值会在 180° 附近跳变。对应 `arm_reach_env.py:35-42`。"""
    return 2.0 * np.arccos(np.clip(abs(q[3]), 0.0, 1.0))


def wxyz_to_xyzw(q):
    """MuJoCo (w,x,y,z) → Isaac (x,y,z,w)。只在 backend 边界调用。"""
    return np.array([q[1], q[2], q[3], q[0]])


def xyzw_to_wxyz(q):
    """Isaac (x,y,z,w) → MuJoCo (w,x,y,z)。只在需要写回 MuJoCo 时用
    （目前只有查看器里挪目标标记球）。"""
    return np.array([q[3], q[0], q[1], q[2]])


def quat_to_rot6d(q):
    """四元数 → 旋转矩阵的前两列（拼成 6 维）。

    ⚠️⚠️ 这里有**故意的错位，必须原样保留，不要"顺手改对"** ⚠️⚠️

    原函数（`arm_reach_env.py:23-32`）的 docstring 写着「输入 (x,y,z,w)」，
    但函数体把 `q[:,0]` 当成 **w** 用 —— 也就是按 (w,x,y,z) 的顺序展开。
    而喂给它的 `q_rel` 来自 isaacgym 的 `quat_mul`，**确实是 (x,y,z,w)**。

    所以网络实际看到的 6D 向量是「把这个 (x,y,z,w) 四元数的四个分量重新解释成
    (w,x,y,z) 之后算出的 R 前两列」。**策略就是在这个表示上训练出来的**，
    改成标准公式会直接毁掉策略。

    数值对照（同一个 q_rel）：
        真 (x,y,z,w) 下的 R 前两列 = [ 0.42, -0.0149, -0.907,  0.615,  0.740,  0.272]
        本函数实际输出           = [-0.32,  0.272,   0.907, -0.672, -0.740, -0.015]

    下面是从原文件逐字符搬过来的，只是 torch → numpy。
    """
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    col0 = [1 - 2 * (q2 * q2 + q3 * q3),
            2 * (q1 * q2 + q0 * q3),
            2 * (q1 * q3 - q0 * q2)]
    col1 = [2 * (q1 * q2 - q0 * q3),
            1 - 2 * (q1 * q1 + q3 * q3),
            2 * (q2 * q3 + q0 * q1)]
    return np.array(col0 + col1)


# ================================================================ 观测

def build_obs(q, qd, ee_pos_rel, ee_quat_xyzw, target_pos, target_quat_xyzw,
              last_action):
    """30 维观测，对应 `arm_reach_env.py:293-306`。

    参数
        q                 (6,)     关节角 [rad]
        qd                (6,)     关节角速度 [rad/s]
        ee_pos_rel        (3,)     末端位置，**基座系**
        ee_quat_xyzw      (4,)     末端姿态，xyzw
        target_pos        (3,)     目标位置，基座系
        target_quat_xyzw  (4,)     目标姿态，xyzw
        last_action       (6,)     上一步**裁剪之后**的网络输出

    噪声在训练时是关闭的（`arm_reach_config.py:210`），这里也不加。
    """
    q_rel = quat_mul(quat_conjugate(ee_quat_xyzw), target_quat_xyzw)
    obs = np.concatenate([
        (q - C.DEFAULT_DOF_POS) * C.OBS_SCALE_DOF_POS,              # [ 0: 6]
        qd * C.OBS_SCALE_DOF_VEL,                                    # [ 6:12]
        (target_pos - ee_pos_rel) * C.OBS_SCALE_EEF_POS_ERR,         # [12:15]
        quat_to_rot6d(q_rel) * C.OBS_SCALE_EEF_ROT,                  # [15:21]
        ee_pos_rel * C.OBS_SCALE_EEF_POS,                            # [21:24]
        last_action,                                                 # [24:30]
    ])
    return np.clip(obs, -C.CLIP_OBS, C.CLIP_OBS)


# ================================================================ 误差度量

def pos_error(ee_pos_rel, target_pos):
    """末端位置误差 [m]（基座系）。对应 `arm_reach_env.py:310-311`。"""
    return float(np.linalg.norm(np.asarray(target_pos) - np.asarray(ee_pos_rel)))


def ori_error(ee_quat_xyzw, target_quat_xyzw):
    """末端姿态误差 [rad]。对应 `arm_reach_env.py:313-314`。"""
    return quat_angle(quat_mul(quat_conjugate(ee_quat_xyzw), target_quat_xyzw))
