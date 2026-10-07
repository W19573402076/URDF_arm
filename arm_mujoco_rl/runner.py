"""episode 主循环 + 目标采样 + 指标汇总。

主循环逐字复刻 `legged_robot.step()` / `post_physics_step()` 的时序，三处必须对齐：

  1. **obs 在 decimation 的 4 个物理子步之后算**（不是每个子步里算）
  2. `obs[24:30]` 是**刚施加的、裁剪之后**的动作；裁剪发生在乘 action_scale **之前**
  3. **重力前馈滞后一个控制步**：在 decimation 之后更新，下一控制步才用

加上 `reset()` 内部那一记 `step(zeros)`，一个 episode 是 **150 个策略步**
（终止判据 `episode_length_buf > 150`，buf 从 1 数到 151）。

[可移植] 上真机时原样带走。
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

import config as C
import controller
import kinematics


@dataclass
class EpisodeSpec:
    """一个 episode 的初始条件。导出成 npz 就能做两侧同场景 A/B。"""
    init_q: np.ndarray            # (6,) 初始关节角
    target_pos: np.ndarray        # (3,) 基座系
    target_quat_xyzw: np.ndarray  # (4,)


@dataclass
class EpisodeResult:
    pos_err: float = 0.0          # [m]
    ori_err: float = 0.0          # [rad]
    steps: int = 0
    target_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    traj: Optional[dict] = None   # 需要时逐控制步落盘


def _decimate(backend, action_clipped, gravity_ff):
    """一个控制步 = 4 个物理子步。每个子步都用**当前**状态重算力矩，
    但 action 和 gravity_ff 在一个控制步内不变（对应 `legged_robot.py:88-92`）。"""
    for _ in range(C.DECIMATION):
        q, qd = backend.get_joint_state()
        tau = controller.compute_torques(action_clipped, q, qd, gravity_ff)
        backend.set_joint_torques(tau)
        backend.step()


def run_episode(backend, policy, spec, use_gravity_ff=True, collect_traj=False):
    """跑一个 episode，返回终止瞬间的误差。"""
    n = backend.n_dof
    backend.reset(spec.init_q, np.zeros(n))
    target_pos = np.asarray(spec.target_pos, dtype=float)
    target_quat = np.asarray(spec.target_quat_xyzw, dtype=float)

    # 构造期 gravity_ff = 0（Isaac 的 _init_buffers 也是 zeros）
    gravity_ff = np.zeros(n) if use_gravity_ff else None
    last_action = np.zeros(n)

    traj = {k: [] for k in ("q", "qd", "ee_pos", "ee_quat", "obs",
                            "action", "tau", "gravity_ff")} if collect_traj else None

    # ---- env.reset() 内部的那一记 step(zeros) ----
    _decimate(backend, np.zeros(n), gravity_ff)
    if use_gravity_ff:
        gravity_ff = backend.gravity_torque()
    episode_len = 1

    while True:
        # ---- 1) obs：用 decimation 之后的状态 ----
        q, qd = backend.get_joint_state()
        ee_pos, ee_quat = backend.get_ee_pose()
        obs = kinematics.build_obs(q, qd, ee_pos, ee_quat,
                                   target_pos, target_quat, last_action)

        # ---- 2) 策略输出 + 裁剪 ----
        action = controller.clip_action(policy.act(obs))
        last_action = action

        if collect_traj:
            traj["q"].append(q); traj["qd"].append(qd)
            traj["ee_pos"].append(ee_pos); traj["ee_quat"].append(ee_quat)
            traj["obs"].append(obs); traj["action"].append(action)
            traj["gravity_ff"].append(gravity_ff.copy() if gravity_ff is not None
                                      else np.zeros(n))

        # ---- 3) decimation ----
        _decimate(backend, action, gravity_ff)

        # ---- 4) post_physics_step 等价 ----
        episode_len += 1
        if use_gravity_ff:
            gravity_ff = backend.gravity_torque()      # 下一步才用 → 一步滞后

        if episode_len > C.MAX_EPISODE_LENGTH:         # 151 > 150
            break

    # 终止瞬间的误差（对应 Isaac 的 reset_idx 里、复位之前取的那一次）
    ee_pos, ee_quat = backend.get_ee_pose()
    res = EpisodeResult(
        pos_err=kinematics.pos_error(ee_pos, target_pos),
        ori_err=kinematics.ori_error(ee_quat, target_quat),
        steps=episode_len,
        target_pos=target_pos.copy())
    if collect_traj:
        res.traj = {k: np.array(v) for k, v in traj.items()}
    return res


# ================================================================ 目标与初始状态采样

def sample_init_q(rng, backend):
    """`default_dof_pos + U(-0.1, 0.1)`，再夹到硬限位（`arm_reach_env.py:164-173`）。"""
    lo, hi = backend.joint_limits()
    q = C.DEFAULT_DOF_POS + rng.uniform(-0.1, 0.1, size=backend.n_dof)
    return np.clip(q, lo, hi)


class TargetSampler:
    """正解生成目标关节角，再拒绝采样（`arm_reach_env.py:221-289`）。

    语义上有两处很容易改"错"，必须保留：

      - best 的更新发生在剔除达标行**之前**，且只有仍 pending 的行会被重采 ——
        所以这是「拒绝采样 + 全程取最优分」，**不是**取第一个合法的
      - 用**硬限位** `jnt_range`，不是软限位

    与 Isaac 唯一的实质差异：Isaac 采样时跑了一步物理再读位姿，我们用纯正解读采样点，
    差约 1e-4 rad 量级，可忽略。
    """

    def __init__(self, backend, rng):
        self.b = backend
        self.rng = rng
        lo, hi = backend.joint_limits()
        m = C.TARGET_JOINT_MARGIN
        self.slo = lo + m * (hi - lo)
        self.shi = hi - m * (hi - lo)

    def sample(self, n=1):
        best_pos = np.zeros((n, 3))
        best_quat = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))     # xyzw 单位四元数
        best_score = np.full(n, -1e9)
        pending = np.arange(n)

        for _ in range(C.TARGET_MAX_TRIES):
            if len(pending) == 0:
                break
            qs = self.rng.uniform(self.slo, self.shi, size=(len(pending), self.b.n_dof))
            still = []
            for row, q in zip(pending, qs):
                pos, quat = self.b.ee_pose_from_q(q)
                score = pos[2] - np.linalg.norm(pos[:2])       # 越高、越靠基座越好
                if score > best_score[row]:
                    best_score[row], best_pos[row], best_quat[row] = score, pos, quat
                r = np.linalg.norm(pos[:2])
                if not (pos[2] >= C.TARGET_MIN_Z and r <= C.TARGET_MAX_RADIUS):
                    still.append(row)                          # 只重采没达标的
            pending = np.array(still, dtype=int)

        return best_pos, best_quat

    def make_specs(self, n, init_rng):
        pos, quat = self.sample(n)
        return [EpisodeSpec(sample_init_q(init_rng, self.b), pos[i], quat[i])
                for i in range(n)]


# ================================================================ 指标

def summarize(results: List[EpisodeResult], verbose=True):
    """和 `arm_rl/play.py:96-124` 同口径的表格。"""
    pos = np.array([r.pos_err for r in results]) * 1000.0      # mm
    ori = np.degrees(np.array([r.ori_err for r in results]))   # 度
    ok = ((pos / 1000.0 < C.SUCCESS_POS_TOL) &
          (np.array([r.ori_err for r in results]) < C.SUCCESS_ORI_TOL))

    m = {"n": len(pos), "pos_mm": pos, "ori_deg": ori, "success": ok}
    if not verbose:
        return m

    print()
    print("=" * 62)
    print("  episode 数            %d" % len(pos))
    print("  位置误差 [mm]         mean %7.1f   median %7.1f   p95 %7.1f"
          % (pos.mean(), np.median(pos), np.percentile(pos, 95)))
    print("  姿态误差 [度]         mean %7.2f   median %7.2f   p95 %7.2f"
          % (ori.mean(), np.median(ori), np.percentile(ori, 95)))
    print("  成功率                %6.1f%%   (%d/%d)"
          % (100.0 * ok.mean(), ok.sum(), len(ok)))
    print("=" * 62)

    print("\n  不同容差下的达标率（位置 / 姿态）：")
    for pt, ot in [(20, 5.7), (30, 8.6), (50, 11.5), (80, 17.2), (100, 22.9)]:
        r = ((pos < pt) & (ori < ot)).mean()
        print("      < %4d mm 且 < %4.1f°   ->  %6.1f%%" % (pt, ot, 100 * r))
    print("      位置 < 20 mm 单独        ->  %6.1f%%" % (100 * (pos < 20).mean()))
    print("      姿态 < 0.1 rad 单独      ->  %6.1f%%"
          % (100 * (np.array([r.ori_err for r in results]) < 0.1).mean()))

    # 误差是否集中在低 z 目标 —— 区分「模型差异」和「逻辑 bug」的关键
    tz = np.array([r.target_pos[2] for r in results])
    tr = np.linalg.norm(np.array([r.target_pos[:2] for r in results]), axis=1)
    print("\n  按目标高度分层（位置中位误差）:")
    for lo_, hi_ in [(0.0, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 1.0)]:
        sel = (tz >= lo_) & (tz < hi_)
        if sel.sum():
            print("      z ∈ [%.2f,%.2f)  n=%3d   位置中位 %6.1f mm"
                  % (lo_, hi_, sel.sum(), np.median(pos[sel])))
    print("  按目标半径分层:")
    for lo_, hi_ in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.55), (0.55, 0.7)]:
        sel = (tr >= lo_) & (tr < hi_)
        if sel.sum():
            print("      r ∈ [%.2f,%.2f)  n=%3d   位置中位 %6.1f mm"
                  % (lo_, hi_, sel.sum(), np.median(pos[sel])))
    return m
