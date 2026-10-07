#!/usr/bin/env python3
"""在 Isaac Gym 侧导出对拍数据（golden trace），给 ../verify.py 用。

**必须跑在 legged_gym 环境里**（要 isaacgym + torch，且 ninja 要在 PATH 上）：

    conda activate legged_gym
    cd /home/wck/mujocoproject/URDF/arm_mujoco_rl
    python tools/dump_isaac_trace.py --out trace.npz

导出两组数据：

1) **FK 扫掠**（`fk_*`）：随机采 N 组关节角，写进仿真、走一步物理、读 Link6 刚体状态。
   同时记下**走完那一步之后的关节角** `q_after` —— 因为 PhysX 的刚体位姿对应的是积分后的
   关节角，用 `q_after` 去比 MuJoCo 的正解才是严格的逐位对比（用写进去的 `q` 比会带上
   一步积分的偏差，约 1e-3 rad 量级）。

2) **整段 episode**（`step_*`）：跑一个完整 episode，每个控制步落盘复现所需的全部量。
   这是最有价值的检查 —— 拿它重放我们的 obs/action/torque 三个环节，一次能抓出
   四元数顺序、rot6d 错位、符号、缩放、切片错位的所有 bug。

四元数一律以 Isaac 的 (x,y,z,w) 顺序落盘。
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)                       # arm_mujoco_rl/
_ARM_RL = os.path.join(os.path.dirname(_ROOT), "arm_rl")

sys.path.insert(0, _ROOT)
sys.path.insert(0, _ARM_RL)

import numpy as np  # noqa: E402

# 导入顺序照抄 arm_rl/play.py，别重排（isaacgym 必须在 torch 之前）
import isaacgym  # noqa: F401,E402
from legged_gym.envs import *  # noqa: F401,F403,E402
from legged_gym.utils import task_registry  # noqa: E402
from legged_gym.utils.helpers import update_cfg_from_args  # noqa: E402

from arm_reach_config import ArmReachCfg, ArmReachCfgPPO  # noqa: E402
from arm_reach_env import ArmReach  # noqa: E402

import torch  # noqa: E402
from legged_gym.utils.helpers import get_args  # noqa: E402

LOG_ROOT = os.path.join(_ARM_RL, "logs")


def register():
    if "arm_reach" not in task_registry.task_classes:
        task_registry.register("arm_reach", ArmReach, ArmReachCfg(), ArmReachCfgPPO())


def build(args):
    register()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, args)
    env_cfg.env.num_envs = 1
    train_cfg.runner.resume = True
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=LOG_ROOT)
    return env, runner


def _set_dofs_and_step(env, q):
    """写关节角 → 走一步物理 → 读 (积分后的关节角, 末端位姿)。

    这一步物理是必须的：PhysX 的刚体变换只在 simulate 之后才更新。代价是关节角会被
    积分改变一点点（约 1e-3 rad），所以把积分后的值也返回，对比时用它。
    """
    ids = torch.tensor([0], dtype=torch.int32, device=env.device)
    env.dof_state_3d[0, :, 0] = torch.as_tensor(q, dtype=torch.float, device=env.device)
    env.dof_state_3d[0, :, 1] = 0.0
    env._write_dof_states(ids)

    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)

    q_after = env.dof_state_3d[0, :, 0].clone()
    pos = (env.rigid_body_states[0, env.eef_index, 0:3] - env.env_origins[0]).clone()
    quat = env.rigid_body_states[0, env.eef_index, 3:7].clone()      # xyzw
    return q_after, pos, quat


def dump_fk(env, n, seed):
    """FK 扫掠。在关节行程内缩 joint_margin 的范围内采样，和训练时的目标采样一致。"""
    rng = np.random.default_rng(seed)
    m = env.cfg.target.joint_margin
    lo = env.dof_pos_limits_raw[:, 0].cpu().numpy()
    hi = env.dof_pos_limits_raw[:, 1].cpu().numpy()
    slo = lo + m * (hi - lo)
    shi = hi - m * (hi - lo)

    q_in = np.zeros((n, env.num_dof))
    q_after = np.zeros((n, env.num_dof))
    pos = np.zeros((n, 3))
    quat = np.zeros((n, 4))
    for i in range(n):
        q = rng.uniform(slo, shi)
        q_in[i] = q
        qa, p, qq = _set_dofs_and_step(env, q)
        q_after[i] = qa.cpu().numpy()
        pos[i] = p.cpu().numpy()
        quat[i] = qq.cpu().numpy()
        if (i + 1) % 8 == 0:
            print("  FK 扫掠 %d/%d" % (i + 1, n))
    return q_in, q_after, pos, quat


def _manual_reset(env):
    """复刻 base_task 的 reset()，但把 init_q 和 target 抓出来。

    base_task.reset() = reset_idx() + step(zeros)。拆开写是为了在两者之间记录
    复位后的关节角和本 episode 的目标 —— A/B 同场景对比要用它们当初始条件。
    """
    ids = torch.arange(env.num_envs, dtype=torch.int32, device=env.device)
    env.reset_idx(ids)
    init_q = env.dof_state_3d[0, :, 0].detach().clone()
    target_pos = env.target_pos[0].detach().clone()
    target_quat = env.target_quat[0].detach().clone()
    zeros = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    obs, _, _, _, _ = env.step(zeros)
    return init_q, target_pos, target_quat, obs


def dump_scenarios(env, runner, n, max_steps):
    """跑 N 个 episode，导出 (init_q, target, 终止误差)。

    这是 A/B 同场景对比的基准 —— MuJoCo 侧喂同一批初始条件，比较终点的位置/姿态误差。
    """
    policy = runner.get_inference_policy(device=env.device)
    init_q = np.zeros((n, env.num_dof))
    tpos = np.zeros((n, 3))
    tquat = np.zeros((n, 4))
    pos_err = np.zeros(n)
    ori_err = np.zeros(n)

    for i in range(n):
        q0, tp, tq, obs = _manual_reset(env)
        init_q[i] = q0.cpu().numpy()
        tpos[i] = tp.cpu().numpy()
        tquat[i] = tq.cpu().numpy()
        for _ in range(max_steps):
            with torch.no_grad():
                a = policy(obs)
            obs, _, _, _, _ = env.step(a)
            if bool(env.reset_buf.any()):
                break
        pos_err[i] = env.episode_eef_pos_err[0].item()
        ori_err[i] = env.episode_eef_ori_err[0].item()
        if (i + 1) % 20 == 0:
            print("  场景 %d/%d   位置中位 %.1f mm  成功率 %.0f%%"
                  % (i + 1, n, np.median(pos_err[:i + 1]) * 1000,
                     100 * np.mean((pos_err[:i + 1] < 0.02) & (ori_err[:i + 1] < 0.1))))
    return init_q, tpos, tquat, pos_err, ori_err


def dump_episode(env, runner, max_steps):
    """跑一个 episode，逐控制步落盘。"""
    policy = runner.get_inference_policy(device=env.device)

    # 钩住 _compute_torques，抓每个控制步**第一个物理子步**的输入和输出。
    # 一个控制步里它被调 4 次，只有第一次的 q/qd 是控制步开始时的状态。
    hook = {"first": True, "q": None, "qd": None, "tau": None, "act_in": None}

    orig_compute = env._compute_torques

    def hooked(actions):
        tau = orig_compute(actions)
        if hook["first"]:
            hook["q"] = env.dof_pos[0].detach().clone()
            hook["qd"] = env.dof_vel[0].detach().clone()
            hook["tau"] = tau[0].detach().clone()
            hook["act_in"] = actions[0].detach().clone()
            hook["first"] = False
        return tau

    env._compute_torques = hooked

    init_q, tp, tq, obs = _manual_reset(env)

    rec = {k: [] for k in (
        "q", "qd", "ee_pos", "ee_quat", "target_pos", "target_quat",
        "last_action", "obs", "action_raw", "action_clipped",
        "tau_q", "tau_qd", "tau_out", "gravity_ff")}

    for t in range(max_steps):
        # 记录「决策前」的状态：这一帧的 obs 就是用这些量算出来的
        rec["q"].append(env.dof_pos[0].detach().cpu().numpy().copy())
        rec["qd"].append(env.dof_vel[0].detach().cpu().numpy().copy())
        rec["ee_pos"].append(env.ee_pos_rel[0].detach().cpu().numpy().copy())
        rec["ee_quat"].append(env.ee_quat[0].detach().cpu().numpy().copy())
        rec["target_pos"].append(env.target_pos[0].detach().cpu().numpy().copy())
        rec["target_quat"].append(env.target_quat[0].detach().cpu().numpy().copy())
        rec["last_action"].append(env.actions[0].detach().cpu().numpy().copy())
        rec["obs"].append(obs[0].detach().cpu().numpy().copy())
        rec["gravity_ff"].append(env.gravity_ff[0].detach().cpu().numpy().copy())

        with torch.no_grad():
            action_raw = policy(obs)
        rec["action_raw"].append(action_raw[0].detach().cpu().numpy().copy())
        rec["action_clipped"].append(
            torch.clip(action_raw, -env.cfg.normalization.clip_actions,
                       env.cfg.normalization.clip_actions)[0].detach().cpu().numpy().copy())

        hook["first"] = True
        # legged_gym 的 step 返回 5 个：obs, privileged_obs, rew, reset_buf, extras
        obs, _, _, _, _ = env.step(action_raw)

        rec["tau_q"].append(hook["q"].cpu().numpy().copy())
        rec["tau_qd"].append(hook["qd"].cpu().numpy().copy())
        rec["tau_out"].append(hook["tau"].cpu().numpy().copy())

        if (t + 1) % 25 == 0:
            print("  episode 第 %d 步" % (t + 1))

        if bool(env.reset_buf.any()):
            print("  episode 在第 %d 步终止" % (t + 1))
            break

    env._compute_torques = orig_compute
    rec["init_q"] = [init_q.cpu().numpy()]
    rec["ep_target_pos"] = [tp.cpu().numpy()]
    rec["ep_target_quat"] = [tq.cpu().numpy()]
    return {k: np.array(v) for k, v in rec.items()}, t + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="isaac_trace.npz")
    ap.add_argument("--scenarios-out", default=None,
                    help="A/B 同场景对比用的场景文件（init_q + 目标 + Isaac 的真实结果）")
    ap.add_argument("--fk-samples", type=int, default=32)
    ap.add_argument("--scenarios", type=int, default=0,
                    help="额外跑 N 个 episode 导出 A/B 场景，0 = 不导出")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    known, rest = ap.parse_known_args()

    # get_args() 用 gymutil 的解析器，认不了本脚本的参数，摘出去再交回去
    sys.argv = [sys.argv[0], "--task", "arm_reach", "--headless", "--num_envs", "1"] + rest

    args = get_args()
    if args.task == "anymal_c_flat":
        args.task = "arm_reach"

    print("构建环境…")
    env, runner = build(args)
    print("  num_dof=%d  num_obs=%d  max_episode_length=%s"
          % (env.num_dof, env.num_obs, env.max_episode_length))
    print("  env_origins[0] =", env.env_origins[0].detach().cpu().numpy())
    print("  default_dof_pos =", env.default_dof_pos[0].detach().cpu().numpy())
    print("  kp =", env.p_gains.detach().cpu().numpy())
    print("  kd =", env.d_gains.detach().cpu().numpy())
    print("  torque_limits =", env.torque_limits.detach().cpu().numpy())
    print("  decimation =", env.cfg.control.decimation, " sim.dt =", env.cfg.sim.dt)

    print("\nFK 扫掠 %d 组…" % known.fk_samples)
    q_in, q_after, fk_pos, fk_quat = dump_fk(env, known.fk_samples, known.seed)

    print("\n跑一个 episode…")
    step, n_steps = dump_episode(env, runner, known.max_steps)

    out = {"fk_q_in": q_in, "fk_q_after": q_after,
           "fk_pos": fk_pos, "fk_quat": fk_quat,
           "n_steps": np.array([n_steps]),
           "default_dof_pos": env.default_dof_pos[0].detach().cpu().numpy(),
           "kp": env.p_gains.detach().cpu().numpy(),
           "kd": env.d_gains.detach().cpu().numpy(),
           "torque_limits": env.torque_limits.detach().cpu().numpy(),
           "decimation": np.array([env.cfg.control.decimation]),
           "sim_dt": np.array([env.cfg.sim.dt]),
           "action_scale": np.array([env.cfg.control.action_scale]),
           "env_origin": env.env_origins[0].detach().cpu().numpy()}
    for k, v in step.items():
        out["step_" + k] = v

    if known.scenarios > 0:
        print("\n跑 %d 个 episode 导出 A/B 场景…" % known.scenarios)
        iq, tp, tq, pe, oe = dump_scenarios(env, runner, known.scenarios, 200)
        out.update({"sc_init_q": iq, "sc_target_pos": tp, "sc_target_quat": tq,
                    "sc_pos_err": pe, "sc_ori_err": oe})
        s_out = known.scenarios_out or known.out.replace(".npz", "_scenarios.npz")
        np.savez_compressed(s_out, init_q=iq, target_pos=tp, target_quat=tq,
                            pos_err=pe, ori_err=oe)
        print("已写出场景文件 %s" % os.path.abspath(s_out))
        print("  Isaac 侧基线：位置中位 %.1f mm，严格成功率 %.1f%%"
              % (np.median(pe) * 1000,
                 100 * np.mean((pe < 0.02) & (oe < 0.1))))

    np.savez_compressed(known.out, **out)
    print("\n已写出 %s" % os.path.abspath(known.out))
    print("  字段: %s" % ", ".join(sorted(out.keys())))


if __name__ == "__main__":
    main()
