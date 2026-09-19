#!/usr/bin/env python3
"""加载训练好的策略，评估末端位姿到达精度。

    # 加载 logs/ 下最新一次的 checkpoint，跑 200 个 episode 出统计
    python play.py --episodes 200

    # 开窗口看策略实际怎么动（策略照常跑，只是同时渲染）
    python play.py --episodes 20 --render

    # 指定某次训练
    python play.py --load_run Sep17_19-00-00_ --checkpoint 1500

评估口径：每个 episode 结束（超时）那一瞬间的末端误差，阈值见
arm_reach_config.ArmReachCfg.rewards 的 success_pos_tol / success_ori_tol。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import isaacgym  # noqa: F401
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import update_cfg_from_args

from arm_reach_config import ArmReachCfg, ArmReachCfgPPO
from arm_reach_env import ArmReach

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
LOG_ROOT = os.path.join(_HERE, "logs")


def register():
    if "arm_reach" not in task_registry.task_classes:
        task_registry.register("arm_reach", ArmReach, ArmReachCfg(), ArmReachCfgPPO())


def percentile(a, q):
    return float(np.percentile(a, q))


def play(args):
    register()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, args)

    if args.num_envs is None:
        env_cfg.env.num_envs = 64
    train_cfg.runner.resume = True

    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=LOG_ROOT)
    policy = ppo_runner.get_inference_policy(device=env.device)

    pos_tol = env_cfg.rewards.success_pos_tol
    ori_tol = env_cfg.rewards.success_ori_tol
    max_len = int(env.max_episode_length)

    print()
    print("评估 %d 个 episode（%d 个并行环境，每 episode %d 步）"
          % (args.episodes, env.num_envs, max_len))
    print("判定达标：位置 < %.0f mm 且 姿态 < %.3f rad"
          % (pos_tol * 1000, ori_tol))

    obs = env.get_observations()
    pos_errs, ori_errs = [], []
    steps_this_run = 0

    while len(pos_errs) < args.episodes:
        with torch.no_grad():
            actions = policy(obs)
        obs, _, _, _, _ = env.step(actions)
        steps_this_run += 1

        # reset_idx 在复位前把最终误差写进了这两个 buffer，这里取刚好结束的那些环境
        done = env.reset_buf.bool()
        if done.any():
            pos_errs.extend(env.episode_eef_pos_err[done].cpu().tolist())
            ori_errs.extend(env.episode_eef_ori_err[done].cpu().tolist())

        if steps_this_run > 40 * max_len:
            print("跑了 %d 步还没凑够 %d 个 episode，提前结束" % (steps_this_run, args.episodes))
            break

    pos_m = np.array(pos_errs[:args.episodes])          # [m]
    ori_rad = np.array(ori_errs[:args.episodes])        # [rad]
    pos = pos_m * 1000.0                                # [mm]
    ori = np.degrees(ori_rad)                           # [度]
    ok = (pos_m < pos_tol) & (ori_rad < ori_tol)

    print()
    print("=" * 62)
    print("  episode 数            %d" % len(pos))
    print("  位置误差 [mm]         mean %7.1f   median %7.1f   p95 %7.1f"
          % (pos.mean(), np.median(pos), percentile(pos, 95)))
    print("  姿态误差 [度]         mean %7.2f   median %7.2f   p95 %7.2f"
          % (ori.mean(), np.median(ori), percentile(ori, 95)))
    print("  成功率                %6.1f%%   (%d/%d)"
          % (100.0 * ok.mean(), ok.sum(), len(ok)))
    print("=" * 62)

    # 多档容差下的达标率。严格阈值（配置里那个）没达标不代表策略没用，
    # 这张表能看出误差具体分布在什么量级。
    print()
    print("  不同容差下的达标率（位置 / 姿态）：")
    for pt, ot_deg in [(20, 5.7), (30, 8.6), (50, 11.5), (80, 17.2), (100, 22.9)]:
        r = ((pos_m * 1000 < pt) & (ori * np.pi / 180 < np.radians(ot_deg))).mean()
        print("      < %4d mm 且 < %4.1f°   ->  %6.1f%%" % (pt, ot_deg, 100 * r))
    print("      位置 < 20 mm 单独        ->  %6.1f%%" % (100 * (pos_m * 1000 < 20).mean()))
    print("      姿态 < 0.1 rad 单独      ->  %6.1f%%" % (100 * (ori_rad < 0.1).mean()))

    # 关节是否贴着限位跑
    lim = env.dof_pos_limits
    outside = ((env.dof_pos < lim[:, 0]) | (env.dof_pos > lim[:, 1])).float().mean()
    print("  末端位置范围 [m]      x[%.2f, %.2f]  y[%.2f, %.2f]  z[%.2f, %.2f]"
          % (env.ee_pos_rel[:, 0].min(), env.ee_pos_rel[:, 0].max(),
             env.ee_pos_rel[:, 1].min(), env.ee_pos_rel[:, 1].max(),
             env.ee_pos_rel[:, 2].min(), env.ee_pos_rel[:, 2].max()))
    print("  关节超出软限位的比例  %.2f%%" % (100.0 * outside.item()))


if __name__ == "__main__":
    # get_args() 用的是 gymutil 的解析器，认不了本脚本专有的参数，
    # 所以先从 sys.argv 里把这些摘出来再把剩下的交回去。
    argv = sys.argv[1:]
    episodes = 200
    if "--episodes" in argv:
        i = argv.index("--episodes")
        episodes = int(argv[i + 1])
        del argv[i:i + 2]
    sys.argv = [sys.argv[0]] + argv

    args = get_args()
    if args.task == "anymal_c_flat":   # get_args 的默认任务是腿足的，换成我们的
        args.task = "arm_reach"
    args.episodes = episodes
    play(args)
