#!/usr/bin/env python3
"""末端 6D 位姿到达任务的训练入口。

跑在 legged_gym conda 环境里（Python 3.8，装着 isaacgym + rsl_rl）：

    /home/wck/miniconda3/envs/legged_gym/bin/python train.py --headless

代码不放进已安装的 legged_gym 包里，而是从本目录注册任务，再借用它的
task_registry 和 rsl_rl 的 OnPolicyRunner —— 这样重装 legged_gym 不会丢代码。
"""

import os
import sys

# 让 `python train.py` 能 import 同目录的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入顺序必须跟 legged_gym/scripts/train.py 完全一致，原因有两个：
#   1. 必须先 import isaacgym 再 import torch，否则报
#      "PyTorch was imported before isaacgym modules"
#   2. legged_gym.envs 和 legged_gym.utils.task_registry 之间有循环导入，
#      只有先从 legged_gym.envs 进才不会炸
import isaacgym  # noqa: F401
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import update_cfg_from_args

from arm_reach_config import ArmReachCfg, ArmReachCfgPPO
from arm_reach_env import ArmReach

_HERE = os.path.dirname(os.path.abspath(__file__))
LOG_ROOT = os.path.join(_HERE, "logs")


def register():
    """把任务注册进 legged_gym 的全局 registry。

    legged_gym 自带的任务是在 legged_gym/envs/__init__.py 里注册的，这里用自己的
    模块名注册，不碰那个文件。
    """
    if "arm_reach" not in task_registry.task_classes:
        task_registry.register("arm_reach", ArmReach, ArmReachCfg(), ArmReachCfgPPO())


def train(args):
    register()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, args)

    print("=" * 70)
    print("任务        : %s" % args.task)
    print("仿真环境数  : %d" % env_cfg.env.num_envs)
    print("观测/动作   : %d / %d" % (env_cfg.env.num_observations, env_cfg.env.num_actions))
    print("物理/控制   : %.4f s / %.4f s（%.0f Hz 控制）"
          % (env_cfg.sim.dt, env_cfg.sim.dt * env_cfg.control.decimation,
             1.0 / (env_cfg.sim.dt * env_cfg.control.decimation)))
    print("episode     : %.1f s = %d 控制步"
          % (env_cfg.env.episode_length_s,
             round(env_cfg.env.episode_length_s / (env_cfg.sim.dt * env_cfg.control.decimation))))
    print("日志目录    : %s/%s" % (LOG_ROOT, train_cfg.runner.experiment_name))
    print("=" * 70)

    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=LOG_ROOT)

    print("\n开始训练，%d 次迭代" % train_cfg.runner.max_iterations)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations,
                     init_at_random_ep_len=True)


if __name__ == "__main__":
    args = get_args()
    # 本任务只有这一个 task，默认值改成它，省得每次都写 --task
    if args.task == "anymal_c_flat":
        args.task = "arm_reach"
    train(args)
