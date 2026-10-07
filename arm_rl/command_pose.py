#!/usr/bin/env python3
"""给末端下位姿指令，用训好的策略把末端开过去。

这是策略被训练的那个任务（随机目标位姿到达）的直接用法，所以也是检验它到底能用到
什么程度最直接的方式。

    # 用预设目标（窗口里按 1~5 随时切换）
    python command_pose.py --preset 1

    # 直接指定位置 + 欧拉角（rpy 单位是度，绕世界固定轴 XYZ 外旋）
    python command_pose.py --pos 0.0 0.40 0.30 --euler -130 2 -63

    # 直接指定位置 + 四元数 (w x y z)
    python command_pose.py --pos 0.0 0.40 0.30 --quat -0.36477 0.76849 -0.48307 0.20741

    # 随机目标（用环境自己的采样器，和训练时的目标分布一致）
    python command_pose.py --random

跑在 legged_gym 环境里（要先 conda activate，见 README）。窗口里：
    1~5     切换预设目标（会重置 episode，从零位重新走过去）
    R / 空格  要一个随机目标（训练分布的采样，用来感受策略的典型水平）
    V       暂停/恢复画面刷新
    ESC   退出
终端每 0.5 秒打印一次末端误差，看得出策略实际能到多少。

**关于精度**：策略对不同目标的表现差别很大。用 400 个随机目标评测，位置误差的中位数
是 70 mm、均值 110 mm、p95 363 mm，同时满足 20 mm 和 5.7° 的只有 6%。所以有的目标能到
4 cm 出头，有的会停在 30 cm 开外 —— 这是策略本身的能力上限，不是这个脚本的问题（见
README 里的分析：加性奖励让位置和姿态互相妥协）。

位置是**基座坐标系**下的（基座在原点），和训练时目标的表达方式一致。
"""

import argparse
import math
import os
import queue
import sys
import threading
import time

# ── 强制禁用输入法，必须在任何 X11 / GLFW 连接建立之前 ────────
# 否则中文输入法（ibus/fcitx）会抢走窗口的键盘事件：按 R 想换目标，结果弹出输入法
# 候选框，按键根本没送到窗口。这组变量让 XOpenIM 返回 NULL，GLFW 就不会创建 XIC。
# 做法抄自 ./ball/simulate.py，那里注释写得很清楚，是中文 Linux 下 GLFW 的老坑。
os.environ["XMODIFIERS"] = "@im=none"
os.environ["GTK_IM_MODULE"] = "none"
os.environ["QT_IM_MODULE"] = "none"
os.environ["QT4_IM_MODULE"] = "none"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import isaacgym  # noqa: F401
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import update_cfg_from_args

from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import quat_from_euler_xyz
import numpy as np
import torch

from arm_reach_config import ArmReachCfg, ArmReachCfgPPO
from arm_reach_env import ArmReach, quat_angle, quat_conjugate, quat_mul

_HERE = os.path.dirname(os.path.abspath(__file__))
LOG_ROOT = os.path.join(_HERE, "logs")

CMDLINE = "命令行"        # 终端输入的目标，用来和预设区分

# 预设目标：(基座系位置 [m], 四元数 wxyz)
# 这些是用正解反查出来的 —— 在关节行程内采 20 万组合法构型做正解，取末端最接近
# 指定位置的那一组，用它的完整位姿，所以位置和姿态都保证可达（和训练时目标的
# 生成方式一致）。距基座 0.37~0.62 m，z 0.14~0.54 m，都在可达范围内。
POSES = {
    "1 正前方-中": ([0.0079, 0.4036, 0.3027],
                 [-0.36477, 0.76849, -0.48307, 0.20741]),
    "2 正前方-低": ([0.0078, 0.3416, 0.1419],
                 [0.45655, -0.08458, -0.88545, -0.01973]),
    "3 左侧": ([-0.3553, 0.2275, 0.2949],
              [0.67889, 0.39306, -0.61800, 0.05186]),
    "4 右侧": ([0.3697, 0.2250, 0.2976],
              [-0.01057, -0.71393, 0.33564, -0.61444]),
    "5 高位": ([0.0039, 0.2902, 0.5432],
              [-0.97736, -0.03824, -0.10380, 0.18037]),
}
PRESET_KEYS = {
    "preset1": "1 正前方-中", "preset2": "2 正前方-低", "preset3": "3 左侧",
    "preset4": "4 右侧", "preset5": "5 高位",
}

TERMINAL_HELP = """终端里可以随时输入新目标（回车生效，会像切预设一样重置 episode）：
    pos x y z                     只改位置 [m]
    euler r p y                   只改姿态，欧拉角 [度]，世界固定轴 XYZ 外旋
    quat w x y z                  只改姿态，四元数 (w 在前)
    pos x y z euler r p y         位置和姿态一起改（关键词可以任意组合、顺序随意）
    random                        随机目标（训练分布）
    preset 3                      用预设 3
    help                          再打印一次这段
    quit                          退出
只改一项时另一项保持当前目标的值。也支持逗号分隔。"""


class CommandPoseArm(ArmReach):
    """把目标从「随机采样」换成「外部指定」。"""

    def _init_buffers(self):
        super()._init_buffers()
        # 指令目标（基座系）。先给个占位，脚本构造完 env 会立刻 set_target 覆盖掉。
        self.cmd_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.cmd_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.cmd_quat[:, 3] = 1.0
        self.use_random_target = False    # True 时交回父类的随机采样（= 训练分布）
        self.on_preset_change = None      # 脚本挂的回调，切预设时重置 episode

    def set_target(self, pos, quat):
        """pos/quat 是基座系下的。"""
        self.cmd_pos[:] = torch.tensor(pos, dtype=torch.float, device=self.device)
        self.cmd_quat[:] = torch.tensor(quat, dtype=torch.float, device=self.device)

    def _resample_commands(self, env_ids):
        """覆写掉父类的随机采样：直接用外部指定的目标。

        留了一个开关，打开就交回父类 —— 父类的采样是「随机关节角正解」，和训练时
        目标分布完全一致，用来感受策略的典型水平比手挑的预设更靠谱。
        """
        n = len(env_ids)
        if n == 0:
            return
        if self.use_random_target:
            super()._resample_commands(env_ids)
            return
        self.target_pos[env_ids] = self.cmd_pos[env_ids]
        self.target_quat[env_ids] = self.cmd_quat[env_ids]

    # ---------------------------------------------------------------- 窗口
    def render(self, sync_frame_time=True):
        """照着基类的实现重写，加上预设切换和目标标记的绘制。

        基类的 render 会把 viewer 事件取空，所以不能先调 super() 再处理自己的键。
        """
        if self.viewer is None:
            return
        if self.gym.query_viewer_has_closed(self.viewer):
            sys.exit()

        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.value <= 0 or not evt.action:
                continue
            if evt.action in PRESET_KEYS:
                if self.on_preset_change is not None:
                    self.on_preset_change(PRESET_KEYS[evt.action])
            elif evt.action == "random_target":
                if self.on_preset_change is not None:
                    self.on_preset_change(None)
            elif evt.action == "QUIT":
                sys.exit()
            elif evt.action == "toggle_viewer_sync":
                self.enable_viewer_sync = not self.enable_viewer_sync

        if self.device != 'cpu':
            self.gym.fetch_results(self.sim, True)

        if self.enable_viewer_sync:
            self._draw_target()
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            if sync_frame_time:
                self.gym.sync_frame_time(self.sim)
        else:
            self.gym.poll_viewer_events(self.viewer)

    def _draw_target(self):
        """在窗口里画出目标位姿：一个线框球 + 一个坐标系（红=x 绿=y 蓝=z）。

        注意 draw_lines 的 env 参数传 None 时 pose 按世界坐标解释；传具体 env 时按该
        env 的局部坐标。这里已经把 env_origins 加回去了，所以必须传 None，否则偏移两次。
        """
        if getattr(self, "_target_sphere", None) is None:
            return
        self.gym.clear_lines(self.viewer)
        # 画的是 env 当前真正在追的目标（target_pos/target_quat），不是 cmd_pos ——
        # 用随机目标时 cmd_pos 是过期的，画它会指错地方
        for i in range(self.num_envs):
            world = self.target_pos[i].cpu().numpy() + self.env_origins[i].cpu().numpy()
            q = self.target_quat[i].cpu().numpy()       # (x,y,z,w)，gymapi 就是这个顺序
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(float(world[0]), float(world[1]), float(world[2]))
            pose.r = gymapi.Quat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            gymutil.draw_lines(self._target_sphere, self.gym, self.viewer, None, pose)
            gymutil.draw_lines(self._target_axes, self.gym, self.viewer, None, pose)


def register_and_build(args):
    if "arm_reach" not in task_registry.task_classes:
        task_registry.register("arm_reach", ArmReach, ArmReachCfg(), ArmReachCfgPPO())
    task_registry.register("arm_reach_cmd", CommandPoseArm, ArmReachCfg(), ArmReachCfgPPO())
    env_cfg, train_cfg = task_registry.get_cfgs(name="arm_reach_cmd")
    env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, args)
    train_cfg.runner.resume = True
    env, env_cfg = task_registry.make_env(name="arm_reach_cmd", args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name="arm_reach_cmd", args=args, train_cfg=train_cfg, log_root=LOG_ROOT)
    return env, env_cfg, ppo_runner


def parse_pose_args(argv):
    """从命令行里摘出 --pos / --euler / --quat / --preset。"""
    def take(flag, n):
        if flag in argv:
            i = argv.index(flag)
            vals = [float(v) for v in argv[i + 1:i + 1 + n]]
            del argv[i:i + 1 + n]
            return vals
        return None

    pos = take("--pos", 3)
    euler = take("--euler", 3)
    quat = take("--quat", 4)
    preset = take("--preset", 1)
    use_random = "--random" in argv
    if use_random:
        argv.remove("--random")
    return pos, euler, quat, (int(preset[0]) if preset else None), use_random


def main():
    argv = sys.argv[1:]
    pos, euler, quat, preset, use_random = parse_pose_args(argv)
    sys.argv = [sys.argv[0]] + argv
    args = get_args()
    if args.task == "anymal_c_flat":
        args.task = "arm_reach_cmd"
    if args.num_envs is None:
        args.num_envs = 1

    # 决定初始目标
    preset_names = list(POSES.keys())
    if use_random:
        p, q = POSES[preset_names[0]]      # 占位，下面会打开随机采样
        print("使用随机目标（训练分布）")
    elif pos is None:
        name = preset_names[(preset or 1) - 1]
        p, q = POSES[name]
        print("使用预设 %d：%s" % (preset or 1, name))
    else:
        p = pos
        if quat is not None:
            q = quat
        elif euler is not None:
            r, pi, y = [math.radians(v) for v in euler]
            q = quat_from_euler_xyz(torch.tensor([r]), torch.tensor([pi]), torch.tensor([y]))
            q = q[0].numpy().tolist()
        else:
            q = POSES[preset_names[0]][1]
            print("没给姿态（--euler 或 --quat），用预设 1 的姿态")
        print("使用命令行指定的位姿")

    env, env_cfg, ppo_runner = register_and_build(args)

    # 目标标记的几何体，只在窗口模式用
    env._target_sphere = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0.2, 0.2))
    env._target_axes = gymutil.AxesGeometry(0.09)

    if env.viewer is not None:
        for name in PRESET_KEYS:
            env.gym.subscribe_viewer_keyboard_event(
                env.viewer, getattr(gymapi, "KEY_" + name[-1]), name)
        # R 和空格都绑到「换随机目标」：中文输入法会截获字母键（按 R 弹候选框），
        # 空格一般不会。脚本开头已经设了 4 个环境变量禁用输入法，空格是双保险。
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_R, "random_target")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_SPACE, "random_target")

    current = {"name": CMDLINE if pos is not None else preset_names[(preset or 1) - 1]}
    pending = {"reset": False, "name": None}
    # 终端输入的「命令行」目标，和预设区分开
    pose_cmd = {"pos": np.array(p, dtype=float), "quat": np.array(q, dtype=float)}

    def on_preset(name):
        """按键回调里只记状态，真正的 reset 放到主循环做 ——
        这个回调是从 render() 里触发的，而 render() 又是 step() 调用的，
        在这里直接 env.reset() 会重入 step()。

        name 为 None 表示要一个随机目标。
        """
        pending["reset"] = True
        pending["name"] = name
        print("\n>>> 切到%s" % ("随机目标（训练分布）" if name is None else "预设 " + name))

    env.on_preset_change = on_preset
    env.use_random_target = use_random
    env.set_target(p, q)
    env.reset()
    if use_random:
        current["name"] = "随机目标"

    policy = ppo_runner.get_inference_policy(device=env.device)
    pos_tol = env_cfg.rewards.success_pos_tol
    ori_tol = env_cfg.rewards.success_ori_tol

    # 打印 env 里真正在追的目标（随机模式下 p/q 只是占位）
    print("\n本 episode 目标（基座系）：位置 %s   姿态(wxyz) %s"
          % (np.round(env.target_pos[0].cpu().numpy(), 4),
             np.round(env.target_quat[0].cpu().numpy(), 5)))
    _print_reference_pose(env)


    print("达标阈值：位置 < %.0f mm 且 姿态 < %.1f 度" % (pos_tol * 1000, math.degrees(ori_tol)))
    print("窗口里按 1~5 切预设、R/空格 换随机目标、V 暂停刷新、ESC 退出")
    print()
    print(TERMINAL_HELP)
    print()

    # 后台线程读终端，主循环非阻塞地取 —— 直接用 input() 会把仿真的主循环卡住
    cmd_q = queue.Queue()

    def _stdin_reader():
        try:
            for line in sys.stdin:
                cmd_q.put(line.rstrip("\n"))
        except Exception:
            pass

    threading.Thread(target=_stdin_reader, daemon=True).start()

    obs = env.get_observations()
    step = 0
    last_print = 0.0
    best = None
    with torch.no_grad():
        while True:
            # ---- 处理终端输入（非阻塞）----
            while not cmd_q.empty():
                line = cmd_q.get().strip()
                if not line:
                    continue
                low = line.lower()
                if low in ("help", "?", "h"):
                    print()
                    print(TERMINAL_HELP)
                elif low in ("quit", "exit", "q!"):
                    print("\n退出")
                    return
                elif low in ("random", "rand", "r"):
                    pending["reset"], pending["name"] = True, None
                    print("\n>>> 切到随机目标（训练分布）")
                elif low.startswith(("preset", "p ")) and low.split()[-1] in "12345":
                    name = preset_names[int(low.split()[-1]) - 1]
                    pending["reset"], pending["name"] = True, name
                    print("\n>>> 切到预设 %s" % name)
                elif low.strip() in "12345" and len(low.strip()) == 1:
                    name = preset_names[int(low.strip()) - 1]
                    pending["reset"], pending["name"] = True, name
                    print("\n>>> 切到预设 %s" % name)
                else:
                    try:
                        # 用「当前目标」当基准，只改提到的分量
                        cur_p = env.target_pos[0].cpu().numpy()
                        cur_q = env.target_quat[0].cpu().numpy()      # (x,y,z,w)
                        new_p, new_q = parse_target_command(line, cur_p, cur_q)
                    except ValueError as e:
                        print("\n[输入有误] %s（输入 help 看用法）" % e)
                        continue
                    pose_cmd["pos"], pose_cmd["quat"] = new_p, new_q
                    pending["reset"], pending["name"] = True, CMDLINE
                    print("\n>>> 新目标  位置 %s  姿态(wxyz) %s"
                          % (np.round(new_p, 4),
                             np.round([new_q[3], new_q[0], new_q[1], new_q[2]], 5)))

            if pending["reset"]:
                pending["reset"] = False
                name = pending["name"]
                env.use_random_target = name is None
                if name is not None:
                    if name == CMDLINE:
                        p_, q_ = pose_cmd["pos"], pose_cmd["quat"]
                    else:
                        p_, q_ = POSES[name]
                    env.set_target(p_, q_)
                    print("    目标位置 %s" % np.round(p_, 3))
                env.reset()          # 从零位重新走过去，和训练时的 episode 一致
                obs = env.get_observations()
                best = None
                step = 0
                if name is None:
                    print("    目标位置 %s" % np.round(env.target_pos[0].cpu().numpy(), 3))
                    current["name"] = "随机目标"

            actions = policy(obs)
            obs, _, _, _, _ = env.step(actions)
            step += 1

            pos_err = env._pos_error().mean().item()
            ori_err = env._ori_error().mean().item()
            if best is None or pos_err < best[0]:
                best = (pos_err, ori_err)

            now = time.time()
            if now - last_print > 0.5:
                last_print = now
                ok = "达标" if (pos_err < pos_tol and ori_err < ori_tol) else "    "
                print("  [%s] %-12s 位置 %7.1f mm  姿态 %6.2f 度   本 episode 最好 %5.1f mm / %.2f 度"
                      % (ok, current["name"], pos_err * 1000, math.degrees(ori_err),
                         best[0] * 1000, math.degrees(best[1])), end="\r", flush=True)

            # episode 走完自动重置，best 也跟着清零
            if bool(env.reset_buf.any()):
                best = None


def _print_reference_pose(env):
    """打印零位（default_dof_pos）时末端的位姿，作为自己设目标时的参照起点。

    ⚠️ **不能直接读 `env.ee_pos_rel` / `env.ee_quat`**。这两个量只在
    `_post_physics_step_callback` 里刷新，而那只在 `step()` 的路径上；调用本函数时
    刚做完 `env.reset()`、一步都没走过，读到的是上一次刷新留下的**缓存值**，
    和当前关节角对不上（实测姿态能差 100° 以上）。而且 `reset_idx` 只写了关节角，
    PhysX 的刚体变换要 `simulate` 之后才更新 —— 这也是 `_resample_commands` 里
    要先走一步再读的原因。

    所以这里显式把关节角设成 default、走一步物理、读回来，再把关节状态还原。
    走那一步会让关节角被积分改变约 1e-3 rad，对「给个参照起点」这个用途无所谓。
    """
    ids = torch.arange(env.num_envs, dtype=torch.int32, device=env.device)
    keep = env.dof_state_3d.clone()
    env.dof_state_3d[:, :, 0] = env.default_dof_pos
    env.dof_state_3d[:, :, 1] = 0.0
    env._write_dof_states(ids)

    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)
    env.gym.refresh_rigid_body_state_tensor(env.sim)

    ee_p = (env.rigid_body_states[0, env.eef_index, 0:3]
            - env.env_origins[0]).cpu().numpy()
    ee_q = env.rigid_body_states[0, env.eef_index, 3:7].cpu().numpy()   # (x,y,z,w)

    # 还原，别影响这一局的初始状态
    env.dof_state_3d[:] = keep
    env._write_dof_states(ids)
    env.gym.refresh_dof_state_tensor(env.sim)

    eul = _quat_to_euler_deg(ee_q)
    print("零位时末端（可直接拿来当起点改）：")
    print("    --pos %.3f %.3f %.3f --euler %.1f %.1f %.1f" % (ee_p[0], ee_p[1], ee_p[2],
                                                               eul[0], eul[1], eul[2]))
    print("    --pos %.3f %.3f %.3f --quat %.5f %.5f %.5f %.5f"
          % (ee_p[0], ee_p[1], ee_p[2], ee_q[3], ee_q[0], ee_q[1], ee_q[2]))


def parse_target_command(line, cur_pos, cur_quat):
    """解析终端里输入的目标指令，返回 (pos, quat) 或抛 ValueError。

    支持 pos / euler / quat 三个关键词的任意组合，没提到的分量保持原值。
    四元数在内部一律用 Isaac Gym 的 (x,y,z,w) 顺序，用户输入的是 (w,x,y,z)。
    """
    toks = line.replace(",", " ").split()
    pos = np.array(cur_pos, dtype=float).copy()
    quat = np.array(cur_quat, dtype=float).copy()      # (x,y,z,w)
    i = 0
    n = len(toks)
    while i < n:
        t = toks[i].lower()
        if t in ("pos", "p"):
            if i + 3 >= n:
                raise ValueError("pos 后面要跟 3 个数")
            pos = np.array([float(v) for v in toks[i + 1:i + 4]])
            i += 4
        elif t in ("euler", "rpy", "e"):
            if i + 3 >= n:
                raise ValueError("euler 后面要跟 3 个数")
            r, p, y = [math.radians(float(v)) for v in toks[i + 1:i + 4]]
            q = quat_from_euler_xyz(torch.tensor([r]), torch.tensor([p]), torch.tensor([y]))
            quat = q[0].numpy().astype(float)
            i += 4
        elif t in ("quat", "q"):
            if i + 4 >= n:
                raise ValueError("quat 后面要跟 4 个数 (w x y z)")
            w, x, y, z = [float(v) for v in toks[i + 1:i + 5]]
            quat = np.array([x, y, z, w])
            i += 5
        else:
            raise ValueError("看不懂的输入 '%s'" % toks[i])
    return pos, quat


def _quat_to_euler_deg(q_xyzw):
    """四元数 (x,y,z,w) -> 绕世界固定轴 XYZ 外旋的 (roll, pitch, yaw)，单位度。

    和脚本接收 --euler 时的约定一致（R = Rz(yaw)·Ry(pitch)·Rx(roll)）。
    """
    x, y, z, w = [float(v) for v in q_xyzw]
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))

    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


if __name__ == "__main__":
    main()
