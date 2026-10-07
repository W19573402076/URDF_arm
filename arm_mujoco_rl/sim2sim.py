#!/usr/bin/env python3
"""在 MuJoCo 里跑 Isaac Gym 训好的 PPO 策略（sim2sim）。

    # 批量评测，出和 play.py 同口径的表格
    /home/wck/miniconda3/bin/python sim2sim.py --episodes 200 --headless

    # A/B 同场景对比（场景由 tools/dump_isaac_trace.py --scenarios 导出）
    /home/wck/miniconda3/bin/python sim2sim.py --scenarios isaac_scenarios.npz --headless

    # 开窗口看策略跑
    /home/wck/miniconda3/bin/python sim2sim.py --render

先跑 `verify.py` 确认映射链路干净，再跑这个。
"""

import argparse
import os
import sys
import time
from pathlib import Path

# 输入法会抢走窗口的键盘事件（按 ESC 想退出结果弹候选框），必须在建立 X11/GLFW 连接前禁掉。
# 做法抄自 ../arm_mujoco/ik_control.py。
os.environ["XMODIFIERS"] = "@im=none"
os.environ["GTK_IM_MODULE"] = "none"
os.environ["QT_IM_MODULE"] = "none"
os.environ["QT4_IM_MODULE"] = "none"

import numpy as np  # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import config as C                 # noqa: E402
import controller                  # noqa: E402
import kinematics                  # noqa: E402
import runner as R                 # noqa: E402
from backend import MuJoCoBackend  # noqa: E402
from policy import Policy          # noqa: E402

DEFAULT_XML = _HERE.parent / "arm_mujoco" / "URDF_arm.xml"
DEFAULT_CKPT = _HERE.parent / "arm_rl" / "logs" / "Sep19_19-10-57_" / "model_20000.pt"


def make_backend(a):
    integrator = None
    if a.integrator:
        import mujoco
        integrator = getattr(mujoco.mjtIntegrator, "mjINT_" + a.integrator.upper())
    return MuJoCoBackend(a.xml, vel_limit=a.vel_limit,
                         ground_collision=a.ground_collision,
                         integrator=integrator)


def load_scenarios(path):
    d = np.load(path)
    return [R.EpisodeSpec(d["init_q"][i], d["target_pos"][i], d["target_quat"][i])
            for i in range(len(d["init_q"]))], d


def run_batch(a):
    backend = make_backend(a)
    policy = Policy.from_checkpoint(a.checkpoint)
    rng = np.random.default_rng(a.seed)

    if a.scenarios:
        specs, ref = load_scenarios(a.scenarios)
        if a.episodes and a.episodes < len(specs):
            specs = specs[:a.episodes]
        print("A/B 模式：%d 个场景来自 %s" % (len(specs), a.scenarios))
    else:
        ref = None
        sampler = R.TargetSampler(backend, rng)
        print("自采样 %d 个 episode…" % a.episodes)
        specs = sampler.make_specs(a.episodes, rng)
        if a.dump_scenarios:
            np.savez_compressed(a.dump_scenarios,
                                init_q=np.array([s.init_q for s in specs]),
                                target_pos=np.array([s.target_pos for s in specs]),
                                target_quat=np.array([s.target_quat_xyzw for s in specs]))
            print("  已导出场景到 %s" % a.dump_scenarios)

    t0 = time.time()
    results = []
    for i, spec in enumerate(specs):
        results.append(R.run_episode(backend, policy, spec,
                                     use_gravity_ff=not a.no_gravity_ff))
        if (i + 1) % max(1, len(specs) // 5) == 0:
            print("  %d/%d" % (i + 1, len(specs)))
    dt = time.time() - t0
    print("  %d 个 episode 用了 %.1f s（%.2f s/个）" % (len(specs), dt, dt / len(specs)))

    m = R.summarize(results)

    if ref is not None:
        print("\n" + "=" * 62)
        print("  A/B 对比（同一批场景，Isaac vs MuJoCo）")
        print("=" * 62)
        n = len(results)
        ip = ref["pos_err"][:n] * 1000.0
        io = np.degrees(ref["ori_err"][:n])
        isucc = ((ref["pos_err"][:n] < C.SUCCESS_POS_TOL) &
                 (ref["ori_err"][:n] < C.SUCCESS_ORI_TOL))
        print("  %-10s %10s %10s %10s %10s" % ("", "位置中位", "位置p95", "姿态中位", "成功率"))
        print("  %-10s %8.1fmm %8.1fmm %8.2f° %9.1f%%"
              % ("Isaac", np.median(ip), np.percentile(ip, 95), np.median(io), 100 * isucc.mean()))
        print("  %-10s %8.1fmm %8.1fmm %8.2f° %9.1f%%"
              % ("MuJoCo", np.median(m["pos_mm"]), np.percentile(m["pos_mm"], 95),
                 np.median(m["ori_deg"]), 100 * m["success"].mean()))
        dp = np.median(m["pos_mm"]) - np.median(ip)
        ds = 100 * (m["success"].mean() - isucc.mean())
        print("\n  差值：位置中位 %+.1f mm   成功率 %+.1f 个百分点" % (dp, ds))
        ok = abs(dp) <= 3.0 and abs(ds) <= 5.0
        print("  判据（位置中位差 ≤3mm 且成功率差 ≤5pp）: %s"
              % ("通过 ✓" if ok else "**不通过 ✗**"))
    return results


def run_viewer(a):
    """开窗口跑策略。复用 ik_control.py 的 launch_passive + 固定 dt 节流骨架。"""
    import mujoco
    import mujoco.viewer

    backend = make_backend(a)
    policy = Policy.from_checkpoint(a.checkpoint)
    rng = np.random.default_rng(a.seed)

    if a.scenarios:
        specs, _ = load_scenarios(a.scenarios)
    else:
        specs = R.TargetSampler(backend, rng).make_specs(max(1, a.episodes), rng)

    spec_i = {"i": 0}
    print(__doc__)
    print("窗口操作：R/空格 = 换下一个目标，ESC = 退出\n")

    def key_callback(keycode):
        import glfw
        if keycode in (glfw.KEY_R, glfw.KEY_SPACE):
            spec_i["i"] = (spec_i["i"] + 1) % len(specs)
            print("\n>>> 切到场景 %d/%d" % (spec_i["i"] + 1, len(specs)))
        elif keycode == glfw.KEY_ESCAPE:
            raise KeyboardInterrupt

    try:
        import glfw  # noqa: F401
        cb = key_callback
    except ImportError:
        cb = None

    dt = backend.model.opt.timestep
    set_camera_once = [True]

    def frame_camera(viewer):
        """把相机拉到能装下整个工作空间的位置。

        ⚠️ 必须等第一次 `viewer.sync()` **之后**再设 —— 查看器的首次 sync 会做一次
        自动取景，把之前设的相机参数整个盖掉（`../arm_mujoco/view_arm.py:137-139`
        记过这个坑）。不设的话目标稍微偏一点就跑到画面外，看着像"球没动"。
        """
        # distance=2.2 是用离屏渲染逐个目标点试出来的：工作空间
        # （z ∈ [0.05, 0.70]，半径 ∈ [0, 0.65]）15 个采样点全部落在画面内。
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [0.0, 0.05, 0.25]
        viewer.cam.distance = 2.2
        viewer.cam.azimuth = -62.0
        viewer.cam.elevation = 22.0

    with mujoco.viewer.launch_passive(backend.model, backend.data,
                                      key_callback=cb) as viewer:
        while viewer.is_running():
            spec = specs[spec_i["i"]]
            backend.reset(spec.init_q, np.zeros(backend.n_dof))
            tp = np.asarray(spec.target_pos, float)
            tq = np.asarray(spec.target_quat_xyzw, float)
            # 把场景里那颗标记球挪到真正的目标上。不设的话它会一直停在 XML 里的
            # 固定位置 (0, 0.35, 0.35)，看着像目标，其实和策略在追的东西无关。
            backend.set_target_marker(tp, tq)

            gravity_ff = np.zeros(backend.n_dof) if not a.no_gravity_ff else None
            last_action = np.zeros(backend.n_dof)
            ep = 0
            best = None
            while viewer.is_running() and ep <= C.MAX_EPISODE_LENGTH:
                t0 = time.time()
                q, qd = backend.get_joint_state()
                ee_pos, ee_quat = backend.get_ee_pose()
                obs = kinematics.build_obs(q, qd, ee_pos, ee_quat, tp, tq, last_action)
                action = controller.clip_action(policy.act(obs))
                last_action = action

                for _ in range(C.DECIMATION):
                    q, qd = backend.get_joint_state()
                    tau = controller.compute_torques(action, q, qd, gravity_ff)
                    backend.set_joint_torques(tau)
                    backend.step()
                    viewer.sync()
                    if set_camera_once[0]:
                        frame_camera(viewer)
                        set_camera_once[0] = False

                ep += 1
                if not a.no_gravity_ff:
                    gravity_ff = backend.gravity_torque()

                # 实时误差。best 记本 episode 最好的一刻（达标判定取的是终点误差，
                # 但过程里最好能到多少更直观）
                ee_pos, ee_quat = backend.get_ee_pose()
                pe = kinematics.pos_error(ee_pos, tp)
                oe = np.degrees(kinematics.ori_error(ee_quat, tq))
                if best is None or pe < best:
                    best = pe
                ok = "达标" if (pe < C.SUCCESS_POS_TOL
                               and np.radians(oe) < C.SUCCESS_ORI_TOL) else "    "
                print("  [%s] 场景 %d/%d  步 %3d/%d   位置 %6.1f mm  姿态 %5.2f°   "
                      "本 episode 最好 %5.1f mm"
                      % (ok, spec_i["i"] + 1, len(specs), ep, C.MAX_EPISODE_LENGTH,
                         pe * 1000, oe, best * 1000), end="\r")

                # 每控制步 4*dt = 1/30 s，按实时节流（拿不到实时就自然降速）
                slack = C.DECIMATION * dt - (time.time() - t0)
                if slack > 0:
                    time.sleep(slack)

            print()   # 结束一个 episode 后换行，免得下一个的实时行盖住上一行


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--scenarios", default=None, help="A/B 场景文件（npz）")
    ap.add_argument("--dump-scenarios", default=None, help="把自采样的场景导出成 npz")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--xml", default=str(DEFAULT_XML))
    ap.add_argument("--no-gravity-ff", action="store_true",
                    help="关掉重力前馈（用来单独看它的影响）")
    ap.add_argument("--vel-limit", type=float, default=None,
                    help="关节速度限幅，对齐 Isaac 的 URDF velocity=3.14")
    ap.add_argument("--ground-collision", action="store_true",
                    help="开手臂-地面碰撞（自碰撞仍关），对齐 Isaac")
    ap.add_argument("--integrator", default=None, choices=["euler", "implicit", "implicitfast"],
                    help="覆写积分器（默认用 XML 里的 implicitfast）")
    a = ap.parse_args()

    if not os.path.exists(a.checkpoint):
        sys.exit("找不到 checkpoint: %s" % a.checkpoint)
    if a.render:
        run_viewer(a)
    else:
        run_batch(a)


if __name__ == "__main__":
    main()
