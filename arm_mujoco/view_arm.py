#!/usr/bin/env python3
"""在 MuJoCo 里打开 URDF_arm 六轴机械臂。

先跑一次 build_model.py 生成 URDF_arm.xml，然后：

    python view_arm.py                  # 开窗口，六个关节按正弦摆动
    python view_arm.py --static_pose    # 开窗口，停在零位
    python view_arm.py --camera front   # 用模型里预置的相机
    python view_arm.py --headless --steps 1000   # 不开窗口，只跑物理，验证模型

用 mujoco 自带的被动查看器：鼠标左键拖动=转视角，滚轮=缩放，右键拖动=平移，
空格=暂停，ESC=退出。左侧面板可以拖动关节滑块、看接触点、切换相机。
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

HERE = Path(__file__).resolve().parent
XML = HERE / "URDF_arm.xml"
JOINTS = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"]


def parse_args():
    p = argparse.ArgumentParser(description="在 MuJoCo 中打开 URDF_arm 机械臂")
    p.add_argument("--headless", action="store_true",
                   help="不开窗口，只跑物理（用来验证模型能否加载）")
    p.add_argument("--static_pose", action="store_true",
                   help="停在零位不动（默认六个关节按正弦摆动）")
    p.add_argument("--amplitude_scale", type=float, default=0.35,
                   help="摆动幅度占关节行程的比例，默认 0.35")
    p.add_argument("--frequency", type=float, default=0.25,
                   help="正弦摆动频率（Hz），默认 0.25")
    p.add_argument("--steps", type=int, default=0,
                   help="headless 模式下跑多少步后退出，默认 1000")
    p.add_argument("--camera", type=str, default=None,
                   help="启动时用的相机名，模型里预置了 front")
    p.add_argument("--no_skybox", action="store_true", help="不加载天空盒贴图")
    return p.parse_args()


def load_model():
    if not XML.is_file():
        sys.exit("找不到 %s，先运行：python build_model.py" % XML)
    return mujoco.MjModel.from_xml_path(str(XML))


def print_model_info(model, data):
    print()
    print("连杆 %d 个：" % (model.nbody - 1))
    for i in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print("  [%d] %-8s mass=%.4f kg" % (i, name, model.body_mass[i]))
    print("  （base_link 被 MuJoCo 并进了 worldbody，质量不参与动力学）")

    print()
    print("关节 %d 个，执行器 %d 个：" % (model.njnt, model.nu))
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        lo, hi = model.jnt_range[i]
        act = model.jnt_actfrcrange[i]
        print("  [%d] %-7s qpos[%d]  限位 [%.3f, %.3f]  effort ±%.0f N·m"
              % (i, name, model.jnt_qposadr[i], lo, hi, act[1]))
    print()


def make_targets(model, args):
    """每个关节的摆动中心和幅度。零位超出行程的（Joint3）贴到最近的限位。"""
    n = model.njnt
    center = np.zeros(n)
    amplitude = np.zeros(n)
    for i in range(n):
        lo, hi = model.jnt_range[i]
        center[i] = min(max(0.0, lo), hi)
        if not args.static_pose:
            amplitude[i] = args.amplitude_scale * (hi - lo) / 2.0
    return center, amplitude


def main():
    args = parse_args()
    model = load_model()
    data = mujoco.MjData(model)

    # 先摆到 keyframe 里的零位，再让位置伺服接管
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    print_model_info(model, data)

    center, amplitude = make_targets(model, args)
    print("摆动中心：", np.round(center, 3))
    print("摆动幅度：", np.round(amplitude, 3))
    data.ctrl[:] = center
    print()

    max_steps = args.steps if args.steps > 0 else 1000

    if args.headless:
        print("headless 模式，跑 %d 步……" % max_steps)
        for step in range(max_steps):
            data.ctrl[:] = center + amplitude * math.sin(
                2.0 * math.pi * args.frequency * step * model.opt.timestep)
            mujoco.mj_step(model, data)
            if step % (max_steps // 4 or 1) == 0:
                print("  step %4d  关节角 %s" % (step, np.round(data.qpos, 3)))
        print("结束：模型加载并仿真 %d 步正常，最终关节角 %s"
              % (max_steps, np.round(data.qpos, 3)))
        print("最终控制量 %s，最大速度 %.4f" % (np.round(data.ctrl, 3), np.abs(data.qvel).max()))
        return

    print("打开查看器，ESC 退出")
    start = time.time()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        fixed_cam = None
        if args.camera:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
            if cam_id < 0:
                print("没有名为 %s 的相机，用自由视角" % args.camera)
            else:
                fixed_cam = cam_id
        set_camera_once = True
        if args.no_skybox:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SKYBOX] = 0

        dt = model.opt.timestep
        while viewer.is_running():
            # 按真实时间推进，否则物理跑得比钟表快，正弦频率就不是 0.25 Hz 了
            t = time.time() - start
            data.ctrl[:] = center + amplitude * math.sin(2.0 * math.pi * args.frequency * t)
            mujoco.mj_step(model, data)
            viewer.sync()
            if set_camera_once:
                # 查看器第一次 sync 会做自动取景，把相机设置盖掉，所以放到这之后再设。
                # 机械臂高约 0.4 m，俯视 19°、距离 1.3 m 能把整条臂放进画面。
                if fixed_cam is not None:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = fixed_cam
                else:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                    viewer.cam.lookat[:] = [0.0, 0.0, 0.2]
                    viewer.cam.distance = 1.3
                    viewer.cam.azimuth = -55.0
                    viewer.cam.elevation = 19.0
                set_camera_once = False
            slack = dt - (time.time() - start - t)
            if slack > 0:
                time.sleep(slack)
    print("已退出")


if __name__ == "__main__":
    main()
