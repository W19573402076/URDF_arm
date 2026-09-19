#!/usr/bin/env python3
"""在 Isaac Gym 里打开 URDF_arm 六轴机械臂。

先跑一次 build_assets.py 生成 assets/，然后：

    python view_arm.py                 # 开窗口，六个关节按正弦摆动
    python view_arm.py --static_pose   # 开窗口，保持零位（只看模型/调姿态用）
    python view_arm.py --num_envs 4    # 一排摆 4 个
    python view_arm.py --headless --steps 240   # 不开窗口，只跑物理，用来验证资源能不能加载
    python view_arm.py --free_base     # 基座不焊死在世界坐标系上（会受重力掉下来）

窗口里：鼠标左键拖动=转视角，滚轮=缩放，中键=平移，ESC=退出。
"""

import math
from pathlib import Path

import numpy as np
from isaacgym import gymapi, gymutil

HERE = Path(__file__).resolve().parent
ASSET_ROOT = str(HERE / "assets")
ASSET_FILE = "urdf/URDF_arm.urdf"

# 关节 PD 增益。源 URDF 的 effort 限制偏小（末端三个轴只有 3 N·m），
# 这里放宽一点，纯粹是为了在窗口里摆得动，不影响 URDF 文件本身。
STIFFNESS = 60.0
DAMPING = 3.0
MIN_EFFORT = 30.0
MIN_VELOCITY = 10.0


def parse_args():
    return gymutil.parse_arguments(
        description="在 Isaac Gym 中打开 URDF_arm 机械臂",
        headless=True,
        custom_parameters=[
            {"name": "--static_pose", "action": "store_true",
             "help": "保持零位不动（默认六个关节按正弦摆动）"},
            {"name": "--num_envs", "type": int, "default": 1, "help": "环境数量，默认 1"},
            {"name": "--spacing", "type": float, "default": 1.2, "help": "多环境时间距（米）"},
            {"name": "--free_base", "action": "store_true",
             "help": "基座不固定在世界坐标系上（默认固定）"},
            {"name": "--use_mesh_materials", "action": "store_true",
             "help": "使用网格自带材质（STL 不含材质，一般不用加）"},
            {"name": "--amplitude_scale", "type": float, "default": 0.35,
             "help": "摆动幅度占关节行程的比例，默认 0.35"},
            {"name": "--frequency", "type": float, "default": 0.25, "help": "正弦摆动频率（Hz），默认 0.25"},
            {"name": "--steps", "type": int, "default": 0,
             "help": "headless 模式下跑多少步后退出，默认 240"},
        ],
    )


def create_sim(gym, args):
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = False  # 这个例子走 CPU 张量接口，简单稳一点

    if args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id,
                         args.physics_engine, sim_params)
    if sim is None:
        raise RuntimeError("创建 sim 失败")

    # 注意：PlaneParams() 的默认法线是 (0,1,0)，也就是 Y 轴向上。
    # 本仿真用 Z 轴向上，必须显式改成 (0,0,1)，否则地板是竖直的，
    # 会直接切进机械臂里，把关节弹飞。
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane_params.static_friction = 1.0
    plane_params.dynamic_friction = 1.0
    plane_params.restitution = 0.0
    gym.add_ground(sim, plane_params)
    return sim


def load_arm(gym, sim, args):
    options = gymapi.AssetOptions()
    options.fix_base_link = not args.free_base
    options.use_mesh_materials = args.use_mesh_materials
    options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)

    print("加载模型：%s/%s" % (ASSET_ROOT, ASSET_FILE))
    asset = gym.load_asset(sim, ASSET_ROOT, ASSET_FILE, options)
    if asset is None:
        raise RuntimeError("加载失败。先确认已经跑过 build_assets.py 且 assets/ 存在。")
    return asset


def print_asset_info(gym, asset):
    dof_names = gym.get_asset_dof_names(asset)
    dof_props = gym.get_asset_dof_properties(asset)

    print()
    print("连杆 %d 个：" % gym.get_asset_rigid_body_count(asset))
    for i in range(gym.get_asset_rigid_body_count(asset)):
        print("  [%d] %s" % (i, gym.get_asset_rigid_body_name(asset, i)))

    print()
    print("关节 %d 个：" % gym.get_asset_joint_count(asset))
    for i in range(gym.get_asset_joint_count(asset)):
        print("  [%d] %-8s 类型=%s" % (i, gym.get_asset_joint_name(asset, i),
                                      gym.get_asset_joint_type(asset, i)))
    print()
    for i, name in enumerate(dof_names):
        lim = ("[%.3f, %.3f]" % (dof_props["lower"][i], dof_props["upper"][i])
               if dof_props["hasLimits"][i] else "无限位")
        print("DOF %d  %-8s 限位 %s  effort=%.2f" % (i, name, lim, dof_props["effort"][i]))
    print()


def build_targets(dof_props, args):
    """算出每个关节的摆动中心、幅度和周期。"""
    n = len(dof_props["lower"])
    lower = dof_props["lower"].astype(np.float64)
    upper = dof_props["upper"].astype(np.float64)

    center = np.zeros(n)
    amplitude = np.zeros(n)
    for i in range(n):
        if not dof_props["hasLimits"][i]:
            lower[i], upper[i] = -math.pi, math.pi
        # 中心取零位，零位超出行程就贴到最近的限位
        center[i] = min(max(0.0, lower[i]), upper[i])
        if args.static_pose:
            amplitude[i] = 0.0
        else:
            amplitude[i] = args.amplitude_scale * (upper[i] - lower[i]) / 2.0

    return center, amplitude


def main():
    args = parse_args()

    gym = gymapi.acquire_gym()
    sim = create_sim(gym, args)
    asset = load_arm(gym, sim, args)
    print_asset_info(gym, asset)

    # 摆成方阵，方便一次看多个
    num_envs = max(1, args.num_envs)
    num_per_row = int(math.ceil(math.sqrt(num_envs)))
    spacing = args.spacing
    env_lower = gymapi.Vec3(-spacing / 2.0, -spacing / 2.0, 0.0)
    env_upper = gymapi.Vec3(spacing / 2.0, spacing / 2.0, spacing)

    envs, actors = [], []
    dof_states = np.zeros(gym.get_asset_dof_count(asset), dtype=gymapi.DofState.dtype)

    for i in range(num_envs):
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        actor = gym.create_actor(env, asset, pose, "URDF_arm", i, 1)

        # 让关节走位置伺服，这样正弦目标会真的被跟踪
        dof_props = gym.get_actor_dof_properties(env, actor)
        dof_props["driveMode"][:] = gymapi.DOF_MODE_POS
        dof_props["stiffness"][:] = STIFFNESS
        dof_props["damping"][:] = DAMPING
        dof_props["effort"][:] = np.maximum(dof_props["effort"], MIN_EFFORT)
        dof_props["velocity"][:] = np.maximum(dof_props["velocity"], MIN_VELOCITY)
        gym.set_actor_dof_properties(env, actor, dof_props)

        gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)
        gym.set_actor_dof_position_targets(env, actor,
                                           np.zeros(len(dof_states), dtype=np.float32))
        envs.append(env)
        actors.append(actor)

    # 质量只有 actor 这一层能拿到，放在建完 actor 之后打印
    body_props = gym.get_actor_rigid_body_properties(envs[0], actors[0])
    body_names = gym.get_actor_rigid_body_names(envs[0], actors[0])
    print("质量：")
    for name, prop in zip(body_names, body_props):
        print("  %-10s %.4f kg" % (name, prop.mass))
    print()

    dof_props = gym.get_actor_dof_properties(envs[0], actors[0])
    center, amplitude = build_targets(dof_props, args)
    print("摆动幅度（弧度）：", np.round(amplitude, 3))
    print()

    # 相机，对准方阵中心
    half = spacing * (num_per_row - 1) / 2.0
    cam_target = gymapi.Vec3(0.0, 0.0, 0.25)
    viewer = None
    if not args.headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise RuntimeError("创建窗口失败，可以加 --headless 只跑物理")
        gym.viewer_camera_look_at(
            viewer, None,
            gymapi.Vec3(half + 0.65, -half - 0.65, half * 0.5 + 0.55),
            cam_target,
        )

    max_steps = args.steps if args.steps > 0 else (0 if not args.headless else 240)
    step, t = 0, 0.0
    if args.headless:
        print("headless 模式，跑 %d 步……" % max_steps)

    while True:
        if viewer is not None and gym.query_viewer_has_closed(viewer):
            break
        if args.headless and step >= max_steps:
            break

        gym.simulate(sim)
        gym.fetch_results(sim, True)

        targets = center + amplitude * math.sin(2.0 * math.pi * args.frequency * t)
        for env, actor in zip(envs, actors):
            gym.set_actor_dof_position_targets(env, actor, targets.astype(np.float32))

        if viewer is not None:
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

        t += 1.0 / 60.0
        step += 1
        if args.headless and step % 60 == 0:
            pos = gym.get_actor_dof_states(envs[0], actors[0], gymapi.STATE_POS)["pos"]
            print("  step %4d  关节角 %s" % (step, np.round(pos, 3)))

    if args.headless:
        pos = gym.get_actor_dof_states(envs[0], actors[0], gymapi.STATE_POS)["pos"]
        print("结束：模型加载并仿真 %d 步正常，最终关节角 %s" % (step, np.round(pos, 3)))

    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
