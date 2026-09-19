#!/usr/bin/env python3
"""拖拽目标点，用逆运动学控制机械臂末端。

场景里有一个红色半透明的 mocap 目标球，**Ctrl + 鼠标拖动** 就能移动它（Ctrl + 右键
拖动是旋转）。脚本每帧用阻尼最小二乘 IK 解出关节角，再通过位置伺服把手臂开过去。
末端上有个绿色小球，它和目标红球之间的差距就是当前跟踪误差。

    /home/wck/miniconda3/bin/python ik_control.py

键盘微调（拖拽不好精确控制时用）：
    方向键      目标在 x / y 方向平移 1 cm
    Page Up/Down  目标在 z 方向平移 1 cm
    Q / E       目标绕世界 z 轴转 5°
    R           目标复位到末端当前位姿
    ESC         退出
"""

import math
import os
import time
from pathlib import Path

# ── 强制禁用输入法，必须在任何 X11 / GLFW 连接建立之前 ────────
# 否则中文输入法（ibus/fcitx）会抢走窗口的键盘事件：按 Q/E/R 想操作目标，结果弹出
# 输入法候选框，按键根本没送到窗口。做法抄自 ../ball/simulate.py。
os.environ["XMODIFIERS"] = "@im=none"
os.environ["GTK_IM_MODULE"] = "none"
os.environ["QT_IM_MODULE"] = "none"
os.environ["QT4_IM_MODULE"] = "none"

import numpy as np
import mujoco
import mujoco.viewer

try:
    import glfw
except ImportError:
    glfw = None

HERE = Path(__file__).resolve().parent
XML = HERE / "URDF_arm.xml"
EE_BODY = "Link6"

STEP_POS = 0.01      # 每次按键平移 [m]
STEP_ROT = math.radians(5.0)

# IK 参数
IK_LAMBDA = 0.05      # 阻尼系数，太小会在奇异位形附近震荡，太大收敛慢
# 每帧的关节增量上限 [rad]，500 Hz 下约 0.75 rad/s。
#
# 这个值不能大。实测扫过 0.0005 / 0.001 / 0.002 / 0.004 / 0.010：
#   0.001 -> 稳态 5.1 mm（最稳）    0.002 -> 10.4 mm（开始抖）
#   0.004 -> 20.5 mm                0.010 -> 26.9 mm（抖得厉害）
# 取 0.0015 是稳和快的折中。**不要**改成「离得远用大档、接近了用小档」那种双档切换：
# 阈值处 10 倍的增益突变会形成极限环，实测手臂会一路发散出去（末端跑到目标外 3 m）。
IK_STEP = 0.001
IK_LEAD = 0.20        # 指令角最多能超前实际关节角多少 [rad]，抗积分饱和用

# 任务空间阻尼（D 项）系数，量纲是秒。没有这一项时外环是个**纯积分器**，而纯积分器
# 驱动带滞后的被控对象必然产生极限环 —— 实测手臂停下来后会以 0.58 Hz 一直前后晃
# （抓屏连拍测出来的，主频 0.581 Hz；36 秒后位置误差仍有 1.44 mm、|qvel| 0.10 rad/s）。
#
# 这一项让误差不是被直接积分，而是先减去当前的任务空间速度：
#     dq = J† (err − IK_KD · J·qvel)
# J·qvel 的量纲是 (m/s, rad/s)，乘上秒后和 err 的 (m, rad) 可以直接相减。
# **别写成 `IK_KD * qvel`** —— 那样少了雅可比和量纲换算，实测(0.1·qvel)会把手臂推飞
# 400 mm、根本到不了目标。
#
# 实测（静止目标，30 s，测最后 14 s）：
#     IK_KD=0   位置误差均值 2.43 mm、ptp 3.80 mm、|qvel| 0.112 rad/s，永不停止
#     IK_KD=0.05 收敛到 0.000000 mm、|qvel| 恰好 0，最终关节角与 IK_KD=0 的解几乎相同
#                （不是卡死，是同一个位形上真停住了）
# 扫过 0.02~0.5 静止时全部收敛；跟踪场景 0.02 位置最好、0.05 以上姿态几乎无误差。
IK_KD = 0.05


# ---------------------------------------------------------------- 四元数工具
# 注意：MuJoCo 的四元数存储顺序是 (w, x, y, z)，和 Isaac Gym 的 (x, y, z, w) 不一样。
# 按后者写会让相对旋转整个算错 —— 表现是姿态误差恒定卡在 180° 附近（伺服一直往反方向
# 推），而「q 与自己的共轭相乘得单位元」这类自检却能通过，因为那和顺序约定无关。
def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([[math.cos(angle / 2.0)], axis * math.sin(angle / 2.0)])


def quat_to_rotvec(q):
    """四元数 (w,x,y,z) -> 旋转向量（轴 * 角）。q 和 -q 是同一旋转，先归一到 w >= 0。"""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    w = min(1.0, max(-1.0, q[0]))
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(0.0, 1.0 - w * w))
    if s < 1e-9:
        return np.zeros(3)
    return q[1:] / s * angle


# ---------------------------------------------------------------- IK
def ik_step(model, ik_data, target_pos, target_quat, ee_id, q_actual, qvel,
            q_cmd, lo, hi):
    """一步阻尼最小二乘笛卡尔伺服，返回 (新的关节指令角, 位姿误差范数)。

    误差和雅可比用 **实际** 关节角 q_actual 算，但增量 **累加** 到指令角 q_cmd 上。

    这两个必须分开，否则会有一个消不掉的稳态误差：位置伺服在重力下要靠
    `kp * (ctrl - qpos) = 重力力矩` 才能撑住，实测 kp=100 时 |ctrl - qpos| 恒为
    0.03 rad 左右。如果每帧都从 q_actual 重新算 `ctrl = q_actual + dq`，就是个纯比例
    环，那个偏差永远存在，末端会稳定停在离目标 8 cm 的地方（实测）。累加到指令上
    相当于加了积分作用，偏差会被慢慢消掉。

    也刻意不做「迭代到收敛的 IK 求解器」，那两种写法都试过且都失败：
      - 纯 DLS 直接加到 qpos 上，单步 dq 太大会震荡（Joint1 一步从 0 跳到 0.9、
        Joint5 顶到限位，姿态误差在 1~3 rad 之间来回摆）
      - 加回溯线搜索，又容易找不到下降方向而提前卡住（位置误差反而涨到 377 mm）
    摊成每帧一小步就稳了，这也是 Isaac Gym 里 franka_cube_ik_osc.py 的 control_ik 写法。
    目标不可达时不会发散，只会停在离目标最近的地方。
    """
    n = model.njnt
    ik_data.qpos[:n] = q_actual
    mujoco.mj_kinematics(model, ik_data)
    mujoco.mj_comPos(model, ik_data)

    err = np.concatenate([
        target_pos - ik_data.xpos[ee_id],
        quat_to_rotvec(quat_mul(target_quat, quat_conj(ik_data.xquat[ee_id]))),
    ])

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, ik_data, jacp, jacr, ee_id)
    J = np.vstack([jacp[:, :n], jacr[:, :n]])

    err_norm = float(np.linalg.norm(err))

    # D 项：先减去当前任务空间速度再求逆解，见 IK_KD 处的说明
    dq = J.T @ np.linalg.solve(J @ J.T + IK_LAMBDA ** 2 * np.eye(6),
                               err - IK_KD * (J @ qvel))
    dq = np.clip(dq, -IK_STEP, IK_STEP)             # 每帧最多走这么多，防震荡

    q_new = np.clip(q_cmd + dq, lo, hi)
    # 抗积分饱和：指令角最多超前实际关节角 IK_LEAD。没有这一条的话，伺服跑在 500 Hz
    # 上指令会越积越远（实测 |ctrl-qpos| 一路涨到 3.96 rad），机械臂跟不上就彻底失控。
    q_new = np.clip(q_new, q_actual - IK_LEAD, q_actual + IK_LEAD)
    return q_new, err_norm


def main():
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)      # IK 迭代会反复改 qpos，用独立的一份，别污染仿真状态

    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ik_target") < 0:
        raise SystemExit("模型里没有 mocap 目标，先重新生成：python build_model.py")
    if model.nmocap < 1:
        raise SystemExit("模型里没有 mocap 刚体，先重新生成：python build_model.py")

    lo = model.jnt_range[:, 0].copy()
    hi = model.jnt_range[:, 1].copy()

    # 初始姿态用零位（Joint3 的零位在行程外，夹一下）
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    # 目标初始位姿：放在机械臂外面，一打开就能看见、能用鼠标抓到。
    # 位置离臂体最近距离 0.40 m、距基座 0.52 m（可达范围 0.138~0.835 m）。
    # 零位时臂体占的是 +y 侧，所以放 -y 侧不会和模型重叠。
    # 姿态直接用末端当前的（零位姿态）—— 手臂本来就是这个姿态，开局不用转，
    # 实测位置误差 2.3 mm、姿态误差 2.1°。
    # 试过把姿态也换成「反查出的某组构型」的完整位姿（那样位置和姿态都严格可达），
    # 结果位置虽然到了 4.3 mm，姿态却差 23° —— 是 6 维位姿在雅可比伺服下的折中问题。
    data.mocap_pos[0] = np.array([0.3017, -0.2967, 0.3120])
    data.mocap_quat[0] = data.xquat[ee_id].copy()

    print(__doc__)
    print("末端位置 %s" % np.round(data.xpos[ee_id], 3))
    if glfw is None:
        print("（没装 glfw，键盘微调不可用，只能用鼠标拖动目标）")

    def key_callback(keycode):
        pos = data.mocap_pos[0].copy()
        quat = data.mocap_quat[0].copy()
        if keycode == glfw.KEY_LEFT:
            pos[0] -= STEP_POS
        elif keycode == glfw.KEY_RIGHT:
            pos[0] += STEP_POS
        elif keycode == glfw.KEY_DOWN:
            pos[1] -= STEP_POS
        elif keycode == glfw.KEY_UP:
            pos[1] += STEP_POS
        elif keycode == glfw.KEY_PAGE_DOWN:
            pos[2] -= STEP_POS
        elif keycode == glfw.KEY_PAGE_UP:
            pos[2] += STEP_POS
        elif keycode == glfw.KEY_Q:
            quat = quat_mul(quat_from_axis_angle([0, 0, 1], +STEP_ROT), quat)
        elif keycode == glfw.KEY_E:
            quat = quat_mul(quat_from_axis_angle([0, 0, 1], -STEP_ROT), quat)
        elif keycode == glfw.KEY_R:
            pos = data.xpos[ee_id].copy()
            quat = data.xquat[ee_id].copy()
        else:
            return
        data.mocap_pos[0] = pos
        data.mocap_quat[0] = quat

    dt = model.opt.timestep
    last_print = 0.0
    q_cmd = data.qpos[:model.njnt].copy()   # 指令角单独维护，见 ik_step 的说明

    with mujoco.viewer.launch_passive(
            model, data, key_callback=None if glfw is None else key_callback) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [0.0, 0.15, 0.3]
        viewer.cam.distance = 1.1
        viewer.cam.azimuth = -60.0
        viewer.cam.elevation = 18.0

        while viewer.is_running():
            t0 = time.time()
            target_pos = data.mocap_pos[0].copy()
            target_quat = data.mocap_quat[0].copy()

            # 误差用实际关节角算，增量累加到指令角上
            q_cmd, err = ik_step(model, ik_data, target_pos, target_quat, ee_id,
                                 data.qpos[:model.njnt].copy(),
                                 data.qvel[:model.njnt].copy(), q_cmd, lo, hi)
            data.ctrl[:] = q_cmd
            mujoco.mj_step(model, data)
            viewer.sync()

            slack = dt - (time.time() - t0)
            if slack > 0:
                time.sleep(slack)

            if time.time() - last_print > 1.0:
                last_print = time.time()
                print("目标 %s  末端 %s  残差 %.4f"
                      % (np.round(target_pos, 3), np.round(data.xpos[ee_id], 3), err),
                      end="\r", flush=True)

    print("\n已退出")


if __name__ == "__main__":
    main()
