#!/usr/bin/env python3
"""用 Isaac 侧导出的 golden trace 对拍移植代码。跑在 conda base。

    /home/wck/mujocoproject/URDF/arm_mujoco_rl/tools/dump_isaac_trace.py --out trace.npz
    /home/wck/miniconda3/bin/python verify.py --trace trace.npz

四项检查，任何一项不过都说明映射有问题，**不要往下做端到端**：

  1. FK      —— MuJoCo 正解 vs Isaac 刚体状态（用积分后的关节角 q_after 比，才严格）
  2. 观测    —— kinematics.build_obs 重放，vs trace 里 Isaac 实际喂给网络的 obs
  3. 动作    —— policy.Policy.act，vs trace 里的网络输出
  4. 力矩    —— controller.compute_torques，vs trace 里的力矩

第 2 项最有价值：它一次能抓出四元数顺序、rot6d 错位、符号、缩放、切片错位的所有 bug。
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import config as C            # noqa: E402
import controller             # noqa: E402
import kinematics             # noqa: E402
from policy import Policy     # noqa: E402

XML_DEFAULT = os.path.join(os.path.dirname(_HERE), "arm_mujoco", "URDF_arm.xml")

# actor 对全零 obs 的输出，写死当回归锚点（首轮实测值）
ZERO_OBS_EXPECTED = np.array([-0.6064, -0.6507, -3.1281, -1.1828, -1.3756, 1.0162])


def check(tag, got, want, tol, unit=""):
    d = np.abs(np.asarray(got) - np.asarray(want))
    mx, mean = float(d.max()), float(d.mean())
    ok = mx <= tol
    print("  %-28s max %.3e  mean %.3e  %s  (判据 %.0e%s)"
          % (tag, mx, mean, "通过 ✓" if ok else "**不通过 ✗**", tol, unit))
    return ok


def check_fk(tr, xml):
    import mujoco
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    ee = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Link6")

    q = tr["fk_q_after"]
    P, Q = tr["fk_pos"], tr["fk_quat"]          # Q 是 xyzw
    dp = np.zeros(len(q))
    dq = np.zeros(len(q))
    dcom = np.zeros(len(q))
    for i in range(len(q)):
        d.qpos[:6] = q[i]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        p = d.xpos[ee]
        quat = d.xquat[ee][[1, 2, 3, 0]]
        if np.dot(quat, Q[i]) < 0:
            quat = -quat
        dp[i] = np.linalg.norm(p - P[i])
        dcom[i] = np.linalg.norm(d.xipos[ee] - P[i])
        dq[i] = 2 * np.arccos(np.clip(abs(np.dot(quat, Q[i])), 0, 1))

    ok1 = check("FK 位置 [m]", dp, np.zeros_like(dp), 1e-4)
    ok2 = check("FK 姿态 [rad]", dq, np.zeros_like(dq), 1e-3)
    print("      （对照：误用质心 xipos 的位置误差 max %.3e m）" % dcom.max())
    return ok1 and ok2


def check_obs(tr):
    """重放观测构造。这是最关键的一项。"""
    n = len(tr["step_q"])
    got = np.zeros((n, C.NUM_OBS))
    for t in range(n):
        got[t] = kinematics.build_obs(
            tr["step_q"][t], tr["step_qd"][t],
            tr["step_ee_pos"][t], tr["step_ee_quat"][t],
            tr["step_target_pos"][t], tr["step_target_quat"][t],
            tr["step_last_action"][t])
    want = tr["step_obs"]
    ok = check("观测 (30 维)", got, want, 1e-5)

    # 不通过时按切片给线索，直接指出是哪一段错了
    if not ok:
        d = np.abs(got - want)
        for nm, sl in (("[ 0: 6] 关节角偏差", slice(0, 6)),
                       ("[ 6:12] 关节角速度", slice(6, 12)),
                       ("[12:15] 位置误差",   slice(12, 15)),
                       ("[15:21] rot6d",      slice(15, 21)),
                       ("[21:24] 末端位置",   slice(21, 24)),
                       ("[24:30] 上一步动作", slice(24, 30))):
            print("        %-20s max %.3e" % (nm, d[:, sl].max()))
    return ok


def check_policy(tr, ckpt):
    p = Policy.from_checkpoint(ckpt)
    z = p.act(np.zeros(C.NUM_OBS))
    ok0 = check("actor(zeros(30)) 回归锚点", z, ZERO_OBS_EXPECTED, 1e-3)

    n = len(tr["step_obs"])
    got = p.act(tr["step_obs"])
    ok1 = check("网络输出", got, tr["step_action_raw"], 1e-5)
    return ok0 and ok1


def check_torques(tr):
    n = len(tr["step_action_clipped"])
    got = np.zeros((n, 6))
    for t in range(n):
        got[t] = controller.compute_torques(
            tr["step_action_clipped"][t],
            tr["step_tau_q"][t], tr["step_tau_qd"][t],
            tr["step_gravity_ff"][t])
    return check("力矩（双重裁剪+前馈）", got, tr["step_tau_out"], 1e-5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="isaac_trace.npz")
    ap.add_argument("--checkpoint",
                    default=os.path.join(os.path.dirname(_HERE), "arm_rl", "logs",
                                         "Sep19_19-10-57_", "model_20000.pt"))
    ap.add_argument("--xml", default=XML_DEFAULT)
    a = ap.parse_args()

    if not os.path.exists(a.trace):
        sys.exit("找不到 trace 文件 %s\n先在 legged_gym 环境里跑 tools/dump_isaac_trace.py"
                 % a.trace)
    tr = np.load(a.trace)
    print("trace: %s" % a.trace)
    print("  FK 样本 %d 组,  episode %d 步\n" % (len(tr["fk_q_after"]), len(tr["step_q"])))

    results = {}
    print("[1/4] FK 对拍（MuJoCo 正解 vs Isaac 刚体状态）")
    results["fk"] = check_fk(tr, a.xml)
    print("\n[2/4] 观测构造")
    results["obs"] = check_obs(tr)
    print("\n[3/4] 策略网络")
    results["policy"] = check_policy(tr, a.checkpoint)
    print("\n[4/4] 力矩链路")
    results["torques"] = check_torques(tr)

    print("\n" + "=" * 62)
    bad = [k for k, v in results.items() if not v]
    if bad:
        print("  未通过: %s" % ", ".join(bad))
        print("  排查顺序见计划文件「不达标时的排查顺序」一节")
    else:
        print("  四项全部通过 —— 映射链路是干净的，可以做端到端了")
    print("=" * 62)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
