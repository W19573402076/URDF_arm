"""仿真实体抽象 + MuJoCo 实现。

`RobotBackend` 是「仿真器无关」那条边界的本体：策略、观测构造、控制律都只依赖这个协议，
不依赖 MuJoCo。上真机时照着这个协议写一个实物后端即可，其余代码原样带走。

MuJoCo 实现里有两件事是**运行时改造**，不改 `../arm_mujoco/URDF_arm.xml`
（那个文件被 `ik_control.py` 的位置伺服依赖着）：

  1. 把 `<position>` 位置伺服改成纯力矩执行器
  2. 把物理步长从 0.002 改成 1/120，对齐 Isaac 训练时的 120 Hz

照做的原因和踩过的坑见各自函数的注释。
"""

from typing import Optional, Protocol, Tuple

import mujoco
import numpy as np

import config as C
from kinematics import wxyz_to_xyzw, xyzw_to_wxyz


class RobotBackend(Protocol):
    """真机 / 仿真都要满足的接口。刻意用 Protocol 而不是基类 ——
    真机不该被仿真语义绑架（比如 `reset` 在真机上可能是"等机械臂走到初始位姿"）。"""

    n_dof: int

    def reset(self, q: np.ndarray, qd: np.ndarray) -> None: ...
    def get_joint_state(self) -> Tuple[np.ndarray, np.ndarray]: ...
    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]: ...
    def set_joint_torques(self, tau: np.ndarray) -> None: ...
    def step(self) -> None: ...
    def joint_limits(self) -> Tuple[np.ndarray, np.ndarray]: ...
    def ee_pose_from_q(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: ...
    def gravity_torque(self) -> Optional[np.ndarray]: ...


class MuJoCoBackend:
    """MuJoCo 后端的实现。"""

    EE_BODY = "Link6"

    def __init__(self, xml_path, vel_limit=None, ground_collision=False,
                 integrator=None, exact_timestep=True):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.model.opt.timestep = C.PHYS_DT if exact_timestep else self.model.opt.timestep
        if integrator is not None:
            self.model.opt.integrator = integrator

        self._to_torque_actuators()
        if ground_collision:
            self._enable_ground_collision_only()

        self.data = mujoco.MjData(self.model)
        # 只给正解和重力前馈用，**永不 step** —— 这样绝不污染正在推进的仿真状态
        self.scratch = mujoco.MjData(self.model)

        self.n_dof = self.model.njnt
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.EE_BODY)
        if self.ee_id < 0:
            raise ValueError("模型里找不到 %s" % self.EE_BODY)

        self.vel_limit = vel_limit

    # ------------------------------------------------------------ 运行时改造

    def _to_torque_actuators(self):
        """把 `<position>` 位置伺服变成纯力矩执行器：force = ctrl。

        位置伺服的实质是 general actuator，`force = gain*ctrl + bias`，其中
        `gain = kp`、`bias = -kp*qpos - kv*qvel`。把它改成 gain=1、bias=0 就退化成
        纯力矩源。力矩上限仍由执行器的 forcerange 和关节的 actuatorfrcrange 保证
        （实测两者都是 [20,20,20,3,3,3]）。

        ⚠️ `actuator_ctrlrange` 必须一起清掉。XML 里位置伺服的 ctrlrange 被设成了
        **关节行程**，不清的话 ctrl 会被当成目标角度钳制 —— 实测 `ctrl=3.0` 会被
        削成 `2.79`（Joint3 的上限），力矩就错了。
        """
        self.model.actuator_gainprm[:, 0] = 1.0
        self.model.actuator_biasprm[:, :3] = 0.0
        self.model.actuator_ctrlrange[:] = 0.0
        self.model.actuator_ctrllimited[:] = 0

    def _enable_ground_collision_only(self):
        """开启「手臂 ↔ 地面」碰撞，但保持自碰撞关闭 —— 对齐 Isaac 训练时的设置
        （`self_collisions = 1` 关自碰，地面碰撞是开的）。

        用 contype/conaffinity 位掩码实现，不动 XML：
          手臂 geom: contype=2, conaffinity=0
          地面 geom: contype=1, conaffinity=3
        手-地: (2&3) 非零 → 碰；  手-手: (2&0) 与 (2&0) 都是 0 → 不碰。
        """
        ground = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        mocap_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ik_target")
        for g in range(self.model.ngeom):
            if g == ground:
                self.model.geom_contype[g] = 1
                self.model.geom_conaffinity[g] = 3
            elif self.model.geom_bodyid[g] == mocap_body:
                # 目标标记球是纯可视化用的，千万别让它参与碰撞 ——
                # 漏掉这一条的话它会和地面碰（(2 & 3) ≠ 0），白白给求解器加接触。
                self.model.geom_contype[g] = 0
                self.model.geom_conaffinity[g] = 0
            else:
                self.model.geom_contype[g] = 2
                self.model.geom_conaffinity[g] = 0

    # ------------------------------------------------------------ 接口

    def reset(self, q, qd):
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[:self.n_dof] = q
        self.data.qvel[:self.n_dof] = qd
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def get_joint_state(self):
        return (self.data.qpos[:self.n_dof].copy(),
                self.data.qvel[:self.n_dof].copy())

    def get_ee_pose(self):
        """(基座系位置, xyzw 四元数)。

        用 `xpos` 而**不是** `xipos` —— 前者是刚体坐标系原点，后者是质心，零位
        两者差 94.6 mm，远超 2 cm 的达标阈值，用错观测全废。
        MuJoCo 的基座焊在 world 原点，所以世界坐标就是基座系坐标。
        """
        return (self.data.xpos[self.ee_id].copy(),
                wxyz_to_xyzw(self.data.xquat[self.ee_id]))

    def set_joint_torques(self, tau):
        self.data.ctrl[:] = tau

    def step(self):
        mujoco.mj_step(self.model, self.data)
        if self.vel_limit is not None:
            # Isaac 侧的关节速度被 URDF 的 velocity="3.14" 硬限幅，MuJoCo 默认没有。
            # 这个开关只是复刻那个约束，默认关闭（见 sim2sim.py 的 --vel-limit）。
            np.clip(self.data.qvel[:self.n_dof], -self.vel_limit, self.vel_limit,
                    out=self.data.qvel[:self.n_dof])

    def joint_limits(self):
        return (self.model.jnt_range[:, 0].copy(), self.model.jnt_range[:, 1].copy())

    def ee_pose_from_q(self, q):
        """正解，**不污染仿真状态**（走 scratch）。目标采样和调试都用它。"""
        s = self.scratch
        s.qpos[:] = 0.0
        s.qpos[:self.n_dof] = q
        s.qvel[:] = 0.0
        mujoco.mj_forward(self.model, s)
        return (s.xpos[self.ee_id].copy(), wxyz_to_xyzw(s.xquat[self.ee_id]))

    # ------------------------------------------------------------ 可视化（非协议的一部分）

    def set_target_marker(self, pos, quat_xyzw):
        """把场景里那颗目标标记球挪到目标位姿。**纯可视化，不影响动力学**
        （mocap 刚体是运动学的，且它的 geom 关掉了碰撞）。

        标记球就是 `ik_control.py` 用的那个 `ik_target`：半透明红球 + 三色坐标轴，
        轴朝向即目标姿态。没有 mocap 刚体时静默跳过。
        """
        if self.model.nmocap < 1:
            return
        self.data.mocap_pos[0] = np.asarray(pos, dtype=float)
        self.data.mocap_quat[0] = xyzw_to_wxyz(np.asarray(quat_xyzw, dtype=float))

    def gravity_torque(self):
        """重力前馈 τ_g = -Σᵢ J_comᵢᵀ(mᵢ·g)，**只含重力不含科氏**。

        做法是 `qfrc_bias` 在 qvel=0 时的值 —— 此时科氏/离心项为 0，剩下的就是纯重力。
        实测它与 Isaac 的 `-Σ J^T(m g)` **逐点完全相等**（见 verify.py 的第 4 项）。

        ⚠️ 不能直接读 `data.qfrc_bias`（带速度时含科氏，而 Isaac 没补偿那部分），
        也不能在 live data 上把 qvel 清零再还原（容易漏还原），所以走 scratch。
        """
        s = self.scratch
        s.qpos[:] = self.data.qpos
        s.qvel[:] = 0.0
        mujoco.mj_forward(self.model, s)
        return s.qfrc_bias[:self.n_dof].copy()
