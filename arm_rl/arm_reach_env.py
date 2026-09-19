"""末端 6D 位姿到达任务的环境。

继承 legged_gym 的 LeggedRobot 拿到现成的 sim 搭建、张量 API 接线、PD 力矩控制、
reset 骨架、按命名约定注册奖励、TensorBoard 集成；只覆写腿足专用或结构不同的部分。

被控对象是 Link6 的坐标系（源 URDF 没有工具尖，Link6 就是最后一级连杆）。
目标位姿由「随机采一组合法关节角 → 正解算出末端位姿」得到，这样目标 100% 可达且
不会超关节行程。
"""

# 必须先 import isaacgym 再 import torch，否则会报
# "PyTorch was imported before isaacgym modules"
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_conjugate, quat_mul, to_torch, torch_rand_float

import torch

from legged_gym.envs.base.legged_robot import LeggedRobot

from arm_reach_config import ArmReachCfg


def quat_to_rot6d(q):
    """四元数 (x,y,z,w) -> 旋转矩阵的前两列，拼成 6 维。

    用 6D 旋转表示而不是四元数或欧拉角：四元数有 ±q 双重覆盖（同一姿态两个表示），
    欧拉角有万向锁，都会让策略面对不连续的输入。6D 表示是连续的。
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    col0 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], dim=-1)
    col1 = torch.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], dim=-1)
    return torch.cat([col0, col1], dim=-1)


def quat_angle(q):
    """两个姿态之间的夹角（即相对旋转的旋转角），单位弧度。

    四元数取绝对值再 acos，因为 q 和 -q 表示同一个旋转，不取绝对值会在
    180° 附近跳变。
    """
    w = torch.abs(q[:, 3]).clamp(max=1.0)
    return 2.0 * torch.acos(w)


class ArmReach(LeggedRobot):

    # ------------------------------------------------------------------ 初始化

    def _process_dof_props(self, props, env_id):
        """基类会存下 dof_pos_limits，但会按 soft_dof_pos_limit 收缩成软限位。
        这里额外留一份原始的硬限位，采样目标关节角时要用。"""
        props = super()._process_dof_props(props, env_id)
        if env_id == 0:
            self.dof_pos_limits_raw = torch.zeros(
                self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits_raw[i, 0] = props["lower"][i].item()
                self.dof_pos_limits_raw[i, 1] = props["upper"][i].item()
        return props

    def _init_buffers(self):
        super()._init_buffers()

        # 末端（Link6）的位姿要从刚体状态张量里读，基类没有 acquire 这个张量。
        # 注意 self.dof_state 的原始形状是 (num_envs*num_dof, 2)，要 reshape 成三维才能
        # 按 env 索引 —— 基类只做了 view(...)[..., 0] 取出 dof_pos/dof_vel。
        self.dof_state_3d = self.dof_state.view(self.num_envs, self.num_dof, 2)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, 13)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.eef_index = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.actor_handles[0], "Link6")

        # 视图，refresh 之后自动反映新值
        self.ee_pos = self.rigid_body_states[:, self.eef_index, 0:3]
        self.ee_quat = self.rigid_body_states[:, self.eef_index, 3:7]

        # 前馈重力补偿用的雅可比张量。
        # 注意 jacobian 的刚体维度是 num_bodies-1（base_link 是固定基座，被并掉了），
        # jacobian[:, i] 对应 Link(i+1)，所以质量数组要从 props[1:] 取。
        # 这个雅可比是世界系、且参考点在刚体**质心**（用有限差分验证过），
        # 正好是算重力力矩需要的形式。
        jac = self.gym.acquire_jacobian_tensor(self.sim, self.cfg.asset.name)
        self.jacobian = gymtorch.wrap_tensor(jac)
        props = self.gym.get_actor_rigid_body_properties(self.envs[0], self.actor_handles[0])
        self.jac_body_masses = torch.tensor(
            [p.mass for p in props[1:]], dtype=torch.float, device=self.device)
        self.gravity_ff = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.gravity_world = to_torch(self.cfg.sim.gravity, device=self.device)
        # 基座系下的末端位置。刚体状态张量给的是世界坐标，含各 env 的网格偏移
        # （env_spacing 决定的 env_origins），必须减掉才和「目标-末端」的相对量一致，
        # 否则不同 env 的观测尺度会差几十米。基座固定且无旋转，所以减去原点就是基座系。
        self.ee_pos_rel = self.ee_pos - self.env_origins

        self.target_pos = self.commands[:, :3]
        self.target_quat = self.commands[:, 3:7]

        # success 奖励只在每个 episode 里给一次
        self._success_given = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 每个 episode 结束时的末端误差，给 play.py 做评估用。
        # 记在 reset_idx 里（复位之前那一瞬间取），所以是精确的最终误差。
        self.episode_eef_pos_err = torch.zeros(self.num_envs, device=self.device)
        self.episode_eef_ori_err = torch.zeros(self.num_envs, device=self.device)

    def _get_noise_scale_vec(self, cfg):
        """覆写基类：它的实现是针对腿足观测结构的硬编码切片，而且 [24:36] 会越过
        本任务 30 维观测的边界，静默截断。本任务默认关噪声，要开就得重新写尺度。"""
        self.add_noise = self.cfg.noise.add_noise
        if self.add_noise:
            raise NotImplementedError(
                "本任务的观测结构与基类不同，开噪声需要先按 compute_observations 重写噪声尺度")
        return torch.zeros(self.num_obs, device=self.device)

    # ------------------------------------------------------------------ 每一步

    def _post_physics_step_callback(self):
        """基类在这里做指令周期重采样、推搡机器人、地形课程。到达任务都不要。

        这里只把刚体状态张量刷新一下 —— 末端位姿每步都要用，而基类的
        post_physics_step 只刷新了 root state 和接触力。
        """
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.ee_pos_rel = self.ee_pos - self.env_origins
        self._update_gravity_ff()

    def _update_gravity_ff(self):
        """前馈重力补偿：τ_g = -Σ_i J_com_i^T · (m_i · g)。

        不加这一项的话，PD 在重力下有稳态误差 err = τ_g/kp，Joint2 在 kp=80 时对应
        约 0.19 rad 的关节偏差，会同时污染末端的位置和姿态 —— 实测策略只能把误差
        在两者之间分配，调奖励权重也只是把瓶颈从姿态换到位置。

        J 是「世界系 + 参考点在质心」的雅可比（有限差分验证过），所以 J^T·(m·g) 直接
        就是重力对广义坐标的力矩，取负号即保持力矩。公式和 MuJoCo 的 qfrc_bias 逐点
        对比过，数值完全吻合。

        每控制步算一次（在 _post_physics_step_callback 里调用），比 decimation 里的
        每个物理步算便宜。代价是有一控制步的滞后，对缓变的重力力矩无所谓。
        """
        self.gym.refresh_jacobian_tensors(self.sim)
        tau = torch.zeros_like(self.gravity_ff)
        for i in range(self.jacobian.shape[1]):
            f = (self.jac_body_masses[i] * self.gravity_world).view(1, 3, 1)
            tau += (self.jacobian[:, i, 0:3, :] * f).sum(dim=1)
        self.gravity_ff[:] = -tau

    def _compute_torques(self, actions):
        """基类的 PD + 前馈重力补偿，再一起按关节力矩上限截断。"""
        torques = super()._compute_torques(actions) + self.gravity_ff
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def check_termination(self):
        """只按超时终止。手臂固定在基座上不会摔，也没有接触终止。

        刻意不在到达目标时提前终止：让策略学会在目标位姿上保持住，
        而不是冲过去就结束。
        """
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf = self.time_out_buf.to(dtype=self.reset_buf.dtype)

    # ------------------------------------------------------------------ 复位

    def _reset_dofs(self, env_ids):
        """覆写基类：它的实现是 default_dof_pos * rand(0.5, 1.5)，对机械臂会越界 ——
        Joint3 的零位是 0.35（行程下限），乘 0.5 就掉到行程外面去了。
        改成在零位附近加小扰动再夹到行程内。"""
        n = len(env_ids)
        q = self.default_dof_pos + torch_rand_float(-0.1, 0.1, (n, self.num_dof), device=self.device)
        q = torch.max(torch.min(q, self.dof_pos_limits_raw[:, 1]), self.dof_pos_limits_raw[:, 0])
        self.dof_state_3d[env_ids, :, 0] = q
        self.dof_state_3d[env_ids, :, 1] = 0.0
        self._write_dof_states(env_ids)

    def _write_dof_states(self, env_ids):
        ids = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(ids), len(ids))

    def reset_idx(self, env_ids):
        """覆写基类：去掉地形课程、指令课程、base height 那套腿足逻辑。"""
        if len(env_ids) == 0:
            return

        # 在复位之前量一下末端离目标还有多远，作为评估指标。
        # 这时 _post_physics_step_callback 刚刷新过刚体状态，ee_pos_rel 还是本 episode
        # 结束时的值，commands 里也还是本 episode 的目标（_resample_commands 在下面才调）。
        pos_err = self._pos_error()[env_ids]
        ori_err = self._ori_error()[env_ids]
        reached = ((pos_err < self.cfg.rewards.success_pos_tol) &
                   (ori_err < self.cfg.rewards.success_ori_tol)).float()
        self.episode_eef_pos_err[env_ids] = pos_err
        self.episode_eef_ori_err[env_ids] = ori_err
        metrics = {
            "eef_pos_err_m": torch.mean(pos_err),
            "eef_ori_err_rad": torch.mean(ori_err),
            "eef_success_rate": torch.mean(reached),
        }

        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)

        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self._success_given[env_ids] = False

        self.extras["episode"] = dict(metrics)
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s)
            self.episode_sums[key][env_ids] = 0.0
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    # ------------------------------------------------------------------ 目标采样

    def _sample_joint_pos(self, n):
        """在关节行程内缩 joint_margin 的范围内均匀采样。"""
        m = self.cfg.target.joint_margin
        lo = self.dof_pos_limits_raw[:, 0]
        hi = self.dof_pos_limits_raw[:, 1]
        slo = lo + m * (hi - lo)
        shi = hi - m * (hi - lo)
        u = torch.rand(n, self.num_dof, device=self.device)
        return slo + u * (shi - slo)

    def _resample_commands(self, env_ids):
        """采一组合法关节角，正解算出末端位姿，当作本 episode 的目标。

        用正解生成而不是几何采样：关节行程是有限的（Joint2 只有 [-2.79, 0.35]，
        Joint3 只有 [0.35, 2.79]，Joint5 只有 ±1.57），几何采样出的姿态有相当比例
        物理上够不到，那些目标只会往奖励里灌噪声。正解生成天然保证目标可达。

        采完之后要做拒绝采样：随机关节角里有一部分会让末端低于地面或伸得太远，
        这些目标在有碰撞的仿真里够不到。
        """
        n = len(env_ids)
        if n == 0:
            return

        keep = self.dof_state_3d[env_ids].clone()   # (n, num_dof, 2) 复位后的状态，待会还原

        best_pos = torch.zeros(n, 3, device=self.device)
        best_quat = torch.zeros(n, 4, device=self.device)
        best_quat[:, 3] = 1.0
        best_score = torch.full((n,), -1e9, device=self.device)

        pending = torch.arange(n, device=self.device)
        for _ in range(self.cfg.target.max_tries):
            if len(pending) == 0:
                break
            rows = env_ids[pending]
            self.dof_state_3d[rows, :, 0] = self._sample_joint_pos(len(pending))
            self.dof_state_3d[rows, :, 1] = 0.0
            self._write_dof_states(rows)

            # 走一步物理，用引擎自己的正解算末端位姿。
            # 这一步会让所有 env 都前进一帧，对到达任务无关紧要。
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # 换算到基座系再做校验，世界坐标里带着几十米的 env 偏移
            pos = (self.rigid_body_states[rows, self.eef_index, 0:3]
                   - self.env_origins[rows]).clone()
            quat = self.rigid_body_states[rows, self.eef_index, 3:7].clone()

            score = pos[:, 2] - torch.norm(pos[:, :2], dim=1)   # 越高、越靠近基座越好
            better = score > best_score[pending]
            upd = pending[better]
            best_pos[upd] = pos[better]
            best_quat[upd] = quat[better]
            best_score[upd] = score[better]

            ok = (pos[:, 2] >= self.cfg.target.min_z) & \
                 (torch.norm(pos[:, :2], dim=1) <= self.cfg.target.max_radius)
            pending = pending[~ok]

        self.target_pos[env_ids] = best_pos
        self.target_quat[env_ids] = best_quat

        # 还原到复位时的关节状态
        self.dof_state_3d[env_ids] = keep
        self._write_dof_states(env_ids)
        self.gym.refresh_dof_state_tensor(self.sim)

    # ------------------------------------------------------------------ 观测

    def compute_observations(self):
        """30 维观测。见 arm_reach_config.ArmReachCfg.env.num_observations。"""
        q_rel = quat_mul(quat_conjugate(self.ee_quat), self.target_quat)
        self.obs_buf = torch.cat((
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,   # 6
            self.dof_vel * self.obs_scales.dof_vel,                            # 6
            (self.target_pos - self.ee_pos_rel) * self.obs_scales.eef_pos_err,  # 3
            quat_to_rot6d(q_rel) * self.obs_scales.eef_rot,                    # 6
            self.ee_pos_rel * self.obs_scales.eef_pos,                         # 3
            self.actions,                                                      # 6
        ), dim=-1)

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    # ------------------------------------------------------------------ 奖励

    def _pos_error(self):
        return torch.norm(self.target_pos - self.ee_pos_rel, dim=1)

    def _ori_error(self):
        return quat_angle(quat_mul(quat_conjugate(self.ee_quat), self.target_quat))

    def _reward_tracking_eef_pos(self):
        """粗定位：核宽 0.3 m，覆盖整个工作空间。"""
        err = self._pos_error()
        return torch.exp(-err ** 2 / self.cfg.rewards.tracking_sigma_pos ** 2)

    def _reward_tracking_eef_both_fine(self):
        """联合精定位：位置和姿态**同时**小才给高分。

        exp(-(e_p/σ_p)² - (e_o/σ_o)²) 里两项是指数上的加，任何一项大都会把整项压到 0，
        所以策略没法用「姿态好、位置差」去换「位置好、姿态差」——

        之前精定位写成位置一项、姿态一项相加，策略优化加权和，而达标判据是逻辑与，
        两者不等价，结果六次训练成功率始终卡在 7~8%，位置和姿态的中位数在剧烈互换。
        改成联合核就是为了堵掉这条路。
        """
        ep = self._pos_error()
        eo = self._ori_error()
        r = self.cfg.rewards
        return torch.exp(-ep ** 2 / r.tracking_sigma_pos_fine ** 2
                         - eo ** 2 / r.tracking_sigma_ori_fine ** 2)

    def _reward_eef_pos_lin(self):
        """位置误差的线性项（scale 为负，所以这是惩罚）。

        平方指数核在误差趋近 0 时梯度也为 0，精度区间里信号很弱 —— 误差 0.02 m 和
        0.1 m 对应的奖励只差 10%。线性项的梯度是常数，能一直把末端往目标拉，
        配合指数核做粗定位，才有机会收敛到 20 mm 的达标阈值。
        """
        return self._pos_error()

    def _reward_tracking_eef_ori(self):
        """粗定位：核宽 1.0 rad。"""
        err = self._ori_error()
        return torch.exp(-err ** 2 / self.cfg.rewards.tracking_sigma_ori ** 2)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_success(self):
        """到达目标的一次性奖励：每个 episode 最多给一次，避免策略学会了
        在目标附近来回蹭奖励。"""
        reached = (self._pos_error() < self.cfg.rewards.success_pos_tol) & \
                  (self._ori_error() < self.cfg.rewards.success_ori_tol)
        first_time = reached & (~self._success_given)
        self._success_given |= reached
        return first_time.float()
