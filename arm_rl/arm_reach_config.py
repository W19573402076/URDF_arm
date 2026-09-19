"""末端 6D 位姿到达任务的配置。

继承 legged_gym 的 LeggedRobotCfg / LeggedRobotCfgPPO，只覆盖机械臂需要的部分。
腿足相关的东西（地形课程、速度指令、base height、足端接触）全部关掉或改成机械臂语义。

配置的分组语义和覆盖方式是 legged_gym 的约定：内嵌 class 继承父类的同名 class 再改字段，
BaseConfig.__init__ 会把每个内嵌 class 实例化成对象，所以 cfg.control.stiffness 这种点号
访问才成立。
"""

import os

# arm_rl/ 的上一级是 URDF/，机械臂资产在 arm_isaacgym/ 里（与查看器共用同一份）
_HERE = os.path.dirname(os.path.abspath(__file__))
_URDF_DIR = os.path.dirname(_HERE)
ASSET_FILE = os.path.join(_URDF_DIR, "arm_isaacgym", "assets", "urdf", "URDF_arm.urdf")

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class ArmReachCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        # 观测 30 维，见 arm_reach_env.ArmReach.compute_observations
        num_observations = 30
        num_privileged_obs = None
        num_actions = 6
        env_spacing = 2.0
        send_timeouts = True
        episode_length_s = 5.0  # 30 Hz 控制 -> 150 个控制步

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'      # 固定基座，平地就够
        curriculum = False
        measure_heights = False  # 关掉高度采样，基类就不会建 height_points

    class commands(LeggedRobotCfg.commands):
        # 这里复用了 commands 这个缓冲区来存末端目标位姿：前 3 维位置，后 4 维四元数 (x,y,z,w)。
        # cfg.commands.num_commands 决定 self.commands 的列数。
        num_commands = 7
        curriculum = False
        heading_command = False
        resampling_time = 1e9    # 实际不起作用：env 里覆写了 _post_physics_step_callback

    class target:
        """目标位姿采样的约束。目标由随机关节角正解生成，这些阈值用来剔除
        物理上够不到或会撞地面的目标。"""
        joint_margin = 0.1      # 采样关节角时相对行程内缩的比例，避免目标正好落在限位
        min_z = 0.05            # 末端最低高度 [m]，低于这个值会撞地面
        max_radius = 0.65       # 末端距基座的最大水平距离 [m]
        max_tries = 8           # 拒绝采样最多重试几次

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.0]
        rot = [0.0, 0.0, 0.0, 1.0]
        # action = 0 时 PD 的目标角。零位必须是 6 个关节全给，
        # 基类 _init_buffers 会逐个 dof 去查这个字典。
        # Joint3 的零位 0 不在它的行程 [0.35, 2.79] 内，只能取下限 0.35。
        default_joint_angles = {
            "Joint1": 0.0,
            "Joint2": 0.0,
            "Joint3": 0.35,   # 行程下限，不是 0
            "Joint4": 0.0,
            "Joint5": 0.0,
            "Joint6": 0.0,
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        # 扭矩 = p_gains*(action_scale*action + default - dof_pos) - d_gains*dof_vel
        #
        # kp 是试出来的，不是越大越好，这一点反直觉所以记下来。
        #
        # PD 在重力下必然有稳态误差 err = 重力力矩 / kp（Joint2 最大约 15 N·m），
        # 单看这一点应该 kp 越大误差越小 —— 在 MuJoCo 里测也确实如此
        # （kp=400 时稳态末端误差只有 10 mm，kp=80 时是 55 mm）。
        # 但拿到 Isaac Gym 里训练，kp=400 明显更差：
        #     kp 80/20  -> 成功率 7.7%，位置误差中位数 33.8 mm
        #     kp 400/100 -> 成功率 1.0%，位置误差中位数 69.7 mm
        # 原因是力矩上限只有 URDF 给的 20 N·m。kp=400 时只要位置超前 0.05 rad 就吃满
        # 力矩，动作空间长期处于饱和，策略失去了对力矩的精细调节能力。kp=80 时同样
        # 的力矩对应 0.25 rad 的窗口，可控性好得多 —— 而重力下沉是**确定性**的
        # （观测里给了完整关节角），策略可以学着补偿，实测补偿后误差比纯下沉还小。
        #
        # 腕关节（Joint4/5/6）的 URDF effort 只有 3 N·m，kp 按比例取小。
        stiffness = {
            "Joint1": 80.0, "Joint2": 80.0, "Joint3": 80.0,
            "Joint4": 20.0, "Joint5": 20.0, "Joint6": 20.0,
        }
        damping = {
            "Joint1": 4.0, "Joint2": 4.0, "Joint3": 4.0,
            "Joint4": 1.0, "Joint5": 1.0, "Joint6": 1.0,
        }
        action_scale = 0.25
        decimation = 4           # 120 Hz 物理 / 4 = 30 Hz 控制

    class asset(LeggedRobotCfg.asset):
        file = ASSET_FILE
        name = "URDF_arm"
        foot_name = "None"       # 机械臂没有足端，给个不会匹配到任何刚体的名字
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True
        fix_base_link = True     # 固定在基座上
        default_dof_drive_mode = 3   # DOF_MODE_EFFORT，力矩控制
        # 1 = 关掉自碰撞。七个连杆的 CAD 网格在关节处是互相穿插的（MuJoCo 侧实测零位
        # 就有 6 处穿透接触，最深 2.5 cm），自己碰自己会产生持续的接触力，训练没法收敛。
        # 这个值会传给 gym.create_actor(..., collision_filter=...)，同一 actor 内的
        # 形状用同一位掩码，互相不碰。
        self_collisions = 1
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False   # STL 不需要翻转
        # 电机转子折合到关节上的惯量。URDF 没有这个概念，而腕关节绕轴惯量只有 1e-4 量级，
        # 加 0.01 能明显改善数值稳定性。arm_mujoco 那边也是这个值。
        armature = 0.01

    class domain_rand(LeggedRobotCfg.domain_rand):
        # 纯仿真训练，不做域随机化
        randomize_friction = False
        randomize_base_mass = False
        push_robots = False

    class rewards(LeggedRobotCfg.rewards):
        class scales(LeggedRobotCfg.rewards.scales):
            # 基类 _prepare_reward_function 会把非零 scale 自动乘以 self.dt（= 1/30 s），
            # 所以这里的数值是「每秒」量级，实际每步贡献要除以 30。
            termination = 0.0            # 不用终止奖励
            # 粗定位项：分开写，负责在离目标还远的时候提供全局梯度信号。
            tracking_eef_pos = 2.0
            tracking_eef_ori = 2.0
            # 精定位项：**合成一个联合核**，而不是位置一个、姿态一个。
            #
            # 这是针对「策略在位置和姿态之间互相妥协」这个核心问题改的。之前精定位是
            # 两项相加：w_p·exp(-e_p²/σ_p²) + w_o·exp(-e_o²/σ_o²)，策略优化的是加权和；
            # 但达标判据是逻辑与。两者不等价 —— 一项差另一项好，加权和照样高，所以策略
            # 会一头换另一头。实测六次训练成功率都卡在 7~8%，位置和姿态中位数在剧烈互换
            # （v4：位置 26 mm/姿态 14.5°；v5：位置 62 mm/姿态 4.7°），就是这个原因。
            #
            # 联合核 exp(-(e_p/σ_p)² - (e_o/σ_o)²) 里两项是指数上的加，只有**两者都小**
            # 才拿得到高分，结构上就没法用一头换另一头。σ 取了和原来两个精核一样的值，
            # 权重 2.5 = 原来两项之和，最大贡献不变。
            tracking_eef_both_fine = 2.5
            # 位置误差的线性项，梯度恒定，补上指数核在近目标处梯度消失的问题。
            # 误差 0.5 m 时每步贡献约 -0.5*0.5/30 = -0.008，与指数项的 0.024 同量级。
            eef_pos_lin = -0.5
            action_rate = -0.01
            dof_vel = -0.001
            torques = -1e-4
            dof_pos_limits = -2.0
            success = 5.0

            # 关掉基类里腿足专用的项（继承来的默认值非零，必须显式清零）
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            lin_vel_z = 0.0
            ang_vel_xy = 0.0
            orientation = 0.0
            dof_acc = 0.0
            base_height = 0.0
            feet_air_time = 0.0
            collision = 0.0
            feet_stumble = 0.0
            stand_still = 0.0

        only_positive_rewards = False   # 有负惩罚项，不能把总奖励 clip 到 0
        # 指数核的参数：reward = exp(-err^2 / sigma^2)
        #
        # 这两个值必须匹配实际误差尺度，否则梯度信号会消失。实测复位后立即的误差：
        # 位置 mean 0.447 m / p95 0.599 m，姿态 mean 2.20 rad / p95 3.02 rad。
        # 若用 sigma_pos=0.1、sigma_ori=0.3，奖励核的平均值只有 0.0019 / 0.0015，
        # 等于没有信号。下面这组值让复位时的平均奖励核在 0.15 / 0.06 量级，
        # 同时最大值仍接近 1，保留了精度方向上的梯度。
        # 粗定位核：覆盖整个工作空间，保证任何位置都有梯度信号。
        tracking_sigma_pos = 0.3        # 位置误差 [m]
        tracking_sigma_ori = 1.0        # 姿态误差 [rad]
        # 联合精核的两个尺度。只在近目标处起作用，负责把位置和姿态**一起**压下去。
        #
        # 为什么用「粗核 + 精核」而不是把粗核的 sigma 随着训练收窄（退火）：
        # 退火把 sigma_pos 收到 0.08 之后，误差 0.3 m 处的奖励是 exp(-14) ≈ 1e-6，
        # 绝大多数状态拿到的奖励都是 0，PPO 的价值函数学不动，训练反而退化
        # （实测成功率从 7.7% 掉到 1.3%）。两个核叠加就没有这个问题：粗核始终
        # 提供全局信号，精核只负责精度。
        tracking_sigma_pos_fine = 0.05
        tracking_sigma_ori_fine = 0.15
        # 判定「到达」的阈值，给 success 一次性奖励
        success_pos_tol = 0.02          # 2 cm
        success_ori_tol = 0.1           # 0.1 rad ≈ 5.7°
        # 关节限位惩罚的软阈值（行程的百分比）。这里必须给 1.0，不能收窄：
        # Joint3 的零位 0.35 正好是它的行程下限，若设成 0.95（下限变成 0.411），
        # 机械臂静止时就已经在惩罚区里，等于持续扣分 —— 实测有 13% 的关节位置
        # 处于软限位之外。给 1.0 就只在真正越界时惩罚。
        soft_dof_pos_limit = 1.0
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            dof_pos = 1.0
            dof_vel = 0.1
            eef_pos_err = 2.0       # 位置误差，放大一点好学习
            eef_pos = 1.0
            eef_rot = 1.0
        clip_observations = 100.0
        clip_actions = 100.0

    class noise(LeggedRobotCfg.noise):
        # 纯仿真训练没有 sim-to-real 需求。关掉噪声同时也就绕开了基类
        # _get_noise_scale_vec 里那套针对腿足观测结构的硬编码切片。
        add_noise = False

    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [1.6, -1.6, 1.2]
        lookat = [0.0, 0.0, 0.25]

    class sim(LeggedRobotCfg.sim):
        dt = 1.0 / 120.0          # 120 Hz 物理
        substeps = 1
        gravity = [0.0, 0.0, -9.81]
        up_axis = 1               # 1 = z 轴向上

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23
            default_buffer_size_multiplier = 5
            contact_collection = 2


class ArmReachCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        # 网络容量。之前用的是 [256, 128, 64]（比 legged_gym 默认的 [512,256,128] 小），
        # v7 之后判断容量可能是新瓶颈：「位置和姿态同时精确」本质是要求策略学到更精确的
        # 逆动力学模型，调奖励、调 kp、加重力补偿都只在 7~10% 附近晃，像是表达力不够。
        # 这次**只改容量、训练轮数保持 2000 不变**，做单变量对比（上次同时改 kp 和奖励核
        # 退火导致没法归因，是个教训）。
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'

    class algorithm(LeggedRobotCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.005
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.0e-3
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 2000
        save_interval = 100
        experiment_name = 'arm_reach'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
