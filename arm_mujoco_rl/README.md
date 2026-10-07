# 把 RL 策略从 Isaac Gym 移植到 MuJoCo（sim2sim）

把 `../arm_rl` 里用 PPO 训好的「末端 6D 位姿到达」策略（v13，20000 迭代）拿到 MuJoCo 里跑。

**目的不只是"验证一下"，而是上真机的前置。** 所以代码结构刻意分成两半：一半是
「读状态 → 算观测 → 跑网络 → 算力矩」的推理+控制模块（不依赖任何仿真器，上真机时原样带走），
另一半是 MuJoCo 验证台（真机上线时整个不用）。

```
              ┌─────────────── 可移植：上真机时带走 ───────────────┐
              │ config.py     常量（纯数据）                        │
              │ kinematics.py 四元数 + 观测构造 + 误差度量（纯 numpy）│
              │ policy.py     checkpoint → 确定性动作（纯 torch）     │
              │ controller.py 动作 → 关节力矩（纯 numpy）            │
              │ runner.py     episode 主循环 + 目标采样 + 指标       │
              └───────────────────────┬─────────────────────────────┘
                                      │ RobotBackend 协议（backend.py）
              ┌───────────────────────┴─────────────────────────────┐
              │ MuJoCoBackend（仿真）    real_backend（将来自己填）  │
              └─────────────────────────────────────────────────────┘
              sim2sim.py / verify.py / tools/  ← 纯验证台
```

## 结果

用 **200 个完全相同的场景**（由 Isaac 侧导出初始关节角和目标位姿）两边各跑一遍：

| | 位置中位 | 位置 p95 | 姿态中位 | 严格成功率 |
| --- | --- | --- | --- | --- |
| Isaac Gym | 8.1 mm | 28.7 mm | 0.96° | 90.0% |
| **MuJoCo** | **8.0 mm** | **22.2 mm** | **1.00°** | **92.0%** |
| 差值 | −0.1 mm | −6.5 mm | +0.04° | +2.0 pp |

**位置中位差 0.1 mm，成功率差 2 个百分点**（在 200 个样本的噪声范围内）。
自由分布下自采样 200 个 episode 的结果也一致：位置中位 7.7 mm、成功率 93.0%。

## 怎么跑

两个环境交替用：

```bash
cd /home/wck/mujocoproject/URDF/arm_mujoco_rl

# ① 在 legged_gym 环境里导对拍数据（要 isaacgym，且 ninja 要在 PATH 上）
conda activate legged_gym
python tools/dump_isaac_trace.py --out trace.npz --scenarios 200

# ② 回到 conda base 跑验证和移植（要 mujoco + torch）
/home/wck/miniconda3/bin/python verify.py --trace trace.npz

# ③ 批量评测（自采样）
/home/wck/miniconda3/bin/python sim2sim.py --episodes 200 --headless

# ④ A/B 同场景对比（主线判据）
/home/wck/miniconda3/bin/python sim2sim.py --scenarios trace_scenarios.npz --headless

# ⑤ 开窗口看策略跑（R/空格 换目标，ESC 退出）
/home/wck/miniconda3/bin/python sim2sim.py --render
```

**跑端到端之前一定先跑 ②。** 四项检查任何一项不过，说明映射链路有问题，
继续跑端到端只会在错误的映射上得出"策略不行"的结论。

### 查看器里能看到什么

`--render` 时那颗半透明的红球（带红/绿/蓝三色坐标轴，轴朝向就是目标姿态）**就是本 episode
真正的目标** —— 它是 XML 里 `ik_control.py` 用的 `ik_target` mocap 刚体，`sim2sim.py`
每个 episode 开始时把它挪到目标位姿。mocap 是运动学的、geom 也关掉了碰撞，所以**纯可视化，
不影响动力学**（headless 与 render 的结果逐位相同）。

终端里每控制步刷新一行，显示当前的位置/姿态误差和本 episode 的最好成绩。

相机在第一次 `sync()` 之后才设（查看器首次 sync 会做自动取景把参数盖掉，
`../arm_mujoco/view_arm.py:137-139` 记过这个坑）。`distance=2.2` 是拿离屏渲染逐个目标点
试出来的：工作空间（z ∈ [0.05, 0.70]、半径 ∈ [0, 0.65]）15 个采样点全部落在画面内。
不设相机的话目标稍微偏一点就跑到画面外，看着像"目标没动"。

## verify.py 检查什么

| 项 | 内容 | 判据 | 实测 |
| --- | --- | --- | --- |
| 1 | FK：MuJoCo 正解 vs Isaac 刚体状态 | 位置 <1e-4 m / 姿态 <1e-3 rad | 1.2e-6 / 8.8e-4 |
| 2 | **观测构造**（重放 30 维观测） | <1e-5 | 4.8e-7 |
| 3 | 策略网络输出 | <1e-5 | 3.8e-6 |
| 4 | 力矩链路（双重裁剪 + 前馈） | <1e-5 | 2.9e-6 |

**第 2 项最有价值** —— 它一次能抓出四元数顺序、`rot6d` 错位、符号、缩放、切片错位的
所有 bug。做法是让 Isaac 逐控制步落盘 `(q, qd, ee_pos, ee_quat, target, last_action, obs)`，
再用我们自己的 `build_obs` 重放比对。

## 移植时最容易翻车的几点

### 1. `quat_to_rot6d` 有一个必须**原样保留**的错位

`../arm_rl/arm_reach_env.py:23-32` 的 docstring 写着「输入 (x,y,z,w)」，但函数体把
`q[:,0]` 当成 **w** 用。而喂进去的 `q_rel` 来自 `isaacgym.torch_utils.quat_mul`，
**确实是 (x,y,z,w)**。

所以网络实际看到的 6D 向量是「把这个 (x,y,z,w) 四元数的四个分量重新解释成 (w,x,y,z)
之后算出的 R 前两列」。**策略就是在这个表示上训练出来的**，改成标准公式会直接毁掉策略。

同一个 `q_rel` 的对照：

```
真 (x,y,z,w) 下的 R 前两列 = [ 0.42, -0.0149, -0.907,  0.615,  0.740,  0.272]
本仓库实际输出             = [-0.32,  0.272,   0.907, -0.672, -0.740, -0.015]
```

`kinematics.quat_to_rot6d` 是从原文件逐字符搬过来的，注释里也标了「别改」。
注意 `quat_angle` 用的是 `q[3]`（在 xyzw 里正好是 w），那一个是**对的**，别跟着一起改。

### 2. 末端必须用 `xpos`（刚体原点），不能用 `xipos`（质心）

零位两者差 **94.6 mm**，而达标阈值才 20 mm。用错观测全废。
`verify.py` 的第 1 项会把 `xipos` 的误差也打出来做对照。

### 3. `actuator_ctrlrange` 必须一起清掉

XML 里的 `<position>` 伺服，其 `ctrlrange` 被设成了**关节行程**。把它改造成力矩执行器时，
只改 `gainprm`/`biasprm` 而忘了清 `ctrlrange`，`ctrl` 会被当成目标角度钳制 ——
实测 `ctrl=3.0` 会被削成 `2.79`（Joint3 的上限）。四条都要：

```python
model.actuator_gainprm[:, 0] = 1.0
model.actuator_biasprm[:, :3] = 0.0
model.actuator_ctrlrange[:] = 0.0
model.actuator_ctrllimited[:] = 0     # ★ 最容易漏
```

### 4. 力矩是**双重裁剪**

```
tau = clip( clip(kp*(target_q - q) - kd*qd, ±lim) + gravity_ff, ±lim )
           └──── 第一道（基类）────┘         └──── 第二道（env 覆写）────┘
```

前馈本身能把关节推到力矩上限，少了第二道就会超限。

### 5. 重力前馈滞后一个控制步

`_update_gravity_ff` 在 `post_physics_step` 里调用，也就是 decimation 的 4 个物理子步
**跑完之后**才更新，下一步才用它。实现上就是把 `gravity_torque()` 的调用放在
`_decimate()` 之后。

另外它**只含重力、不含科氏项**。所以 MuJoCo 侧要用 `qfrc_bias` 在 **qvel=0** 时的值，
不能直接读带速度的实时值。

### 6. 时基

物理 `1/120`、decimation 4、控制 30 Hz、episode **150 个策略步**。
终止判据是 `episode_length_buf > 150`（严格大于），而 `reset()` 内部还会走一记
`step(zeros)`，所以 buf 从 1 数到 151。别写成 `>=`。

### 7. 四元数只在 backend 边界转一次

MuJoCo 是 `(w,x,y,z)`，Isaac 是 `(x,y,z,w)`。`backend.wxyz_to_xyzw` 是唯一的转换点，
之后全链路统一 xyzw。

### 8. obs 在 decimation **之后**算，且 `obs[24:30]` 是刚施加的、**裁剪之后**的动作

裁剪发生在乘 `action_scale` **之前**。`clip_actions = 100.0`，不是 ±1。

## 一个和模型无关的坑：`command_pose.py` 的「零位末端位姿」（已修）

`../arm_rl/command_pose.py` 的 `_print_reference_pose` 原本直接读
`env.ee_pos_rel` / `env.ee_quat`。但这两个量只在 `_post_physics_step_callback` 里刷新，
而那只在 `step()` 的路径上；调用点是在 `env.reset()` 之后、一步都没走过，
所以读到的是**上一次刷新的缓存值**，和当前关节角对不上。加上 `reset_idx` 只写了关节角，
PhysX 的刚体变换要 `simulate` 之后才更新。

实际差多少：位置 19 mm、姿态 **127°**。而 ±0.1 扰动分布**内部**两两姿态差只有 9.8° ——
两个分布完全不重叠，所以那不是"姿态敏感"，是真的读错了值。

**已修**：改成显式把关节角设到 `default_dof_pos`、走一步物理、读回来、再还原
（和 `_resample_commands` 里先走一步再读是同一个道理）。修完打印值：

```
--pos -0.010 -0.001 0.366 --quat -0.00044 -0.00081 0.78390 -0.62089
```

对照 MuJoCo 在同一关节角下的正解 `pos=(-0.0103,-0.0014,0.3660)`、
`quat(xyzw)=(0, 0.78332, -0.62161, 0)` —— **三位小数吻合**，残余差异是那一步积分的漂移。

> 这个 bug 当初把 sim2sim 的验证带偏过一轮：一开始想拿它当交叉验证基准，
> 得出的结论是"两侧 FK 差 127°"。后来改用「Isaac 侧显式 dump 一组已知 qpos 的刚体状态」
> 才把两侧坐实到微米级。修好之后的这个打印值，反过来成了 FK 一致的第二个独立证据
> （走的是和 trace dump 完全不同的代码路径）。

## 已知的模型差异（都做成了开关）

两个模型的关节顺序、轴向（含 Joint4 的斜轴）、限位、力矩上限、质量惯量、armature
**全部一致**（见 `verify.py` 第 1 项，位置误差 1.2 微米）。剩下的差异：

| # | 差异 | 默认 | 开关 |
| --- | --- | --- | --- |
| 1 | 关节速度限 3.14 rad/s（Isaac 有，MuJoCo 无） | 忽略 | `--vel-limit 3.14` |
| 2 | 手臂-地面碰撞（Isaac 开、自碰撞关；MuJoCo 全关） | 关 | `--ground-collision` |
| 3 | 积分器（PhysX vs implicitfast） | 用 XML 的 | `--integrator euler\|implicit\|implicitfast` |
| 4 | 控制器（位置伺服 kp100/kv50 vs 训练 PD kp80,20/kd4,1 + 前馈） | **复刻训练侧** | — |
| 5 | base_link 质量在 MuJoCo 侧被丢弃 | 忽略 | — |

默认值的选取原则是「与当前已被 `ik_control.py` 验证过的 MuJoCo 模型一致」，
这样首轮就能把「策略迁移」和「模型差异」两件事解耦。

第 4 条是**必须复刻**的：MuJoCo 的 XML 装的是位置伺服 kp=100/kv=50，和训练用的
PD + 重力前馈完全不等价。所以 `backend.py` 在运行时把它改造成纯力矩执行器，
**不动 XML**（`ik_control.py` 依赖那套位置伺服）。`arm_mujoco/URDF_arm.xml` 全程没改。

第 1、2 条只在快速瞬态和低 z 目标上起作用。自由分布下按目标高度分层的误差中位数
（z<0.2 → 9.7 mm，z>0.5 → 7.2 mm）有一点这个趋势，但幅度远小于模型差异该有的量级。

**这两条的实测影响（200 个同场景）：**

| 开关 | 位置中位 | 成功率 | 结论 |
| --- | --- | --- | --- |
| 默认 | 8.0 mm | 92.0% | — |
| `--ground-collision` | 8.0 mm | 92.0% | **逐位相同** |
| `--vel-limit 3.14` | 7.9 mm | 91.5% | 200 个里只翻了 1 个 |
| 两个都开 | 7.9 mm | 91.5% | 同上 |

`--ground-collision` 一位都没变，查下来是**手臂全程没碰过地**：200 个 episode 里
最大同时接触数 **0**，手臂最低点 **z = 0.0400 m**（地面在 z=0，而目标采样的
`min_z = 0.05`，任务本身就够不到地面）。所以这条差异对本任务**可证明地无关**，
默认关掉是对的。

`--vel-limit` 只在快速瞬态起作用，而达标判定取的是**终点**误差（终点速度接近 0），
所以也几乎无影响。默认忽略。

## 环境

| 用途 | 环境 | 依赖 |
| --- | --- | --- |
| 导 Isaac trace | conda `legged_gym` | isaacgym 1.0rc4 + torch 2.0.1（`ninja` 要在 PATH 上） |
| 验证 + 移植运行 | conda `base` | mujoco 3.9.0 + torch 2.12.1 |

base 环境里同时有 mujoco 和 torch，所以**策略推理不需要 ONNX、不需要装任何新包** ——
直接从 checkpoint 重建 MLP 跑纯 CPU 前向（`policy.py`）。上真机时这一条同样成立：
只需要 torch。

## 上真机要填的部分

照着 `backend.RobotBackend` 协议写一个实物后端即可，其余代码原样带走：

```python
class MyRobotBackend:
    n_dof = 6
    def reset(self, q, qd): ...            # 让机械臂去初始位姿，或只记录
    def get_joint_state(self): ...          # 读编码器
    def get_ee_pose(self): ...              # 用编码器做正解，返回 (基座系位置, xyzw)
    def set_joint_torques(self, tau): ...   # 下发力矩指令（这里再套安全限幅）
    def step(self): ...                     # 阻塞到下一个 1/120 节拍
    def joint_limits(self): ...             # 关节行程
    def ee_pose_from_q(self, q): ...        # 正解，不改变任何状态
    def gravity_torque(self): ...           # 可选；返回 None 表示不做前馈
```

`gravity_torque` 在仿真里走 `qfrc_bias`，真机上可以换 pinocchio 的 RNEA(qvel=0)
或者自己的重力模型；返回 `None` 时 runner 用零前馈。
