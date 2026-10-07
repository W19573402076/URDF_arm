# URDF_arm —— 六轴机械臂在 MuJoCo / Isaac Gym / 强化学习上的实现

一台六轴机械臂（`base_link` + `Link1~Link6`，`Joint1~Joint6`）的仿真与学习项目。同一份
URDF 源文件，做了四个互相独立的实验：传统逆运动学控制、Isaac Gym 资产加载、
用 PPO 训练末端 6D 位姿到达、以及把训好的策略移植到 MuJoCo（sim2sim）。

各部分都有独立的 README，**踩过的坑都记在里面** —— 那些是这个仓库主要的价值所在，
每个坑都带实测数据，不是「试过不行」的一句话结论。

## 目录结构

```
URDF/
├── URDF_arm/        源 URDF 包（ROS 格式，直接用 SolidWorks 插件导出）
│   ├── urdf/URDF_arm.urdf
│   ├── meshes/*.STL
│   ├── config/joint_names_URDF_arm.yaml
│   └── launch/{display,gazebo}.launch
├── arm_mujoco/      MuJoCo 后端：URDF→MJCF 转换、查看器、拖拽目标 + 阻尼最小二乘 IK
├── arm_isaacgym/    Isaac Gym 后端：资产构建 + 查看器
├── arm_rl/          PPO 训练末端 6D 位姿到达（跑在 legged_gym 上）
└── arm_mujoco_rl/   把 arm_rl 训好的策略移植到 MuJoCo 跑（sim2sim，上真机的前置）
```

## 四个部分

### `arm_mujoco/` —— 拖拽目标 + IK

场景里有一个半透明的红色 mocap 目标球，**Ctrl + 鼠标拖动**就能移动它，脚本每帧用
阻尼最小二乘 IK 解出关节角，通过位置伺服把末端开过去。末端上的绿球和目标红球的
距离就是当前误差。

```bash
python build_model.py     # 从 ../URDF_arm 生成 URDF_arm.xml 和 assets/
python view_arm.py        # 开窗口，六个关节按正弦摆动
python ik_control.py      # 拖拽目标，IK 跟踪
```

**实测**：停下来后位置误差收敛到 **0.000000 mm**（`|qvel|` 恰好 0）；跟踪运动目标时
误差约 **1~3 mm**。

### `arm_isaacgym/` —— Isaac Gym 资产加载

把源 URDF 转成 Isaac Gym 可加载的资源并开查看器。

```bash
python build_assets.py    # 生成 assets/（网格 + 改写路径后的 URDF）
python view_arm.py        # 开窗口，六个关节按正弦摆动
```

### `arm_rl/` —— PPO 训练 6D 位姿到达

用 PPO 训练策略，让末端到达随机指定的 6D 位姿（位置 + 姿态）。跑在 `legged_gym` 上，
env 类继承 `LeggedRobot`，**legged_gym 一行没改**。

```bash
python train.py --headless                     # 训练，默认 4096 环境 2000 迭代
python play.py --episodes 200                  # 评估最新 checkpoint
python command_pose.py                         # 交互式：把末端开到指定位姿，窗口里实时看
```

当前最好模型（`logs/Sep19_19-10-57_/model_20000.pt`，20000 迭代）的评测结果，
400 个随机目标、确定性策略：

| 指标 | 位置误差 | 姿态误差 |
| --- | --- | --- |
| median | **7.6 mm** | **0.99°** |
| mean | 12.1 mm | 1.65° |
| p95 | 30.2 mm | 3.60° |

严格达标率（位置 < 20 mm **且** 姿态 < 0.1 rad）**90.5%**。

这个数字是逐步调出来的，过程记在 [`arm_rl/README.md`](arm_rl/README.md) 里。最有价值的
一条结论是：**之前的瓶颈一直是「算力 + 容量」，不是算法或奖励设计**——七次在奖励和
控制器上的折腾把成功率从 7.7% 抬到 9.5%，而把网络从 `[256,128,64]` 放大到
`[512,256,128]`、训练轮数从 2000 提到 20000，到了 90.5%。

### `arm_mujoco_rl/` —— 把训练好的策略移植到 MuJoCo（sim2sim）

把上面那个策略从 Isaac Gym 拿到 MuJoCo 里跑。代码刻意分成两半：一半是
「读状态 → 算观测 → 跑网络 → 算力矩」的推理+控制模块（**不依赖任何仿真器**，
上真机时原样带走），另一半是 MuJoCo 验证台。

```bash
python tools/dump_isaac_trace.py --out trace.npz --scenarios 200   # legged_gym 环境
python verify.py --trace trace.npz                                  # conda base
python sim2sim.py --episodes 200 --headless
python sim2sim.py --scenarios trace_scenarios.npz --headless        # A/B 同场景对比
python sim2sim.py --render                                          # 开窗口看策略跑
```

**200 个完全相同的场景**，两边各跑一遍：

| | 位置中位 | 位置 p95 | 姿态中位 | 严格成功率 |
| --- | --- | --- | --- | --- |
| Isaac Gym | 8.1 mm | 28.7 mm | 0.96° | 90.0% |
| MuJoCo | **8.0 mm** | **22.2 mm** | **1.00°** | **92.0%** |

位置中位差 **0.1 mm**。移植的难点全在「逐位复刻训练侧的接口契约」上，细节和踩过的坑
记在 [`arm_mujoco_rl/README.md`](arm_mujoco_rl/README.md) —— 其中有一条是训练代码里
**必须原样保留的错位**（`quat_to_rot6d` 的 docstring 和实现不一致，而策略是在那个
表示上训出来的，改成"正确"公式会直接毁掉策略）。

## 环境

各部分的依赖不同：

| 部分 | 环境 | 依赖 |
| --- | --- | --- |
| `arm_mujoco` | conda `base` | `mujoco`（实测 3.9.0）、`numpy`、`glfw`（键盘微调可选） |
| `arm_isaacgym` | conda `legged_gym` | Isaac Gym 1.0rc4（Python 3.8） |
| `arm_rl` | conda `legged_gym` | Isaac Gym 1.0rc4 + `rsl_rl` 1.0.2 + `torch` 2.0.1+cu117 |
| `arm_mujoco_rl` 运行 | conda `base` | `mujoco` 3.9.0 + `torch` 2.12.1（不需要 ONNX） |
| `arm_mujoco_rl` 导 trace | conda `legged_gym` | 同上 Isaac Gym 环境 |

Isaac Gym 和 `legged_gym` 都是**外部依赖，没有放进本仓库**。要跑 `arm_isaacgym/` 和
`arm_rl/` 需要自己准备：

- [Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym)（NVIDIA 官网下载，需要账号）
- [legged_gym](https://github.com/leggedrobotics/legged_gym)

> ⚠️ 三个子 README 里的命令写的是本机路径（`/home/wck/miniconda3/envs/legged_gym/bin/python`
> 这类），照搬前请换成你自己的环境路径。`arm_rl` 必须先 `conda activate`——`gymtorch`
> 要 JIT 编译，需要 `ninja` 在 `PATH` 上。

## 关于网格文件

`URDF_arm/meshes/` 下的 STL 是这台机械臂的 CAD 导出网格，来自源 URDF 包。两个仿真后端
各自持有一份副本（`arm_isaacgym/assets/meshes/`、`arm_mujoco/assets/meshes/`），这是为了
让每个子项目能独立运行。它们在 git 里只存一份（内容寻址去重）。

**如果你要基于本仓库做二次分发，请自行确认这些网格的授权。** 仓库里不含任何厂商的
专有代码，但 CAD 网格的来源和许可需要你自己判断。

## 训练产物

`arm_rl/logs/` 下只保留了 v13 那一个 checkpoint（`model_20000.pt`，4.2 MB），其余
五次训练的 204 个中间 checkpoint 没入库。clone 后可以直接：

```bash
python play.py --episodes 200     # 用保留的这个模型评估
```

## 各部分之间的关系

- `arm_mujoco/` 和 `arm_isaacgym/` 是同一份 URDF 的两个仿真后端。两边都不认
  `package://` 网格路径，都栽在 CAD 网格自碰撞上——但**具体表现完全不同**：
  MuJoCo 侧零位有 6 处穿透接触（最深 2.5 cm），Isaac Gym 侧则是加载无碰撞体的 URDF
  会段错误。细节见各自 README。
- `arm_rl/` 复用 `arm_isaacgym/assets/`（同一份资产，不重复拷贝 35 MB 网格）。
- `arm_mujoco_rl/` 依赖 `arm_rl/` 训出来的 checkpoint 和 `arm_mujoco/URDF_arm.xml`，
  但**不 import 它们** —— 观测构造、控制律、四元数工具都是逐位重写的一份，
  为的是能脱离 Isaac Gym 独立运行（也就是上真机的前提）。

`arm_mujoco/ik_control.py` 和 `arm_mujoco/URDF_arm.xml` 被 `arm_mujoco_rl/` 当作只读模型
复用，前者依赖的位置伺服执行器在后者里是**运行时**改造成力矩执行器的，两边互不干扰。
