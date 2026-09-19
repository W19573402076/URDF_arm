# URDF_arm 在 Isaac Gym 中打开

把 `../URDF_arm` 里的六轴机械臂（base_link + Link1~Link6，Joint1~Joint6）转成 Isaac Gym
可加载的资源，并提供一个查看窗口。

```
arm_isaacgym/
├── build_assets.py     # 从 ../URDF_arm 生成 assets/（网格 + 改写后的 URDF）
├── view_arm.py         # 在 Isaac Gym 中打开模型
└── assets/             # 生成物，可随时删掉重跑
    ├── urdf/URDF_arm.urdf
    └── meshes/*.STL
```

## 环境

已装在 `legged_gym` conda 环境里，用它的解释器跑：

```bash
/home/wck/miniconda3/envs/legged_gym/bin/python view_arm.py
```

（Isaac Gym 在 `/home/wck/Gym/isaacgym`，通过 egg-link 装进该环境，Python 3.8。）

## 用法

```bash
python build_assets.py          # 先生成资源（改过源 URDF 后重跑即可）
python view_arm.py              # 开窗口，六个关节按正弦摆动
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--static_pose` | 保持零位，只看模型或调姿态 |
| `--num_envs 4 --spacing 1.2` | 一次摆多个，方便对比 |
| `--amplitude_scale 0.35` | 摆动幅度占关节行程的比例 |
| `--frequency 0.25` | 摆动频率（Hz） |
| `--free_base` | 基座不焊死在世界坐标系上（会受重力掉下来） |
| `--headless --steps 240` | 不开窗口只跑物理，用来验证资源能否加载 |

窗口里鼠标左键拖动 = 转视角，滚轮 = 缩放，中键 = 平移，ESC = 退出。左侧面板会列出
`base_link`、`Link1~Link6` 和六个关节，可以逐个体看编号。

`--headless` 下会打印连杆质量、关节限位和每 60 步的关节角，用来确认模型确实在动。

## build_assets.py 做了什么

1. 复制 `../URDF_arm/meshes/*.STL` 到 `assets/meshes/`。
2. 生成 `assets/urdf/URDF_arm.urdf`，把 `package://URDF_arm/meshes/x.STL` 改写成
   `../meshes/x.STL`。**Isaac Gym 的 URDF 解析器相对 URDF 文件所在目录找网格，不认
   `package://` 前缀**，所以这一步是必须的。

`--collision` 控制碰撞体怎么生成：

| 取值 | 说明 |
| --- | --- |
| `mesh`（默认） | 碰撞体用原始 STL。七个网格共约 73 万面片，加载约 5 秒，最忠实 |
| `box` | 碰撞体换成各网格的轴对齐包围盒，加载更快，但形状是近似的 |
| `none` | 碰撞体全部删掉。**注意：Isaac Gym 加载无碰撞体的 URDF 会段错误**，别用 |

## 踩过的坑

- **`gymapi.PlaneParams()` 的默认法线是 `(0, 1, 0)`**，即 Y 轴向上。本仿真用 Z 轴向上
  （`UP_AXIS_Z`），直接 `gym.add_ground(sim, gymapi.PlaneParams())` 等于插了一块竖直
  的地板切进机械臂，关节会被瞬间弹飞（第一步速度就有上百 rad/s）。必须显式设
  `normal = (0, 0, 1)`。这个坑跟模型无关，Isaac Gym 自带的 franka 一样中招。
- 用官方的 `joint_monkey.py` 做对照时，它是每帧直接写 DOF 状态（`set_actor_dof_states`），
  所以不受地板朝向影响，容易误判成"模型有问题"。
- Joint3 的零位不在它的行程 `[0.35, 2.79]` 内，所以零位指令会被贴到下限 0.35。这是
  URDF 本身的定义，不是转换错误。Joint2 零位有约 0.075 rad 的重力下沉，加大
  `view_arm.py` 里的 `STIFFNESS` 可以压小。
