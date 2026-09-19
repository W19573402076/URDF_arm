#!/usr/bin/env python3
"""把 ../URDF_arm 的机械臂转成 MuJoCo 模型，输出 URDF_arm.xml + assets/。

转换路线是用 MuJoCo 自带的 URDF 解析器（mujoco.MjModel.from_xml_path）读 URDF，
再用 mujoco.mj_saveLastXML 把编译结果导成 MJCF，然后在导出的骨架上补 MuJoCo 特有的
东西（地面、光照、天空盒、位置伺服、keyframe）。这样惯性张量、关节坐标系、四元数
分解这些都交给 MuJoCo 自己算，不用手写，不会出错。

用法：
    python build_model.py                  # 碰撞体关掉（默认），纯看模型/摆姿态
    python build_model.py --collision mesh # 用原始网格做碰撞体

关于碰撞：七个连杆的 CAD 网格在关节处是互相穿插的，零位时 MuJoCo 就报出 6 处穿透
接触（最深 2.5 cm），静置两秒后关节会被接触力顶到限位。所以默认关掉碰撞，只留可视化
几何体。需要碰的时候再加 --collision mesh。
"""

import argparse
import shutil
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "URDF_arm"
ASSETS = HERE / "assets"
MESH_DIR = ASSETS / "meshes"

# MuJoCo 的 STL 解码器硬上限：面片数必须在 [1, 200000] 之间，超了直接报错。
# Link3.STL 有 228234 个面片，所以超限的网格要转成 OBJ 再加载。
STL_MAX_FACES = 200000

JOINTS = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"]

# 位置伺服增益。URDF 的 effort 只有 20/20/20/3/3/3 N·m，末端三个轴偏软，
# 这里只放宽 kp，力矩仍按 URDF 的 effort 限幅。
#
# kv 必须给够大，否则整个手臂会明显抖动。原因是阻尼比按
# ζ = kv / (2·√(kp·I)) 算，而 I 是**机械臂自身的惯量**（Joint1 要转整条臂，约 1~2 kg·m²），
# 不是下面那个 0.01 的 armature —— armature 相对臂的惯量微不足道，起不到作用。
# kv=5 时 ζ≈0.2，严重欠阻尼。实测扫过（kp=100，目标固定，看稳态）：
#     kv=5  -> 末端误差 12.30 mm，抖动 std 5.36 mm，|qvel| 峰 0.90 rad/s
#     kv=10 ->  5.71 mm / 2.35 mm
#     kv=20 ->  2.69 mm / 1.03 mm
#     kv=50 ->  0.89 mm / 0.26 mm   <-- 用这个
KP = 100.0
KV = 50.0

# 关节电枢（电机转子折合到关节上的惯量），kg·m²。
# URDF 里没有这个概念，而 Link6 绕关节的惯量只有 1.2e-4 kg·m²，kp=100 配上
# dt=0.002 正好卡在显式积分的稳定边界外，仿真会持续抖动（实测关节最大速度
# 29 rad/s）。加上 0.01 的电枢相当于把伺服电机自身的惯量算进去，抖动消失
# （峰值速度 0.0098 rad/s），这也是 MuJoCo 机器人模型的标准做法。
ARMATURE = 0.01

# 与 ball/sphere.xml 保持一致的配色风格
SKY = dict(type="skybox", builtin="gradient",
           rgb1="0.2 0.4 0.8", rgb2="0.55 0.75 0.98", width="256", height="256")
GROUND_TEX = dict(type="2d", builtin="checker", width="256", height="256",
                  rgb1="0.28 0.30 0.28", rgb2="0.50 0.52 0.50")


def stl_face_count(path):
    """二进制 STL 的面片数。返回 None 表示要按 ASCII 处理。"""
    data = path.read_bytes()
    if len(data) < 84:
        return None
    n = struct.unpack("<I", data[80:84])[0]
    return n if len(data) == 84 + n * 50 else None


def stl_to_obj(src, dst):
    """二进制 STL 转 OBJ。纯 numpy，不依赖 trimesh 之类的库。"""
    data = src.read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    faces = np.frombuffer(data[84:84 + n * 50], dtype=np.uint8).reshape(n, 50)
    verts = faces[:, 12:48].copy().view("<f4").reshape(n, 3, 3).astype(np.float64)

    uniq, inv = np.unique(verts.reshape(-1, 3), axis=0, return_inverse=True)
    tris = inv.reshape(n, 3) + 1
    with open(dst, "w") as fh:
        fh.write("# build_model.py 从 %s 转换，%d 顶点 %d 面\n" % (src.name, len(uniq), n))
        np.savetxt(fh, uniq, fmt="v %.6f %.6f %.6f")
        np.savetxt(fh, tris, fmt="f %d %d %d")
    return len(uniq), n


def copy_meshes():
    """复制网格，超限的 STL 转成 OBJ。返回 {mujoco 里用的文件名: 说明}。"""
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    mapping, notes = {}, []
    for mesh in sorted((SRC / "meshes").iterdir()):
        if not mesh.is_file():
            continue
        faces = stl_face_count(mesh) if mesh.suffix.lower() == ".stl" else None
        if faces is not None and faces > STL_MAX_FACES:
            out = MESH_DIR / (mesh.stem + ".obj")
            nv, nf = stl_to_obj(mesh, out)
            notes.append("%s：%d 面 > MuJoCo 上限 %d，已转成 OBJ（%d 顶点）"
                         % (mesh.name, faces, STL_MAX_FACES, nv))
        else:
            out = MESH_DIR / mesh.name
            shutil.copy2(mesh, out)
        mapping[mesh.name] = out.name
    return mapping, notes


def write_urdf(mesh_mapping):
    """生成一份网格路径相对的 URDF，给 MuJoCo 的 URDF 解析器读。

    MuJoCo 和 Isaac Gym 一样不认 package:// 前缀，网格路径是相对 URDF 文件所在目录
    解析的。assets/URDF_arm.urdf 与 assets/meshes/ 同级，所以写成 meshes/x.STL。
    """
    tree = ET.parse(SRC / "urdf" / "URDF_arm.urdf")
    root = tree.getroot()
    for mesh in root.iter("mesh"):
        name = mesh.get("filename").split("/")[-1]
        mesh.set("filename", "meshes/" + mesh_mapping.get(name, name))

    # 加缩进（Python 3.8 的 ET 没有 indent）
    def indent(elem, level=0):
        pad = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = pad + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = pad
            for child in elem:
                indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = pad
        elif level and (not elem.tail or not elem.tail.strip()):
            elem.tail = pad
    indent(root)

    out = ASSETS / "URDF_arm.urdf"
    out.write_text('<?xml version="1.0" encoding="utf-8"?>\n'
                   "<!-- 由 build_model.py 生成，只是给 MuJoCo 的 URDF 解析器做输入的中间产物。\n"
                   "     真正要用的是上一层的 URDF_arm.xml。 -->\n"
                   + ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
    return out


def dump_mjcf(urdf_path):
    """用 MuJoCo 编译 URDF，导出成 MJCF。"""
    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    tmp = ASSETS / "_dump.xml"
    mujoco.mj_saveLastXML(str(tmp), model)
    return tmp, model


def look_at_xyaxes(pos, target):
    """算出 MuJoCo 相机的 xyaxes。相机沿自身 -Z 方向看，+X 向右，+Y 向上。"""
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    z = pos - target
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return " ".join("%.4f" % v for v in np.concatenate([x, y]))


def build_extra_assets():
    """天空盒、地面棋盘格纹理和材质，配色沿用 ball/sphere.xml。"""
    asset = ET.Element("asset")
    ET.SubElement(asset, "texture", name="sky", **SKY)
    ET.SubElement(asset, "texture", name="ground_tex", **GROUND_TEX)
    # 机械臂各连杆的颜色沿用 URDF 里的定义（MuJoCo 已经带进了 geom 的 rgba），
    # 这里只加地面材质。注意 geom 上的 rgba 优先级高于 material，两边都写会白写。
    ET.SubElement(asset, "material", name="ground_mat", texture="ground_tex",
                  texrepeat="40 40", reflectance="0.05")
    return asset


def decorate(tree, collision, model):
    """在 MuJoCo 导出的骨架上补地面、光照、伺服和 keyframe。"""
    root = tree.getroot()
    root.set("model", "URDF_arm")

    # <compiler>：网格目录统一放 assets/meshes/
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("angle", "radian")
    compiler.set("meshdir", "assets/meshes")

    # <asset>：去掉 MuJoCo 导出时加的 content_type，路径只留文件名交给 meshdir
    asset = root.find("asset")
    for mesh in asset.findall("mesh"):
        mesh.attrib.pop("content_type", None)
        mesh.set("file", mesh.get("file").split("/")[-1])
    for tag in ("texture", "material"):
        for node in build_extra_assets().findall(tag):
            asset.append(node)

    # <option>：显式写出来，文件自解释。
    #
    # integrator="implicitfast" 不是可选项，别去掉。位置伺服的 kv 项是显式积分的阻尼，
    # 半隐式欧拉要求 dt < 2·M_ii/kv（M_ii 是**该关节处**的有效惯量，取自 mj_fullM 对角）。
    # 腕部三轴的 M_ii 只有 0.010~0.014，是 armature 主导的，kv=50 时需要
    # dt < 0.40~0.54 ms，而这里是 2 ms —— 超界 4~5 倍，末三轴会以 250 Hz（正好 Nyquist）
    # 持续颤振，J6 摆幅 0.154 rad。**这一点极易漏判**：它表现在腕部**姿态**上而不是末端
    # **位置**上（末端 site 就挂在 Link6 的原点、正在 J6 转轴上，J6 转多少它都不动），
    # 所以所有位置类指标都只有 1~3 mm，看着像没问题，实际静止时一直在抖。
    #
    # 上面 KP/KV 那段注释只讲了阻尼比 ζ = kv/(2√(kp·I))，那条对 J1~J3 是对的
    # （M_ii 0.14~0.23，dt<6 ms，安全）；腕部受的是另一条离散稳定边界约束，看的正好是
    # 被判断为「微不足道」的 armature。implicitfast 让 MuJoCo 对阻尼项隐式求解，
    # 腕部姿态抖动 4.18° → 0.53°、J6 摆幅降到 4.8e-4 rad，且不再受 dt 限制。
    option = ET.Element("option", timestep="0.002", gravity="0 0 -9.81",
                        integrator="implicitfast")
    root.insert(list(root).index(compiler) + 1, option)

    # <visual>：跟 ball/sphere.xml 一样的头灯
    visual = ET.Element("visual")
    ET.SubElement(visual, "headlight", ambient="0.3 0.3 0.4", diffuse="0.5 0.5 0.55",
                  specular="0.1 0.1 0.1")
    root.insert(list(root).index(option) + 1, visual)

    # 地面、灯光、相机
    worldbody = root.find("worldbody")
    ET.SubElement(worldbody, "geom", name="ground", type="plane", size="5 5 0.1",
                  material="ground_mat", friction="1.5 1.0 1.5")
    ET.SubElement(worldbody, "light", name="sun", pos="2 1.5 3", dir="-0.5 -0.4 -1",
                  diffuse="1.0 0.95 0.8", specular="0.4 0.4 0.4", castshadow="true")
    ET.SubElement(worldbody, "light", name="sky", pos="0 0 3", dir="0 0 -1",
                  diffuse="0.4 0.45 0.65", castshadow="false")
    cam_pos, cam_target = (0.5, -0.7, 0.5), (0.0, 0.0, 0.2)
    ET.SubElement(worldbody, "camera", name="front",
                  pos=" ".join("%.4f" % v for v in cam_pos),
                  xyaxes=look_at_xyaxes(cam_pos, cam_target))

    # 可拖拽的 IK 目标标记（ik_control.py 用）。mocap 刚体不受物理影响，
    # 在 MuJoCo 查看器里 Ctrl+拖动就能移动它。初始位置由脚本设成末端当前位姿。
    target = ET.SubElement(worldbody, "body", name="ik_target", mocap="true",
                           pos="0.0 0.35 0.35")
    # 目标标记 = 半透明球 + 三轴十字。机械臂够到目标后必然和它重合，
    # 而 MuJoCo 的渲染器**没有**「逐几何体不做深度测试」的开关（geom_priority 是接触
    # 优先级，不是渲染），真正的「永远画在最上层」得改渲染管线才行。
    # 所以改成让标记在**几何上**外扩：十字的三个臂长 ±7 cm，比任何连杆的截面都长，
    # 无论从哪个角度看都会有部分戳在臂体外，不会被完全遮住。
    # 十字跟着 mocap 刚体走，所以它同时也是目标姿态的可视化。
    ET.SubElement(target, "geom", name="ik_target_geom", type="sphere", size="0.045",
                  rgba="0.9 0.15 0.15 0.28", contype="0", conaffinity="0")
    for axis, rgba in ((0, "1 0.25 0.25 1"), (1, "0.25 1 0.35 1"), (2, "0.35 0.5 1 1")):
        half = [0.006, 0.006, 0.006]
        half[axis] = 0.070                      # 臂长 ±7 cm
        ET.SubElement(target, "geom", name="ik_target_axis%d" % axis, type="box",
                      size=" ".join("%.4f" % v for v in half),
                      rgba=rgba, contype="0", conaffinity="0")
    ET.SubElement(target, "site", name="ik_target_site", type="sphere", size="0.012",
                  rgba="1.0 0.35 0.35 1")

    # 末端标记点，放在 Link6 上，用来看 IK 跟目标差多少
    for body in worldbody.iter("body"):
        if body.get("name") == "Link6":
            ET.SubElement(body, "site", name="ee_site", type="sphere", size="0.012",
                          rgba="0.2 1.0 0.3 1")

    # 机械臂的几何体：默认关碰撞，只做可视化
    for geom in worldbody.iter("geom"):
        if geom.get("name") == "ground":
            continue
        if not collision:
            geom.set("contype", "0")
            geom.set("conaffinity", "0")

    # 给每个关节加上电枢，否则末端小惯量关节的伺服会抖
    for joint in worldbody.iter("joint"):
        joint.set("armature", "%g" % ARMATURE)

    # 位置伺服：MuJoCo 的 URDF 导入不会建 actuator，URDF 里没有执行器这个概念
    actuator = ET.Element("actuator")
    for joint in JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        lo, hi = model.jnt_range[jid]
        effort = model.jnt_actfrcrange[jid]
        ET.SubElement(actuator, "position", name=joint + "_servo", joint=joint,
                      kp="%.0f" % KP, kv="%.0f" % KV,
                      ctrlrange="%.4f %.4f" % (lo, hi),
                      forcerange="%.4f %.4f" % (effort[0], effort[1]))
    root.append(actuator)

    # keyframe：零位，但零位可能落在行程外（Joint3 就是），要贴到最近的限位
    home = []
    for joint in JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        lo, hi = model.jnt_range[jid]
        home.append(min(max(0.0, lo), hi))
    keyframe = ET.Element("keyframe")
    ET.SubElement(keyframe, "key", name="home",
                  qpos=" ".join("%.6f" % v for v in home),
                  ctrl=" ".join("%.6f" % v for v in home))
    root.append(keyframe)
    return home


def indent_xml(elem, level=0):
    pad = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = pad
        for child in elem:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    elif level and (not elem.tail or not elem.tail.strip()):
        elem.tail = pad


def main():
    parser = argparse.ArgumentParser(description="把 URDF_arm 转成 MuJoCo 模型")
    parser.add_argument("--collision", choices=("none", "mesh"), default="none",
                        help="碰撞体：none=只做可视化（默认），mesh=用原始网格")
    args = parser.parse_args()

    if not (SRC / "urdf" / "URDF_arm.urdf").is_file():
        sys.exit("找不到源 URDF：%s" % (SRC / "urdf" / "URDF_arm.urdf"))
    ASSETS.mkdir(exist_ok=True)

    print("1. 复制网格")
    mapping, notes = copy_meshes()
    for note in notes:
        print("   " + note)
    print("   %d 个网格 -> %s" % (len(mapping), MESH_DIR))

    print("2. 生成中间 URDF")
    urdf_path = write_urdf(mapping)
    print("   %s" % urdf_path)

    print("3. 用 MuJoCo 编译并导出 MJCF")
    dump, model = dump_mjcf(urdf_path)
    print("   nbody=%d njnt=%d nq=%d nv=%d nmesh=%d  总质量 %.4f kg"
          % (model.nbody, model.njnt, model.nq, model.nv, model.nmesh, model.body_mass.sum()))

    print("4. 补地面/光照/伺服/keyframe")
    tree = ET.parse(dump)
    home = decorate(tree, args.collision == "mesh", model)
    dump.unlink()

    out = HERE / "URDF_arm.xml"
    indent_xml(tree.getroot())
    out.write_text(ET.tostring(tree.getroot(), encoding="unicode") + "\n", encoding="utf-8")
    print("   零位 keyframe：%s" % " ".join("%.4f" % v for v in home))

    # 自检：生成的 MJCF 必须能编译
    check = mujoco.MjModel.from_xml_path(str(out))
    print()
    print("输出：%s" % out)
    print("校验：编译通过，nq=%d nu=%d nkey=%d 碰撞=%s"
          % (check.nq, check.nu, check.nkey, args.collision))
    print()
    print("接着运行：python view_arm.py")


if __name__ == "__main__":
    main()
