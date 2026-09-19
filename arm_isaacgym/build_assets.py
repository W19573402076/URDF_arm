#!/usr/bin/env python3
"""把 ../URDF_arm 转换成 Isaac Gym 可以直接 load_asset 的资源，输出到 ./assets。

Isaac Gym 的 URDF 解析器是相对 URDF 文件所在目录去找网格的，不认 package:// 前缀，
所以这里要做两件事：
  1. 复制 URDF_arm/meshes/*.STL 到 assets/meshes/
  2. 生成 assets/urdf/URDF_arm.urdf，把 package://URDF_arm/meshes/x.STL 改写成 ../meshes/x.STL

另外，七个 STL 一共约 73 万个三角面片（Link3 一个就 22 万），PhysX 给每个连杆烘焙三角
网格碰撞体非常慢。--collision 可以控制碰撞体怎么生成：

    python build_assets.py                    # mesh: 碰撞体用原始网格，最忠实但加载慢
    python build_assets.py --collision box    # box:  碰撞体用网格的轴对齐包围盒，加载快很多
    python build_assets.py --collision none   # none: 不要碰撞体，纯看外形时用

生成的 URDF 每次都会覆盖，不用手动改。
"""

import argparse
import shutil
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "URDF_arm"
DST = HERE / "assets"


def stl_bounds(path):
    """读二进制 STL 的顶点，返回 (min_xyz, max_xyz)。"""
    data = path.read_bytes()
    declared = struct.unpack("<I", data[80:84])[0]
    if len(data) != 84 + declared * 50:
        # 不是标准二进制 STL，退回按 ASCII 解析
        verts = []
        for line in data.decode("utf-8", "ignore").splitlines():
            parts = line.split()
            if len(parts) == 4 and parts[0] == "vertex":
                verts.append([float(v) for v in parts[1:]])
        if not verts:
            raise ValueError("%s 既不是二进制 STL 也不是 ASCII STL" % path)
        verts = np.asarray(verts, dtype=np.float64)
    else:
        faces = np.frombuffer(data[84:84 + declared * 50], dtype=np.uint8).reshape(declared, 50)
        verts = faces[:, 12:48].copy().view("<f4").reshape(-1, 3).astype(np.float64)

    return verts.min(axis=0), verts.max(axis=0)


def to_local_path(filename, urdf_dir_rel="urdf", mesh_dir_rel="meshes"):
    """package://URDF_arm/meshes/x.STL -> ../meshes/x.STL（相对 URDF 文件所在目录）。"""
    for prefix in ("package://", "file://"):
        if filename.startswith(prefix):
            filename = filename[len(prefix):]
            parts = filename.split("/")
            # 去掉包名（URDF_arm），保留包内路径
            if parts and parts[0] == SRC.name:
                parts = parts[1:]
            filename = "/".join(parts)
    # urdf/ 与 meshes/ 是同级目录，所以要 ../ 回到上一级
    return "../%s" % filename


def indent(elem, level=0, space="  "):
    """Python 3.8 没有 ET.indent，手写一个，只为了输出文件好读。"""
    pad = "\n" + level * space
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + space
        if not elem.tail or not elem.tail.strip():
            elem.tail = pad
        for child in elem:
            indent(child, level + 1, space)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    elif level and (not elem.tail or not elem.tail.strip()):
        elem.tail = pad


def make_box_collision(collision, mesh_path, pad=0.002):
    """把 collision 里的 mesh 换成它的轴对齐包围盒。"""
    lo, hi = stl_bounds(mesh_path)
    size = np.maximum(hi - lo, 1e-3) + pad
    center = (lo + hi) / 2.0

    origin = collision.find("origin")
    if origin is None:
        origin = ET.SubElement(collision, "origin")
        collision.remove(origin)
        collision.insert(0, origin)
    origin.set("xyz", "%.6f %.6f %.6f" % tuple(center))
    origin.set("rpy", "0 0 0")

    geometry = collision.find("geometry")
    for child in list(geometry):
        geometry.remove(child)
    box = ET.SubElement(geometry, "box")
    box.set("size", "%.6f %.6f %.6f" % tuple(size))
    return size


def build(collision_mode):
    src_urdf = SRC / "urdf" / "URDF_arm.urdf"
    src_meshes = SRC / "meshes"
    if not src_urdf.is_file():
        sys.exit("找不到源 URDF：%s" % src_urdf)
    if not src_meshes.is_dir():
        sys.exit("找不到网格目录：%s" % src_meshes)

    dst_urdf_dir = DST / "urdf"
    dst_mesh_dir = DST / "meshes"
    dst_urdf_dir.mkdir(parents=True, exist_ok=True)
    dst_mesh_dir.mkdir(parents=True, exist_ok=True)

    # 1. 复制网格
    copied = 0
    for mesh in sorted(src_meshes.iterdir()):
        if mesh.is_file():
            shutil.copy2(mesh, dst_mesh_dir / mesh.name)
            copied += 1

    # 2. 改写 URDF
    tree = ET.parse(src_urdf)
    root = tree.getroot()

    n_mesh = 0
    n_collision = 0
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for geom_owner in link.findall(tag):
                mesh = geom_owner.find("geometry/mesh")
                if mesh is None:
                    continue
                mesh.set("filename", to_local_path(mesh.get("filename")))
                n_mesh += 1

        if collision_mode == "mesh":
            continue

        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                continue
            if collision_mode == "none":
                link.remove(collision)
            else:
                mesh_file = dst_urdf_dir / mesh.get("filename")
                size = make_box_collision(collision, mesh_file.resolve())
                print("  %-10s 碰撞体 -> box %.3f x %.3f x %.3f m"
                      % (link.get("name"), size[0], size[1], size[2]))
            n_collision += 1

    indent(root)
    out_urdf = dst_urdf_dir / "URDF_arm.urdf"
    header = (
        "<!-- 本文件由 build_assets.py 从 %s 生成，请勿手工修改。\n"
        "     改动：package:// 网格路径改为相对路径（Isaac Gym 要求）；碰撞体模式 = %s -->\n"
        % (src_urdf, collision_mode)
    )
    xml = ET.tostring(root, encoding="unicode")
    out_urdf.write_text('<?xml version="1.0" encoding="utf-8"?>\n' + header + xml + "\n", encoding="utf-8")

    print()
    print("源 URDF      : %s" % src_urdf)
    print("复制网格     : %d 个 -> %s" % (copied, dst_mesh_dir))
    print("改写网格路径 : %d 处" % n_mesh)
    print("碰撞体模式   : %s（处理了 %d 个碰撞体）" % (collision_mode, n_collision))
    print("输出 URDF    : %s" % out_urdf)
    print()
    print("接着运行：python view_arm.py")


def main():
    parser = argparse.ArgumentParser(description="把 URDF_arm 转成 Isaac Gym 资源")
    parser.add_argument("--collision", choices=("mesh", "box", "none"), default="mesh",
                        help="碰撞体生成方式：mesh=原始网格（默认，最忠实但慢），box=包围盒，none=无碰撞")
    build(parser.parse_args().collision)


if __name__ == "__main__":
    main()
