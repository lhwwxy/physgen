from __future__ import annotations

"""3D 首帧渲染器（matplotlib 软件渲染，无 GPU 依赖）。

M1 阶段的 3D 占位渲染：真实透视相机 + 立方体网格 + 地块平面 + 1.7m 人物参照。
数据层（GT）本来就是三维的，这里只是把渲染从 2D 立面升级为 3D 透视。
M2 接入 Genesis 后由其光栅化渲染器替换，接口不变。

坐标约定：GT 数据 y 轴竖直向上；matplotlib 3D 默认 z 轴竖直向上，
因此渲染时做映射 (x, y_up, z_depth) -> plot(X=x, Y=z_depth, Z=y_up)。
"""

import os

# matplotlib 为可选依赖（real extra），缺失时由调用方降级 2D 渲染
plt = None
Poly3DCollection = None


def _load_matplotlib():
    global plt, Poly3DCollection
    if plt is not None:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt  # noqa: E402
        from mpl_toolkits.mplot3d.art3d import \
            Poly3DCollection as _P3  # noqa: E402
        plt, Poly3DCollection = _plt, _P3
    except ImportError as e:
        raise RuntimeError(
            "3D 渲染需要 matplotlib（部署时安装：pip install -e \".[real]\"）") from e


def _cuboid_faces(cx: float, cy_up: float, cz: float,
                  l: float, w: float, h: float):
    """中心 (cx, cy_up, cz)、尺寸 l×h×w（长/高/宽）的立方体六个面。

    输入为 GT 约定（y 竖直向上），输出已是 plot 坐标（Z 竖直向上）。
    """
    x0, x1 = cx - l / 2, cx + l / 2
    z0, z1 = cy_up - h / 2, cy_up + h / 2   # 竖直方向 -> plot Z
    y0, y1 = cz - w / 2, cz + w / 2         # 深度方向 -> plot Y
    return [
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],  # 底
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],  # 顶
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],  # 前(近相机)
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],  # 后
        [(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z1)],  # 左
        [(x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0)],  # 右
    ]


def _shade(rgb: tuple, f: float) -> tuple:
    return tuple(min(1.0, c * f) for c in rgb)


MATERIAL_RGB = {
    "wood": (0.63, 0.43, 0.24), "brick": (0.67, 0.31, 0.24),
    "concrete": (0.71, 0.71, 0.69), "stone": (0.55, 0.55, 0.57),
    "steel": (0.39, 0.43, 0.49), "default": (0.59, 0.59, 0.59),
}


def render_first_frame_3d(objects, ground, states, out_path: str,
                          view: tuple = (22, -55)) -> str:
    """objects: list[SceneObject]; states: {obj_id: ObjectFrameState}(首帧)。"""
    _load_matplotlib()
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    L, W = ground.length_m, ground.width_m
    max_h = max([o.dims_m[2] for o in objects] + [2.0])
    z_top = max(max_h * 1.8, 4.0)

    # 天空背景 + 地块（plot 坐标：地面是 Z=0 的水平面）
    # 每片瓦片一个独立 collection：matplotlib 3D 按 collection 平均深度排序，
    # 整块地面一个 collection 会导致深度排序失真（房子看起来插进地面）
    ax.set_facecolor((0.53, 0.75, 0.92))
    for i in range(int(L)):
        for j in range(int(W)):
            tile = [(i, j, 0), (i + 1, j, 0), (i + 1, j + 1, 0), (i, j + 1, 0)]
            ax.add_collection3d(Poly3DCollection(
                [tile], facecolors=[(0.47, 0.63, 0.35)],
                edgecolors=(0.36, 0.48, 0.27), linewidths=0.4))

    # 物体立方体（按材质着色 + 明暗面）
    for obj in objects:
        st = states[obj.id]
        l, _w, h = obj.dims_m
        rgb = MATERIAL_RGB.get(obj.material, MATERIAL_RGB["default"])
        faces = _cuboid_faces(st.position[0], st.position[1], st.position[2],
                              l, h, _w)
        for fi, face in enumerate(faces):
            f = (0.62, 1.0, 0.92, 0.70, 0.78, 0.84)[fi]
            ax.add_collection3d(Poly3DCollection(
                [face], facecolors=[_shade(rgb, f)],
                edgecolors=(0.15, 0.15, 0.15), linewidths=0.8, zorder=2))

    # 1.7m 人物比例参照（细高方柱，立在地块角落，不与物体重叠）
    px, pz = L * 0.06, W * 0.08
    person_rgb = (0.91, 0.78, 0.66)
    for fi, face in enumerate(_cuboid_faces(px, 0.85, pz, 0.35, 0.25, 1.7)):
        f = (0.6, 1.0, 0.92, 0.7, 0.78, 0.84)[fi]
        ax.add_collection3d(Poly3DCollection(
            [face], facecolors=[_shade(person_rgb, f)],
            edgecolors=(0.2, 0.15, 0.1), linewidths=0.5, zorder=3))
    ax.text(px, pz, 2.0, "1.7m", fontsize=9, color=(0.1, 0.1, 0.1))

    # 相机与坐标范围（Z 竖直向上）
    ax.set_xlim(0, L)
    ax.set_ylim(0, W)
    ax.set_zlim(0, z_top)
    ax.set_box_aspect((L, W, z_top))
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path
