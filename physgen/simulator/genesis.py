from __future__ import annotations

"""GenesisSimulator：真实 3D 物理模拟 + Rasterizer 光栅化渲染。

坐标约定：GT 数据 y 轴竖直向上；Genesis/MuJoCo z 轴竖直向上。
映射：our (x, y_up, z_depth) <-> genesis (x, z_depth, y_up)。

事件表达：
- fall：事件开始前每帧"托住"物体（强制 set_pos）；t_start 时刻抬升到
  drop_height 后释放，由引擎重力接管（碰撞/落地由求解器真实计算）
- move/collide：M3+ 扩展（施加速度/力）
"""

import os

from ..types import ObjectFrameState, SimResult
from .base import Simulator

_gs = None          # 模块级单例，避免重复 init
_INIT_OK = False


def _ensure_genesis():
    global _gs, _INIT_OK
    if _INIT_OK:
        return _gs
    try:
        import genesis as gs
        gs.init(backend=gs.cpu)  # M4 上 Metal 后端待验证，CPU 已足够稳
        _gs = gs
        _INIT_OK = True
        return gs
    except ImportError as e:
        raise RuntimeError(
            "genesis 未安装：pip install genesis-world torch；"
            "或在 config.yaml 中改用 simulator.provider: mock") from e


def _to_genesis(pos) -> tuple:
    """our (x, y_up, z_depth) -> genesis (x, z, y)。"""
    return (float(pos[0]), float(pos[2]), float(pos[1]))


def _to_our(pos) -> tuple:
    """genesis (x, z, y) -> our (x, y_up, z_depth)。"""
    return (float(pos[0]), float(pos[2]), float(pos[1]))


class GenesisSimulator(Simulator):

    def __init__(self, fps: int = 24, resolution: tuple = (1280, 720),
                 renderer: str = "3d"):
        self.fps = fps
        self.resolution = resolution
        self.renderer = renderer
        self.spec = None
        self.run_dir = "runs/current"
        self._scene = None
        self._cam = None
        self._entities: dict[str, object] = {}
        self._rest_pos: dict[str, tuple] = {}     # genesis 坐标
        self._released: set[str] = set()

    # ---------- 场景构建 ----------

    def build(self, spec) -> None:  # noqa: ANN001
        super().build(spec)
        gs = _ensure_genesis()
        L, W = spec.ground.length_m, spec.ground.width_m
        max_h = max([o.dims_m[2] for o in spec.objects] + [2.0])

        self._scene = gs.Scene(
            show_viewer=False,
            renderer=gs.renderers.Rasterizer(),
            # 物理步长固定细粒度（实测 dt>1/45 时高速落体会隧穿地面），
            # 每个视频帧内步进 steps_per_frame 次
            sim_options=gs.options.SimOptions(dt=self._sim_dt()),
        )
        # 地块：薄盒子（Plane 是无限大平面，无法表达地块边界）
        self._scene.add_entity(morph=gs.morphs.Box(
            size=(L, W, 0.1), pos=(L / 2, W / 2, -0.05), fixed=True))
        # 物体（材质配色，便于人眼/VLM 区分与核对）
        from .render3d import MATERIAL_RGB
        for obj in spec.objects:
            l, w, h = obj.dims_m  # our (长, 宽, 高)
            rgb = MATERIAL_RGB.get(obj.material, MATERIAL_RGB["default"])
            self._entities[obj.id] = self._scene.add_entity(
                morph=gs.morphs.Box(size=(l, w, h), pos=_to_genesis(obj.position),
                                    fixed=not obj.is_dynamic),
                surface=gs.surfaces.Default(color=(*rgb, 1.0)))
            self._rest_pos[obj.id] = _to_genesis(obj.position)
        # 相机：侧前方斜俯视，保证房子与前景动态物体同框
        self._cam = self._scene.add_camera(
            pos=(L * 1.45, -W * 1.0, max_h * 2.4),
            lookat=(L * 0.5, W * 0.55, max_h * 0.35),
            res=tuple(self.resolution), fov=50,
        )
        self._scene.build()

    # ---------- 段模拟 ----------

    def _sim_dt(self) -> float:
        """物理步长固定 1/120（防隧穿），与视频帧率解耦。"""
        return 1.0 / 120.0

    def _steps_per_frame(self) -> int:
        return max(1, round((1.0 / self.fps) / self._sim_dt()))

    def _apply_event_holds(self, t: float) -> None:
        """fall 事件：开始前托住；开始瞬间抬升到 drop_height 并释放。"""
        for ev in (self.spec.events or []):
            if ev.type != "fall":
                continue
            for oid in ev.actors:
                if oid not in self._entities:
                    continue
                ent = self._entities[oid]
                drop = float(ev.params.get("drop_height_m", 2.0))
                if t < ev.t_start:
                    ent.set_pos(self._rest_pos[oid])       # 托住
                elif oid not in self._released:
                    gx, gz, gy = self._rest_pos[oid]
                    ent.set_pos((gx, gz, gy + drop))       # 抬升后释放
                    self._released.add(oid)

    def simulate_segment(self, seg_idx: int) -> SimResult:
        seg = self.spec.segments[seg_idx]
        n_frames = max(2, int((seg.t_end - seg.t_start) * self.fps))

        # 段首帧：先定位到段起始状态再渲染
        self._apply_event_holds(seg.t_start)
        out_dir = os.path.join(self.run_dir, f"seg_{seg_idx:02d}")
        os.makedirs(out_dir, exist_ok=True)
        first_frame = os.path.join(out_dir, "first_frame.png")
        rgb, _depth, _seg, _ = self._cam.render()
        from PIL import Image
        Image.fromarray(rgb).save(first_frame)

        # 逐帧推进，记录 GT（每视频帧内多次物理步进）
        gt: dict[str, list[ObjectFrameState]] = {o.id: [] for o in self.spec.objects}
        for i in range(n_frames):
            t = seg.t_start + i / self.fps
            for _ in range(self._steps_per_frame()):
                self._apply_event_holds(t)
                self._scene.step()
            for obj in self.spec.objects:
                ent = self._entities[obj.id]
                p = _to_our(ent.get_pos())
                v = _to_our(ent.get_vel())
                gt[obj.id].append(ObjectFrameState(
                    obj.id, p, v, obj.dims_m))
        return SimResult(
            segment_index=seg_idx, first_frame_path=first_frame,
            fps=self.fps, n_frames=n_frames, gt=gt,
            control_meta={"type": "genesis_rasterizer",
                          "depth": False, "segmentation": False},
        )
