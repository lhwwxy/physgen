from __future__ import annotations

"""MockSimulator：参数化 2D 模拟，纯 Python 计算物理真值（GT）。

- 首帧：PIL 渲染简易立面图（含 1.7m 人物比例参照，缓解尺度幻觉）
- GT：每物体每帧的位置/速度/包围盒；下落事件按匀加速运动学计算
- Genesis 不可用时的降级路径，接口与 GenesisSimulator 完全一致
"""

import math
import os

from PIL import Image, ImageDraw

from ..types import (ObjectFrameState, SceneObject, SimResult, fall_time)


def ffmpeg_available() -> bool:
    import shutil
    return shutil.which("ffmpeg") is not None


def frames_to_video(frame_paths: list[str], fps: int, out_path: str) -> bool:
    """用 ffmpeg 合成 mp4；失败/不可用返回 False（保留帧序列）。"""
    if not frame_paths or not ffmpeg_available():
        return False
    first = frame_paths[0]
    pattern = first.replace("frame_0000", "frame_%04d")
    cmd = (f'ffmpeg -y -loglevel error -framerate {fps} '
           f'-i "{pattern}" -c:v libx264 -pix_fmt yuv420p "{out_path}"')
    os.system(cmd)  # noqa: S605 — 受控的固定参数 ffmpeg 调用
    return os.path.exists(out_path)


def frames_to_gif(frame_paths: list[str], fps: int, out_path: str,
                  max_fps: int = 10, scale: float = 0.5) -> bool:
    """PIL 直接合成 GIF —— 无 ffmpeg 时的可观看降级产物。"""
    if not frame_paths:
        return False
    imgs = [Image.open(p).convert("RGB") for p in frame_paths]
    w, h = imgs[0].size
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    imgs = [im.resize(size, Image.BILINEAR) for im in imgs]
    duration = int(1000 / min(fps, max_fps))
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                 duration=duration, loop=0)
    return os.path.exists(out_path)


class MockSimulator:

    def __init__(self, fps: int = 24, resolution: tuple = (1280, 720),
                 renderer: str = "3d"):
        self.fps = fps
        self.resolution = resolution
        self.renderer = renderer          # 3d（透视）| 2d（立面占位）
        self.spec = None

    def build(self, spec) -> None:  # noqa: ANN001 — 与基类签名一致
        self.spec = spec

    # ---------- 物理真值 ----------

    def _object_state_at(self, obj: SceneObject, t: float) -> ObjectFrameState:
        pos = obj.position
        vel = (0.0, 0.0, 0.0)
        for ev in (self.spec.events or []):
            if obj.id in ev.actors and ev.type == "fall" and ev.t_start <= t <= ev.t_end:
                h0 = float(ev.params.get("drop_height_m", 2.0))
                dt = t - ev.t_start
                y = h0 - 0.5 * 9.8 * dt * dt
                rest = obj.dims_m[2] / 2
                if y <= rest:
                    y = rest
                    vy = 0.0
                else:
                    vy = -9.8 * dt
                pos = (obj.position[0], y, obj.position[2])
                vel = (0.0, vy, 0.0)
        return ObjectFrameState(obj.id, pos, vel, obj.dims_m)

    def simulate_segment(self, seg_idx: int) -> SimResult:
        seg = self.spec.segments[seg_idx]
        n_frames = max(2, int((seg.t_end - seg.t_start) * self.fps))
        gt: dict[str, list[ObjectFrameState]] = {}
        for obj in self.spec.objects:
            states = []
            for i in range(n_frames):
                t = seg.t_start + i / self.fps
                states.append(self._object_state_at(obj, t))
            gt[obj.id] = states

        first_frame = self._render_first_frame(seg_idx, gt)
        return SimResult(
            segment_index=seg_idx, first_frame_path=first_frame,
            fps=self.fps, n_frames=n_frames, gt=gt,
            control_meta={"type": "mock_2d_layout", "ground": vars(self.spec.ground)},
        )

    # ---------- 首帧渲染 ----------

    def _render_first_frame(self, seg_idx: int, gt: dict) -> str:
        out_dir = self._run_dir(seg_idx)
        path = os.path.join(out_dir, "first_frame.png")
        if self.renderer == "3d":
            from .render3d import render_first_frame_3d
            states0 = {oid: sts[0] for oid, sts in gt.items()}
            try:
                return render_first_frame_3d(self.spec.objects, self.spec.ground,
                                             states0, path)
            except RuntimeError as e:
                print(f"[mock-simulator] {e}；本次降级为 2D 立面渲染")
                self.renderer = "2d"
        return self._render_first_frame_2d(seg_idx, gt, path)

    def _render_first_frame_2d(self, seg_idx: int, gt: dict, path: str) -> str:
        W, H = self.resolution
        ground = self.spec.ground
        # 世界宽度留 1.2 倍边距
        world_w = ground.length_m * 1.2
        scale = W / world_w
        horizon = int(H * 0.75)

        img = Image.new("RGB", (W, H), (135, 190, 235))  # 天空
        d = ImageDraw.Draw(img)
        d.rectangle([0, horizon, W, H], fill=(120, 160, 90))  # 地面

        def to_px(x_m: float, y_m: float) -> tuple[int, int]:
            # x: 世界横向 -> 像素；y: 高度 -> 向上
            px = int((x_m + ground.length_m * 0.1) * scale)
            py = horizon - int(y_m * scale * 0.55)  # 伪透视压缩
            return px, py

        # 地块边界线
        p1, p2 = to_px(0, 0), to_px(ground.length_m, 0)
        d.line([p1, (p2[0], p1[1])], fill=(70, 90, 50), width=4)

        colors = {"wood": (160, 110, 60), "brick": (170, 80, 60),
                  "concrete": (180, 180, 175), "stone": (140, 140, 145),
                  "steel": (100, 110, 125), "default": (150, 150, 150)}
        for obj in self.spec.objects:
            states = gt[obj.id]
            st = states[0]
            l, _w, h = obj.dims_m
            cx, cy = st.position[0], st.position[1]
            x0, y0 = to_px(cx - l / 2, cy + h / 2)
            x1, y1 = to_px(cx + l / 2, cy - h / 2)
            d.rectangle([x0, y0, x1, y1], fill=colors.get(obj.material, colors["default"]),
                        outline=(40, 40, 40), width=2)

        # 1.7m 人物比例参照
        px, py = to_px(ground.length_m * 0.08, 0)
        top = to_px(0, 1.7)[1]
        d.ellipse([px - 6, top, px + 6, top + 14], fill=(230, 200, 170))
        d.line([px, top + 14, px, py], fill=(230, 200, 170), width=6)
        d.text((px + 12, top + 4), "1.7m", fill=(30, 30, 30))

        img.save(path)
        return path

    def _run_dir(self, seg_idx: int) -> str:
        base = getattr(self, "run_dir", None) or "runs/current"
        d = os.path.join(base, f"seg_{seg_idx:02d}")
        os.makedirs(d, exist_ok=True)
        return d
