from __future__ import annotations

"""MockGenerator：以首帧为底做确定性微扰动生成占位视频。

真实后端（可灵/HTTP 通用适配器）接入后，只需替换本实现，流水线其余部分不动。
输出：帧序列 PNG + （ffmpeg 可用时）mp4。
"""

import os

from PIL import Image

from ..simulator.mock import frames_to_video
from ..types import SimResult, VideoClip
from .base import VideoGenerator


class MockVideoGenerator(VideoGenerator):

    def generate(self, first_frame_path: str, sim: SimResult,
                 segment_prompt: str, n_frames: int, out_dir: str) -> VideoClip:
        os.makedirs(out_dir, exist_ok=True)
        base = Image.open(first_frame_path).convert("RGB")
        W, H = base.size
        frames: list[str] = []
        for i in range(n_frames):
            # 微扰动：水平漂移 + 轻微亮度变化，模拟镜头推移（确定性，可复现）
            shift = int(2.0 * math_sin(i))
            frame = base.transform(
                (W, H), Image.AFFINE, (1, 0, -shift, 0, 1, 0),
                resample=Image.BILINEAR)
            if i % 8 == 0:
                frame = frame.point(lambda p: min(255, int(p * 1.02)))
            path = os.path.join(out_dir, f"frame_{i:04d}.png")
            frame.save(path)
            frames.append(path)
        video_path = os.path.join(out_dir, "clip.mp4")
        ok = frames_to_video(frames, sim.fps, video_path)
        return VideoClip(segment_index=sim.segment_index, frame_paths=frames,
                         fps=sim.fps, video_path=video_path if ok else None)


def math_sin(x: float) -> float:
    import math
    return math.sin(x / 6.0)
