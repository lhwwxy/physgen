from __future__ import annotations

"""MockRepairer：对异常帧打"已修复"标记（重绘受影响区域），供演示闭环。

真实重绘后端接入后（M4 的 http_generic），由带 mask 的 inpainting API 替换。
Mock 的修复效果 = 异常区域亮度/色彩校正 + 重新落盘帧 PNG，
并在再审查时以更低的视觉噪声评估（fix_factor < 1）。
"""

import os

from PIL import Image, ImageEnhance

from ..types import AuditReport, SimResult, VideoClip
from .base import Repairer


class MockRepairer(Repairer):

    def repair_segment(self, clip: VideoClip, sim: SimResult, report: AuditReport,
                       out_dir: str) -> VideoClip:
        os.makedirs(out_dir, exist_ok=True)
        new_paths = []
        for i, fp in enumerate(clip.frame_paths):
            img = Image.open(fp).convert("RGB")
            img = ImageEnhance.Color(img).enhance(1.15)  # 模拟重绘校正
            path = os.path.join(out_dir, f"frame_{i:04d}.png")
            img.save(path)
            new_paths.append(path)
        return VideoClip(segment_index=clip.segment_index, frame_paths=new_paths,
                         fps=clip.fps, video_path=None)

    def repair_frames(self, clip: VideoClip, sim: SimResult,
                      frame_indices: list[int], out_dir: str) -> VideoClip:
        """只重绘异常帧，其余帧原样复用（邻帧一致性：仅校正异常区域色彩偏差）。"""
        os.makedirs(out_dir, exist_ok=True)
        new_paths = []
        fix_set = set(frame_indices)
        for i, fp in enumerate(clip.frame_paths):
            img = Image.open(fp).convert("RGB")
            if i in fix_set:
                img = ImageEnhance.Brightness(img).enhance(1.03)
                img = ImageEnhance.Color(img).enhance(1.08)
            path = os.path.join(out_dir, f"frame_{i:04d}.png")
            img.save(path)
            new_paths.append(path)
        return VideoClip(segment_index=clip.segment_index, frame_paths=new_paths,
                         fps=clip.fps, video_path=None)
