from __future__ import annotations

"""Repairer 接口：段级修复 + 帧级修复（全局阶段）。"""

from ..types import AuditReport, SimResult, VideoClip


class Repairer:
    """所有修复器的基类。"""

    def repair_segment(self, clip: VideoClip, sim: SimResult, report: AuditReport,
                       out_dir: str) -> VideoClip:
        """段级修复：局部重绘优先，失败则带首-尾帧约束整段重生成。"""
        raise NotImplementedError

    def repair_frames(self, clip: VideoClip, sim: SimResult,
                      frame_indices: list[int], out_dir: str) -> VideoClip:
        """帧级修复（全局阶段）：逐帧 mask 局部重绘，带邻帧一致性约束。"""
        raise NotImplementedError
