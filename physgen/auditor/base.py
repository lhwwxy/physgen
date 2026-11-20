from __future__ import annotations

"""PhysicsAuditor 接口：生成结果 vs 物理真值 -> AuditReport。

audit 模式（外部视频质检）双通道：
- audit_expectations：通道① 验收核对 —— 画面 vs 用户请求解析出的期望清单
- audit_background：通道② 物理巡检 —— 全画面物理一致性（含背景物体间冲突）
"""

from ..config import Thresholds
from ..types import AuditReport, SimResult, VideoClip


class PhysicsAuditor:
    """所有审查器的基类。fix_factor：修复后噪声衰减系数（Mock 专用扩展点）。"""

    def __init__(self, thresholds: Thresholds | None = None):
        self.thresholds = thresholds or Thresholds()

    def audit(self, clip: VideoClip, sim: SimResult,
              fix_factor: float = 1.0) -> AuditReport:
        raise NotImplementedError

    # ---------- audit 模式双通道（默认：不支持时按通过处理） ----------

    def audit_expectations(self, clip: VideoClip, spec) -> AuditReport:
        """通道① 验收核对：画面是否符合用户请求解析出的期望清单。"""
        return AuditReport(segment_index=clip.segment_index, checks=[], anomalies=[])

    def audit_background(self, clip: VideoClip, spec) -> AuditReport:
        """通道② 物理巡检：全画面物理一致性（背景物体间冲突/悬浮/穿模等）。"""
        return AuditReport(segment_index=clip.segment_index, checks=[], anomalies=[])
