from __future__ import annotations

"""PhysicsAuditor 接口：生成结果 vs 物理真值 -> AuditReport。"""

from ..config import Thresholds
from ..types import AuditReport, SimResult, VideoClip


class PhysicsAuditor:
    """所有审查器的基类。fix_factor：修复后噪声衰减系数（Mock 专用扩展点）。"""

    def __init__(self, thresholds: Thresholds | None = None):
        self.thresholds = thresholds or Thresholds()

    def audit(self, clip: VideoClip, sim: SimResult,
              fix_factor: float = 1.0) -> AuditReport:
        raise NotImplementedError
