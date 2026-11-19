from __future__ import annotations

"""GenericHTTPRepairer（M4 交付，接口已就绪）。

接入方式：
    密钥放 .env（PHYSGEN_API_KEY）
    config.yaml: repairer.provider: http_generic + endpoint

职责：
- repair_segment：异常时空 mask -> inpainting API（首-尾帧约束）
- repair_frames：逐帧 mask 局部重绘，payload 携带相邻帧做一致性参考
"""

from ..types import AuditReport, SimResult, VideoClip
from .base import Repairer


class GenericHTTPRepairer(Repairer):

    def __init__(self, api_key: str | None = None, endpoint: str | None = None):
        self.api_key = api_key
        self.endpoint = endpoint

    def repair_segment(self, clip: VideoClip, sim: SimResult, report: AuditReport,
                       out_dir: str) -> VideoClip:
        if not self.api_key or not self.endpoint:
            raise RuntimeError("缺少密钥：请在 .env 配置 PHYSGEN_API_KEY 并在 config.yaml [repairer] 配置 endpoint")
        # TODO(M4): 由 report.anomalies 生成时空 mask -> POST inpainting 任务 -> 轮询下载
        raise NotImplementedError("通用重绘适配器将在 M4 交付")

    def repair_frames(self, clip: VideoClip, sim: SimResult,
                      frame_indices: list[int], out_dir: str) -> VideoClip:
        # TODO(M4): 逐帧 mask 局部重绘 + 邻帧一致性参考
        raise NotImplementedError("通用重绘适配器将在 M4 交付")
