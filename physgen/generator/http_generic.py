from __future__ import annotations

"""GenericHTTPGenerator：通用 HTTP 视频生成适配器（M4 交付，接口已就绪）。

适配任何"提交任务 -> 轮询 -> 下载视频"的平台，只需在 config.yaml 配置：

generator:
  provider: http_generic
  endpoint: https://your-platform/api/generate
  query_path: /status
  payload_template: ...   # 支持 {first_frame} {prompt} {fps} 占位符

    密钥放 .env（PHYSGEN_API_KEY），不写在代码/配置里
"""

from ..types import SimResult, VideoClip
from .base import VideoGenerator


class GenericHTTPGenerator(VideoGenerator):

    def __init__(self, api_key: str | None = None, endpoint: str | None = None,
                 poll_interval_s: float = 2.0, payload_template: dict | None = None):
        self.api_key = api_key
        self.endpoint = endpoint
        self.poll_interval_s = poll_interval_s
        self.payload_template = payload_template or {}

    def generate(self, first_frame_path: str, sim: SimResult,
                 segment_prompt: str, n_frames: int, out_dir: str) -> VideoClip:
        if not self.api_key or not self.endpoint:
            raise RuntimeError("缺少密钥：请在 .env 配置 PHYSGEN_API_KEY 并在 config.yaml [generator] 配置 endpoint")
        # TODO(M4): requests.post(endpoint, json=payload) -> 轮询 -> 下载 -> VideoClip
        raise NotImplementedError("通用 HTTP 适配器将在 M4 交付")
