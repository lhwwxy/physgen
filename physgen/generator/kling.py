from __future__ import annotations

"""KlingGenerator（可灵图生视频，M4 交付，接口已就绪）。

接入方式：
    密钥放 .env（PHYSGEN_API_KEY），不写在代码/配置里
    config.yaml: generator.provider: kling

对接要点：可灵 image2video：提交任务 -> 轮询 task 状态 -> 下载 mp4。
首帧来自 Simulator 渲染；segment_prompt 作为正向提示词；
GT 轨迹以负向提示/运镜参数近似表达（可灵暂无逐轨迹 conditioning）。
"""

from ..types import SimResult, VideoClip
from .base import VideoGenerator


class KlingVideoGenerator(VideoGenerator):

    API_BASE = "https://api.klingai.com/v1"   # 可迭官方 API 常量，也可经 base_url 覆盖

    def __init__(self, api_key: str | None = None, poll_interval_s: float = 2.0,
                 base_url: str | None = None):
        self.api_key = api_key
        self.poll_interval_s = poll_interval_s
        self.base_url = (base_url or self.API_BASE).rstrip("/")

    def generate(self, first_frame_path: str, sim: SimResult,
                 segment_prompt: str, n_frames: int, out_dir: str) -> VideoClip:
        if not self.api_key:
            raise RuntimeError("缺少密钥：请在 .env 配置 PHYSGEN_API_KEY（或 config.yaml [generator].api_key）")
        # TODO(M4):
        # 1. POST {API_BASE}/videos/image2video  (image=首帧, prompt=segment_prompt)
        # 2. 轮询 GET /videos/{task_id} 直到 succeeded
        # 3. 下载 mp4 -> out_dir/clip.mp4，抽帧回填 frame_paths
        raise NotImplementedError("可灵适配器将在 M4 交付")
