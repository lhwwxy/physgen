from __future__ import annotations

"""VideoGenerator 接口：首帧 + 控制信号 + 提示词 -> VideoClip。"""

from ..types import SimResult, VideoClip


class VideoGenerator:
    """所有视频生成器的基类。"""

    def generate(self, first_frame_path: str, sim: SimResult,
                 segment_prompt: str, n_frames: int, out_dir: str) -> VideoClip:
        raise NotImplementedError
