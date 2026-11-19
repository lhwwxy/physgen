from __future__ import annotations

"""SeedanceGenerator：Seedance 系列视频生成适配器（OpenAI Videos 协议）。

接入方式：
    密钥放 .env（PHYSGEN_API_KEY）
    config.yaml（model 与 base_url 均必填，代码不内置任何端点）:
      generator:
        provider: seedance
        model: <你的视频模型名>
        base_url: <OpenAI 兼容网关>/v1 或官方 Ark https://ark.cn-beijing.volces.com/api/v3

协议（探测确认，2026-11）：
- 提交：POST {base}/videos  JSON {model, prompt, seconds}  -> {id, status}
- 图生视频：multipart POST {base}/videos，字段 input_reference=<png 文件>
- 轮询：GET {base}/videos/{id} -> {status: queued|running|succeeded|failed, progress}
- 下载：GET {base}/videos/{id}/content -> mp4 二进制
- 下载后用 imageio（genesis 依赖树自带 ffmpeg）抽帧回填 frame_paths
"""

import base64
import os

from ..types import SimResult, VideoClip
from .base import VideoGenerator


class SeedanceGenerator(VideoGenerator):

    def __init__(self, api_key: str | None, model: str | None,
                 base_url: str | None = None,
                 poll_interval_s: float = 5.0, max_poll_time_s: int = 900):
        import requests
        self.requests = requests
        if not model or not base_url:
            raise ValueError(
                "SeedanceGenerator 需要 config.yaml [generator] 段配置 model 和 "
                "base_url（中转网关如 https://your-gateway/v1，官方 Ark 为 "
                "https://ark.cn-beijing.volces.com/api/v3），代码不内置任何端点")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.poll_interval_s = poll_interval_s
        self.max_poll_time_s = max_poll_time_s

    # ---------- 协议适配 ----------

    def _is_openai_style(self) -> bool:
        return "/v1" in self.base_url  # 中转网关为 OpenAI 兼容端点

    @staticmethod
    def _image_data_uri(path: str) -> str:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{b64}"

    @staticmethod
    def _build_prompt(segment_prompt: str, duration_s: float, fps: int) -> str:
        """物理一致性约束写进提示词；官方 Ark 端点追加参数尾缀。"""
        physics = ("真实物理规律：物体尺寸比例严格符合参照，重物下落加速、"
                   "落地有冲击感，镜头固定平视。")
        return f"{segment_prompt}。{physics}"

    def _submit(self, first_frame_path: str | None, prompt: str,
                seconds: float) -> str:
        h = {"Authorization": f"Bearer {self.api_key}"}
        if self._is_openai_style():
            # OpenAI Videos 协议：JSON 提交；首帧用 multipart input_reference
            if first_frame_path:
                with open(first_frame_path, "rb") as f:
                    r = self.requests.post(
                        f"{self.base_url}/videos", headers=h,
                        data={"model": self.model, "prompt": prompt,
                              "seconds": str(int(round(seconds)))},
                        files={"input_reference": ("first_frame.png", f, "image/png")},
                        timeout=120)
            else:
                r = self.requests.post(
                    f"{self.base_url}/videos", headers=h,
                    json={"model": self.model, "prompt": prompt,
                          "seconds": str(int(round(seconds)))}, timeout=120)
        else:
            # 官方 Ark 协议：contents/generations/tasks，首帧为 data URI
            content = [{"type": "text", "text": prompt}]
            if first_frame_path:
                content.append({"type": "image_url",
                                "image_url": {"url": self._image_data_uri(first_frame_path)},
                                "role": "first_frame"})
            r = self.requests.post(
                f"{self.base_url}/contents/generations/tasks",
                json={"model": self.model,
                      "content": [{"role": "user", "content": content}]},
                headers={**h, "Content-Type": "application/json"}, timeout=60)
        r.raise_for_status()
        d = r.json()
        return d.get("id") or d.get("task_id")

    def _poll(self, task_id: str) -> tuple[str, str | None]:
        """返回 (status, video_url)。"""
        h = {"Authorization": f"Bearer {self.api_key}"}
        import time
        deadline = time.time() + self.max_poll_time_s
        while time.time() < deadline:
            time.sleep(self.poll_interval_s)
            if self._is_openai_style():
                q = self.requests.get(f"{self.base_url}/videos/{task_id}",
                                      headers=h, timeout=60)
                d = q.json()
                status = d.get("status")
                url = (d.get("metadata") or {}).get("url") or d.get("url")
                if url and url.startswith("/"):
                    origin = self.base_url.split("/v1")[0]
                    url = origin + url          # 网关返回相对路径
                if status == "succeeded" and not url:
                    url = f"{self.base_url}/videos/{task_id}/content"
            else:
                q = self.requests.get(
                    f"{self.base_url}/contents/generations/tasks/{task_id}",
                    headers=h, timeout=60)
                d = q.json()
                status = d.get("status")
                url = (d.get("content") or {}).get("video_url") if status == "succeeded" else None
            print(f"[seedance] task {task_id} status={status}")
            if status in ("succeeded", "completed") and url:
                return "succeeded", url
            if status in ("failed", "cancelled", "error"):
                raise RuntimeError(f"Seedance 任务失败: {d}")
        raise RuntimeError(f"Seedance 任务超时: {task_id}")

    # ---------- 主流程 ----------

    def generate(self, first_frame_path: str, sim: SimResult,
                 segment_prompt: str, n_frames: int, out_dir: str) -> VideoClip:
        if not self.api_key:
            raise RuntimeError("缺少密钥：请在 .env 配置 PHYSGEN_API_KEY（或 config.yaml [generator].api_key）")
        os.makedirs(out_dir, exist_ok=True)
        seconds = n_frames / sim.fps
        # Seedance 2.5 时长约束 4-30s：段长过短时钳到下限（GT 时间轴不受影响）
        if not 4.0 <= seconds <= 30.0:
            print(f"[seedance] 段时长 {seconds:.1f}s 超出模型支持的 4-30s，钳制到 "
                  f"{min(max(seconds, 4.0), 30.0):.0f}s")
            seconds = min(max(seconds, 4.0), 30.0)
        prompt = self._build_prompt(segment_prompt, seconds, sim.fps)
        task_id = self._submit(first_frame_path, prompt, seconds)
        print(f"[seedance] 已提交任务 {task_id}")
        _status, url = self._poll(task_id)

        mp4_path = os.path.join(out_dir, "clip.mp4")
        h = {"Authorization": f"Bearer {self.api_key}"}
        resp = self.requests.get(url, headers=h, timeout=300)
        resp.raise_for_status()
        with open(mp4_path, "wb") as f:
            f.write(resp.content)
        frame_paths = self._extract_frames(mp4_path, out_dir)
        return VideoClip(segment_index=sim.segment_index,
                         frame_paths=frame_paths, fps=sim.fps,
                         video_path=mp4_path)

    @staticmethod
    def _extract_frames(mp4_path: str, out_dir: str) -> list[str]:
        import imageio.v3 as iio
        frame_paths = []
        for i, frame in enumerate(iio.imread(mp4_path, plugin="pyav")):
            path = os.path.join(out_dir, f"frame_{i:04d}.png")
            iio.imwrite(path, frame, plugin="pillow")
            frame_paths.append(path)
        return frame_paths
