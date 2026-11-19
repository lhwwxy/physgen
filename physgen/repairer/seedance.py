from __future__ import annotations

"""SeedanceRepairer（M4）：真实修复器。

两级修复策略：
- 段级 repair_segment：把 VLM 异常的修复建议编译进提示词，用 GT 首帧
  + Seedance 图生视频**整段重生成**（修复效果最可靠）
- 帧级 repair_frames：Seedream 图生图逐帧重绘异常帧
  （images/generations + image 字段，出图后缩回原帧尺寸替换）
- 任何 API 失败 → 保留原帧/原片段（带日志），不中断流水线

接入方式：
    config.yaml: repairer: {provider: seedance, model: ..., image_model: ...,
                            base_url: ..., api_key: ...}
Pipeline 会在运行时注入 spec（GT 期望值）与 segment_prompt（当前段提示词）。
"""

import base64
import io
import math
import os
import time

import requests
from PIL import Image

from ..generator.seedance import SeedanceGenerator
from ..repairer.base import Repairer
from ..repairer.mock import MockRepairer
from ..types import AuditReport, SimResult, VideoClip

PHYSICS_SUFFIX = ("真实物理规律：严格保持首帧场景的构图、物体尺寸比例与光照，"
                  "重物下落加速、落地有冲击但不夸张，镜头固定。")

_MIN_IMG_PX = 921600      # 网关对 seedream 的最小像素数限制
_MAX_RETRY = 3
_RETRY_WAIT_S = 25        # 网关 RPM 限制 429 的退避


class SeedanceRepairer(Repairer):

    def __init__(self, api_key: str | None,
                 model: str | None = None,
                 image_model: str | None = None,
                 base_url: str | None = None,
                 max_frames_per_round: int = 4):
        # 模型名与端点均由 config.yaml [repairer] 提供，代码不内置任何默认值
        if not base_url:
            raise ValueError(
                "SeedanceRepairer 需要 config.yaml [repairer] 段配置 base_url，"
                "代码不内置任何端点")
        self.api_key = api_key
        self.image_model = image_model
        self.max_frames_per_round = max_frames_per_round
        self.base_url = base_url.rstrip("/")
        # 复用生成器做段级重生成
        self._gen = SeedanceGenerator(api_key=api_key, model=model,
                                      base_url=base_url)
        self._mock = MockRepairer()
        # Pipeline 注入：
        self.spec = None            # SceneSpec（GT 期望值）
        self.segment_prompt = ""    # 当前段提示词
        self._anomalies_of_round = []  # 帧级修复的异常上下文

    # ---------- 提示词编译 ----------

    def _repair_prompt(self, report: AuditReport) -> str:
        base = self.segment_prompt or "按物理规律运动的真实场景"
        fixes = []
        for a in report.anomalies[:6]:
            tip = a.suggest_fix or a.detail
            if tip:
                fixes.append(f"- {tip}")
        constraint = ""
        if self.spec is not None:
            parts = [f"{o.name} {o.dims_m[0]}×{o.dims_m[1]}×{o.dims_m[2]}米"
                     for o in self.spec.objects]
            constraint = "场景物体与规格：" + "，".join(parts) + "。"
        body = ("修复以下问题：" + "\n".join(fixes)) if fixes else "消除画面中的物理异常"
        return f"{base}。{body}。{constraint}{PHYSICS_SUFFIX}"

    # ---------- 段级：整段重生成 ----------

    def repair_segment(self, clip: VideoClip, sim: SimResult,
                       report: AuditReport, out_dir: str) -> VideoClip:
        if not self.api_key:
            print("[repairer] 缺少 api_key，降级占位修复")
            return self._mock.repair_segment(clip, sim, report, out_dir)
        prompt = self._repair_prompt(report)
        try:
            print(f"[repairer] 段级重生成（Seedance，首帧={os.path.basename(sim.first_frame_path)}）")
            return self._gen.generate(sim.first_frame_path, sim, prompt,
                                      sim.n_frames, out_dir)
        except Exception as e:  # noqa: BLE001
            print(f"[repairer] 段级重生成失败（{e}），保留原片段")
            return clip

    # ---------- 帧级：Seedream 图生图逐帧重绘 ----------

    def repair_frames(self, clip: VideoClip, sim: SimResult,
                      frame_indices: list[int], out_dir: str) -> VideoClip:
        if not self.api_key:
            return self._mock.repair_frames(clip, sim, frame_indices, out_dir)
        os.makedirs(out_dir, exist_ok=True)
        # 修复指令按帧号聚合（VLM 异常的 frame_index 可能对应采样序号，
        # 找不到精确指令时用全段问题的通用修复指令）
        by_frame = {}
        for a in self._anomalies_of_round:
            if a.frame_index is not None:
                by_frame.setdefault(a.frame_index, []).append(a)
        new_paths = list(clip.frame_paths)
        targets = frame_indices[:self.max_frames_per_round]
        for i in targets:
            src = clip.frame_paths[i]
            if not os.path.exists(src):
                continue
            tips = by_frame.get(i) or [a for a in self._anomalies_of_round
                                       if a.frame_index is None]
            instr = "；".join((a.suggest_fix or a.detail) for a in tips[:3]) \
                or "修正画面中的物理错误，保持其余内容不变"
            edited = self._edit_frame(src, instr)
            if edited is None:
                continue  # 修复失败保留原帧
            dst = os.path.join(out_dir, f"frame_{i:04d}.png")
            self._match_original(edited, src).save(dst)
            new_paths[i] = dst
        return VideoClip(segment_index=clip.segment_index, frame_paths=new_paths,
                         fps=clip.fps, video_path=None)

    def set_anomalies(self, anomalies: list) -> None:
        """Pipeline 在每次修复调用前注入当轮审查异常（帧级修复用）。"""
        self._anomalies_of_round = list(anomalies)

    # ---------- Seedream 调用 ----------

    def _edit_frame(self, frame_path: str, instruction: str) -> Image.Image | None:
        # 转 JPEG 压缩载荷：1280×720 PNG 可达数 MB，经本地代理上传极慢
        img = Image.open(frame_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        w, h = img.size
        size = self._clamp_size(w, h)
        payload = {"model": self.image_model,
                   "prompt": f"编辑图像（保持原图构图与风格）：{instruction}",
                   "image": [uri], "size": size, "response_format": "b64_json"}
        for attempt in range(1, _MAX_RETRY + 1):
            try:
                r = requests.post(
                    f"{self.base_url}/images/generations",
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type": "application/json"},
                    json=payload, timeout=(10, 90))
                if r.status_code == 429:
                    print(f"[repairer] {self.image_model} 限流，{ _RETRY_WAIT_S}s 后重试"
                          f"（{attempt}/{_MAX_RETRY}）")
                    time.sleep(_RETRY_WAIT_S)
                    continue
                r.raise_for_status()
                d = r.json()["data"][0]
                if d.get("b64_json"):
                    return Image.open(io.BytesIO(base64.b64decode(d["b64_json"])))
                if d.get("url"):  # 部分网关只给 url
                    img_r = requests.get(d["url"], headers={
                        "Authorization": f"Bearer {self.api_key}"}, timeout=120)
                    img_r.raise_for_status()
                    return Image.open(io.BytesIO(img_r.content))
                return None
            except Exception as e:  # noqa: BLE001
                print(f"[repairer] 帧编辑失败（{e}），重试 {attempt}/{_MAX_RETRY}")
                time.sleep(min(5 * attempt, 15))
        print("[repairer] 帧编辑重试耗尽，保留原帧")
        return None

    @staticmethod
    def _clamp_size(w: int, h: int) -> str:
        """seedream 像素范围 [921600, 4624220]，按原始宽高比缩放到合法区间。"""
        px = w * h
        if px < _MIN_IMG_PX:
            # 向上取到偶数：向下取整会让像素数跌破网关下限（实测 400 请求）
            s = (_MIN_IMG_PX / px) ** 0.5
            w, h = math.ceil(w * s / 2) * 2, math.ceil(h * s / 2) * 2
        elif px > 4624220:
            s = (4624220 / px) ** 0.5
            w, h = int(w * s) // 2 * 2, int(h * s) // 2 * 2
        return f"{w}x{h}"

    @staticmethod
    def _match_original(edited: Image.Image, src_path: str) -> Image.Image:
        """编辑图缩回原帧尺寸（宽高比不一致时中心裁剪再缩放）。"""
        tw, th = Image.open(src_path).size
        img = edited.convert("RGB")
        sw, sh = img.size
        if abs(sw / sh - tw / th) > 0.02:
            target_ratio = tw / th
            if sw / sh > target_ratio:   # 太宽 → 裁两侧
                nw = int(sh * target_ratio)
                x0 = (sw - nw) // 2
                img = img.crop((x0, 0, x0 + nw, sh))
            else:                         # 太高 → 裁上下
                nh = int(sw / target_ratio)
                y0 = (sh - nh) // 2
                img = img.crop((0, y0, sw, y0 + nh))
        return img.resize((tw, th), Image.LANCZOS)
