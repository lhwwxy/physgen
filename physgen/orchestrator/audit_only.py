from __future__ import annotations

"""audit 模式：外部视频质检流水线（平台对接）。

与主流水线（plan→sim→generate→audit→repair）不同，本流程不生成视频，
对**已有视频**做双通道审查 + 优先级修复：

  输入：视频（mp4 或帧序列目录） + 用户原始请求文本
  ① 通道验收核对：画面 vs LLM 从请求解析出的期望清单（白名单原则，
     未提及的外观细节不判异常）
  ② 通道物理巡检：全画面物理一致性（背景物体间冲突/悬浮/穿模等），
     config.audit.background_check 开关控制
  修复优先级：通道① 异常帧先修（直接影响交付），预算内再修通道②
  输出：修复后视频 + 双通道审查报告（HTML）

复用 planner（请求→期望）与 repairer（帧级重绘）模块，本体零改动。
"""

import glob
import os
import time
from pathlib import Path

from ..auditor.base import PhysicsAuditor
from ..config import AppConfig
from ..planner.base import Planner
from ..repairer.base import Repairer
from ..simulator.base import Simulator
from ..simulator.mock import frames_to_gif, frames_to_video
from ..types import (AuditReport, SimResult, UserRequest, VideoClip)
from .providers import (create_auditor, create_planner, create_repairer,
                        create_simulator)
from .report import render_report


class AuditPipeline:
    """外部视频质检：双通道审查 + 优先级修复。

    strict=False（默认）：通道① 参照源 = LLM 从请求解析的期望清单（纯语义）。
    strict=True：先在本地用模拟器（Genesis/mock）按请求模拟一遍拿物理真值，
    通道① 升级为 GT 基审查——轨迹形状、落地时长、重量感等数值期望纳入验收，
    模拟首帧作为构图参照（仅参照位置关系，画风差异不判异常）。
    """

    def __init__(self, cfg: AppConfig, run_dir: str):
        self.cfg = cfg
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        # audit 模式只需要 三个模块（不需要 simulator/generator）
        self.planner: Planner = create_planner(cfg)
        self.auditor: PhysicsAuditor = create_auditor(cfg)
        self.repairer: Repairer = create_repairer(cfg)

    # ---------- 输入 ----------

    def _load_clip(self, video: str, fps_hint: int | None = None) -> VideoClip:
        """mp4 → 抽帧（需 imageio[ffmpeg]）；目录 → 直接读 frame_*.png。"""
        p = Path(video)
        if p.is_dir():
            frames = sorted(glob.glob(os.path.join(p, "frame_*.png"))) or \
                sorted(glob.glob(os.path.join(p, "*.png")))
            if not frames:
                raise FileNotFoundError(f"目录 {p} 中没有 PNG 帧")
            fps = fps_hint or self.cfg.video.fps
            return VideoClip(segment_index=0, frame_paths=frames, fps=fps)
        if not p.is_file():
            raise FileNotFoundError(f"视频不存在：{video}")
        try:
            import imageio.v2 as imageio
        except ImportError as e:
            raise RuntimeError(
                "审查 mp4 需要 imageio[ffmpeg]（pip install -e \".[real]\"），"
                "或直接提供帧序列目录") from e
        out_dir = os.path.join(self.run_dir, "input_frames")
        os.makedirs(out_dir, exist_ok=True)
        reader = imageio.get_reader(str(p))
        meta_fps = reader.get_meta_data().get("fps") or fps_hint or self.cfg.video.fps
        fps = max(1, int(round(meta_fps)))
        frames = []
        for i, frame in enumerate(reader):
            fp = os.path.join(out_dir, f"frame_{i:04d}.png")
            imageio.imwrite(fp, frame)
            frames.append(fp)
        reader.close()
        if not frames:
            raise RuntimeError(f"视频 {video} 抽帧结果为空")
        return VideoClip(segment_index=0, frame_paths=frames, fps=fps)

    # ---------- 主流程 ----------

    def run(self, video: str, prompt: str, strict: bool = False) -> str:
        """返回审查报告路径。strict=True 时先本地模拟出物理真值。"""
        clip = self._load_clip(video)
        spec = self.planner.plan(UserRequest(text=prompt))
        if hasattr(self.auditor, "spec"):
            self.auditor.spec = spec
        if hasattr(self.repairer, "spec"):
            self.repairer.spec = spec
        # 修复器需要的 SimResult 桩：audit 模式无模拟，提供首帧引用即可
        sim = SimResult(segment_index=0, first_frame_path=clip.frame_paths[0],
                        fps=clip.fps, n_frames=len(clip.frame_paths), gt={})
        lane1_fn = self.auditor.audit_expectations

        if strict:  # 本地模拟出 GT：数值物理期望纳入通道①（无 API 费用）
            simulator: Simulator = create_simulator(self.cfg)
            simulator.run_dir = self.run_dir
            simulator.build(spec)
            sim = simulator.simulate_segment(0)
            lane1_fn = self.auditor.audit  # GT 基审查（轨迹/落地时长/重量感）
            print(f"[audit] strict：模拟器({self.cfg.simulator.provider})已出物理真值")

        bg_on = self.cfg.audit.background_check
        lane1 = lane1_fn(clip, sim)
        lane2 = (self.auditor.audit_background(clip, spec)
                 if bg_on else AuditReport(0, [], []))
        repair_log = [
            f"[通道①验收核对] 参照源={'模拟物理真值(GT)' if strict else '请求解析期望'}，"
            f"{len(lane1.anomalies)} 处异常",
            f"[通道②物理巡检] {'启用' if bg_on else '关闭'}，"
            f"{len(lane2.anomalies)} 处异常",
        ]

        budget = max(self.cfg.budget.max_frame_repair_rounds,
                     self.cfg.budget.repair_rounds_per_segment)
        rounds = 0
        while rounds < budget:
            f1 = sorted(lane1.anomalous_frames())
            f2 = sorted(lane2.anomalous_frames())
            targets, lane, anomalies = (f1, "①验收核对", lane1.anomalies) if f1 \
                else (f2, "②物理巡检", lane2.anomalies) if f2 else (None, None, None)
            if not targets:
                break
            rounds += 1
            print(f"[audit] 通道{lane} 第 {rounds} 轮修复：重绘 {len(targets)} 帧")
            if hasattr(self.repairer, "set_anomalies"):
                self.repairer.set_anomalies(anomalies)
            clip = self.repairer.repair_frames(
                clip, sim, targets,
                os.path.join(self.run_dir, f"frame_fix_{rounds}"))
            lane1 = lane1_fn(clip, sim)
            lane2 = (self.auditor.audit_background(clip, spec)
                     if bg_on else AuditReport(0, [], []))
            repair_log.append(
                f"[修复第{rounds}轮] 通道{lane} 重绘 {len(targets)} 帧，"
                f"复审后剩 ①{len(lane1.anomalies)}/②{len(lane2.anomalies)} 处")

        left = len(lane1.anomalies) + len(lane2.anomalies)
        if left:
            repair_log.append(f"[预算耗尽] {left} 处异常降级接受并在报告中标注")
        return self._finalize(clip, prompt, lane1, lane2, repair_log)

    # ---------- 收尾 ----------

    def _finalize(self, clip: VideoClip, prompt: str,
                  lane1: AuditReport, lane2: AuditReport,
                  repair_log: list[str]) -> str:
        final_dir = os.path.join(self.run_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        video_path = os.path.join(final_dir, "audited.mp4")
        gif_path = os.path.join(final_dir, "audited.gif")
        clips_info = [{
            "segment": 0, "frames": len(clip.frame_paths),
            "video": video_path if frames_to_video(
                clip.frame_paths, clip.fps, video_path) else None,
            "gif": gif_path if frames_to_gif(
                clip.frame_paths, clip.fps, gif_path) else None,
        }]
        report_path = render_report(
            self.run_dir, f"[audit 模式] {prompt}",
            [lane1], lane2, repair_log, clips_info)
        print(f"[audit done] 审查报告: {report_path}")
        return report_path


def now_run_dir(base: str = "runs") -> str:
    return os.path.join(base, f"audit_{time.strftime('%Y%m%d_%H%M%S')}")
