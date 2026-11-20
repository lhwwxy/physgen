from __future__ import annotations

"""流水线编排器：段循环（模拟→生成→审查→修复）+ 全局逐帧审查/修复。

流程（设计文档 §3.6）：
  段循环：simulate -> generate -> audit -> (repair, ≤budget 轮) -> checkpoint
  全局：逐帧审查 -> 异常帧直接修复(≤max_frame_repair_rounds) ->
        段内异常帧占比 > frame_anomaly_seg_ratio -> 整段重生成 -> 最终报告
"""

import os
import time

from ..auditor.base import PhysicsAuditor
from ..config import AppConfig
from ..generator.base import VideoGenerator
from ..planner.base import Planner
from ..repairer.base import Repairer
from ..simulator.base import Simulator
from ..simulator.mock import frames_to_gif, frames_to_video
from ..types import (AuditReport, SceneSpec, UserRequest, VideoClip)
from .checkpoint import Checkpoint
from .providers import (create_auditor, create_generator, create_planner,
                        create_repairer, create_simulator)
from .report import render_report


class Pipeline:

    def __init__(self, cfg: AppConfig, run_dir: str):
        self.cfg = cfg
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.planner: Planner = create_planner(cfg)
        self.simulator: Simulator = create_simulator(cfg)
        self.generator: VideoGenerator = create_generator(cfg)
        self.auditor: PhysicsAuditor = create_auditor(cfg)
        self.repairer: Repairer = create_repairer(cfg)
        self.ckpt = Checkpoint(run_dir)
        self.repair_log: list[str] = []
        self._last: dict = {}
        self._fix_state: dict = {}   # seg_index -> 最近一次审查的修复质量系数
        self.global_report = None
        self.simulator.run_dir = run_dir  # MockSimulator 渲染输出目录

    # ---------- 主入口 ----------

    def run(self, request: UserRequest) -> str:
        self._last = {}
        self.repair_log = []
        spec = self.planner.plan(request)
        problems = spec.validate()
        if problems:
            spec = self.planner.replan(spec, problems)
        if hasattr(self.auditor, "spec"):
            self.auditor.spec = spec  # VLM 审查器需要期望值
        if hasattr(self.repairer, "spec"):
            self.repairer.spec = spec  # 真实修复器的 GT 约束
        for p in problems:
            self.repair_log.append(f"[规划修正] {p}")
        self.simulator.build(spec)
        self.ckpt.save(spec, self._config_data(), 0, "segments")
        self._run_segments(spec, start=0)
        self._global_audit_and_repair(spec)
        return self._finalize(spec, request)

    def resume(self, run_id: str | None = None) -> str:
        state = self.ckpt.load()
        if not state:
            raise RuntimeError(f"run 目录 {self.run_dir} 中没有 checkpoint，无法续跑")
        from ..config import from_pipeline_dict, restore_secrets
        self.cfg = restore_secrets(from_pipeline_dict(state["config"]))
        self.planner = create_planner(self.cfg)
        self.simulator = create_simulator(self.cfg)
        self.generator = create_generator(self.cfg)
        self.auditor = create_auditor(self.cfg)
        self.repairer = create_repairer(self.cfg)
        self.simulator.run_dir = self.run_dir
        spec = self._spec_from_state(state)
        if hasattr(self.auditor, "spec"):
            self.auditor.spec = spec  # VLM 审查器需要期望值
        if hasattr(self.repairer, "spec"):
            self.repairer.spec = spec  # 真实修复器的 GT 约束
        self.simulator.build(spec)
        start = state["next_segment"]
        self._recover_done_segments(spec, start)
        print(f"[resume] 从段 {start} 续跑（phase={state['phase']}）")
        self._run_segments(spec, start=start)
        self._global_audit_and_repair(spec)
        request = UserRequest(text=spec.request_text)
        return self._finalize(spec, request)

    def _recover_done_segments(self, spec: SceneSpec, start: int) -> None:
        """断点续跑：已完成的段从磁盘恢复（重算 GT 确定性一致 + 读回帧序列）。"""
        import glob
        for seg in spec.segments[:start]:
            sim = self.simulator.simulate_segment(seg.index)
            seg_dir = os.path.join(self.run_dir, f"seg_{seg.index:02d}")
            candidates = sorted(
                glob.glob(os.path.join(seg_dir, "**", "frame_0000.png"), recursive=True))
            if candidates:
                frame0 = candidates[-1]
                prefix = os.path.dirname(frame0)
                frames = sorted(glob.glob(os.path.join(prefix, "frame_*.png")))
            else:
                frames = []
            self._last[seg.index] = (
                sim,
                VideoClip(segment_index=seg.index, frame_paths=frames, fps=sim.fps),
                AuditReport(segment_index=seg.index, checks=[], anomalies=[]),
            )

    # ---------- 段循环 ----------

    def _run_segments(self, spec: SceneSpec, start: int) -> None:
        for seg in spec.segments[start:]:
            t0 = time.time()
            sim = self.simulator.simulate_segment(seg.index)
            clip = self.generator.generate(
                sim.first_frame_path, sim, seg.prompt, sim.n_frames,
                out_dir=os.path.join(self.run_dir, f"seg_{seg.index:02d}"))
            print(f"[seg {seg.index}] 模拟+生成完成 ({time.time()-t0:.1f}s)")

            report = self.auditor.audit(clip, sim)
            rounds = 0
            while not report.ok and rounds < self.cfg.budget.repair_rounds_per_segment:
                rounds += 1
                print(f"[seg {seg.index}] 发现 {len(report.anomalies)} 处异常，"
                      f"第 {rounds} 轮段级修复...")
                if hasattr(self.repairer, "segment_prompt"):
                    self.repairer.segment_prompt = seg.prompt
                if hasattr(self.repairer, "set_anomalies"):
                    self.repairer.set_anomalies(report.anomalies)
                clip = self.repairer.repair_segment(
                    clip, sim, report,
                    os.path.join(self.run_dir, f"seg_{seg.index:02d}", f"repair_{rounds}"))
                report = self.auditor.audit(clip, sim, fix_factor=0.3 ** rounds)
                self.repair_log.append(
                    f"[seg {seg.index}] 段级修复第{rounds}轮，剩余异常 "
                    f"{len(report.anomalies)}")
            self._fix_state[seg.index] = 0.3 ** rounds
            if report.anomalies:
                self.repair_log.append(
                    f"[seg {seg.index}] 达到修复预算，{len(report.anomalies)} 处异常"
                    f"降级接受（将在全局阶段逐帧复核）")
            self.ckpt.save(spec, self._config_data(), seg.index + 1, "segments")
            self._last[seg.index] = (sim, clip, report)
        print("[pipeline] 全部段生成完成")

    # ---------- 全局逐帧审查 + 直接修复 ----------

    def _global_audit_and_repair(self, spec: SceneSpec) -> None:
        # 演示异常注入只属于段生成阶段；全局审查一律关闭
        if hasattr(self.auditor, "inject_anomaly"):
            self.auditor.inject_anomaly = False
        print("[global] 全局逐帧审查开始")
        frame_checks: list = []
        global_anomalies: list = []
        for seg in spec.segments:
            sim, clip, seg_report = self._last[seg.index]
            # 全局复审：延续该段当前修复质量状态（真实后端即直接重审视频）
            report = self.auditor.audit(
                clip, sim, fix_factor=self._fix_state.get(seg.index, 1.0))
            for c in report.checks:
                frame_checks.append(c)
            for a in report.anomalies:
                global_anomalies.append(a)

            bad_frames = sorted(report.anomalous_frames())
            if not bad_frames:
                continue
            ratio = len(bad_frames) / max(len(clip.frame_paths), 1)
            if ratio > self.cfg.budget.frame_anomaly_seg_ratio:
                print(f"[global] seg {seg.index} 异常帧占比 {ratio:.0%} → 整段重生成")
                if hasattr(self.repairer, "segment_prompt"):
                    self.repairer.segment_prompt = seg.prompt
                if hasattr(self.repairer, "set_anomalies"):
                    self.repairer.set_anomalies(report.anomalies)
                clip = self.repairer.repair_segment(
                    clip, sim, report,
                    os.path.join(self.run_dir, f"seg_{seg.index:02d}", "global_repair"))
                report = self.auditor.audit(clip, sim, fix_factor=0.1)
                global_anomalies.extend(report.anomalies)
                self.repair_log.append(
                    f"[seg {seg.index}] 全局阶段整段重生成，剩余异常 "
                    f"{len(report.anomalies)}")
            else:
                round_n = 0
                while bad_frames and round_n < self.cfg.budget.max_frame_repair_rounds:
                    round_n += 1
                    print(f"[global] seg {seg.index} 逐帧修复 {len(bad_frames)} 帧"
                          f"（第 {round_n} 轮）")
                    if hasattr(self.repairer, "set_anomalies"):
                        self.repairer.set_anomalies(report.anomalies)
                    clip = self.repairer.repair_frames(
                        clip, sim, bad_frames,
                        os.path.join(self.run_dir, f"seg_{seg.index:02d}",
                                     f"global_frame_fix_{round_n}"))
                    report = self.auditor.audit(clip, sim, fix_factor=0.15 ** round_n)
                    bad_frames = sorted(report.anomalous_frames())
                    self.repair_log.append(
                        f"[seg {seg.index}] 全局帧级修复第{round_n}轮，剩余异常帧 "
                        f"{len(bad_frames)}")
                if bad_frames:
                    self.repair_log.append(
                        f"[seg {seg.index}] {len(bad_frames)} 帧修复预算耗尽，保留标注")
            self._last[seg.index] = (sim, clip, report)
        self.global_report = AuditReport(segment_index=-1,
                                         checks=frame_checks,
                                         anomalies=global_anomalies)
        self.clips = {i: v[1] for i, v in self._last.items()}
        print("[global] 全局审查/修复完成")

    # ---------- 收尾 ----------

    def _finalize(self, spec: SceneSpec, request: UserRequest) -> str:
        final_dir = os.path.join(self.run_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        clips_info = []
        for idx, clip in sorted(self.clips.items()):
            video_path = os.path.join(final_dir, f"final_seg_{idx:02d}.mp4")
            ok = frames_to_video(clip.frame_paths, clip.fps, video_path)
            gif_path = os.path.join(final_dir, f"final_seg_{idx:02d}.gif")
            ok_gif = frames_to_gif(clip.frame_paths, clip.fps, gif_path)
            clips_info.append({"segment": idx, "frames": len(clip.frame_paths),
                               "video": video_path if ok else None,
                               "gif": gif_path if ok_gif else None})

        report_path = render_report(
            self.run_dir, request.text,
            [v[2] for v in self._last.values()],
            getattr(self, "global_report", None),
            self.repair_log, clips_info)
        self.ckpt.save(spec, self._config_data(),
                       len(spec.segments), "done")
        print(f"[done] 审查报告: {report_path}")
        return report_path

    # ---------- utils ----------

    def _config_data(self) -> dict:
        from dataclasses import asdict
        cfg = self.cfg
        d = {
            "planner": asdict(cfg.planner), "simulator": asdict(cfg.simulator),
            "generator": asdict(cfg.generator), "auditor": asdict(cfg.auditor),
            "repairer": asdict(cfg.repairer), "thresholds": asdict(cfg.thresholds),
            "budget": asdict(cfg.budget), "video": {"fps": cfg.video.fps},
            "mock": asdict(cfg.mock), "audit": asdict(cfg.audit),
        }
        # 快照脱敏：密钥不落盘（resume 时从环境变量还原）
        for m in ("planner", "simulator", "generator", "auditor", "repairer"):
            d[m]["api_key"] = None
        return d

    def _spec_from_state(self, state: dict) -> SceneSpec:
        from ..types import (GroundSpec, SceneEvent, SceneObject, SegmentSpec)
        s = state["spec"]
        return SceneSpec(
            request_text=s["request_text"],
            ground=GroundSpec(**s["ground"]),
            objects=[SceneObject(**o) for o in s["objects"]],
            events=[SceneEvent(**{**e, "actors": e["actors"], "params": e["params"]})
                    for e in s["events"]],
            segments=[SegmentSpec(**g) for g in s["segments"]],
            style=s.get("style", {}),
        )

    def _last_reset(self) -> None:
        self._last = {}
        self.repair_log = []
