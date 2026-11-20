"""audit 模式（外部视频质检）测试：双通道审查 + 优先级修复 + 开关。"""

import os

import pytest
from PIL import Image

from physgen.config import (AppConfig, AuditCfg, Budget, MockCfg, ProviderCfg,
                            Thresholds, VideoCfg)
from physgen.types import (Anomaly, AuditReport, CheckItem, GroundSpec,
                           SceneEvent, SceneObject, SceneSpec, SegmentSpec,
                           UserRequest)
from physgen.auditor.base import PhysicsAuditor
from physgen.repairer.base import Repairer
from physgen.orchestrator.audit_only import AuditPipeline


# ---------- 假模块：确定性驱动双通道/优先级行为 ----------

class FakePlanner:
    def plan(self, request: UserRequest):
        return SceneSpec(
            request_text=request.text,
            ground=GroundSpec(10, 10),
            objects=[SceneObject("stone", "石块", (2, 2, 1), 10400.0,
                                 "stone", (1, 0.5, 1))],
            events=[SceneEvent(0, 2, "fall", ["stone"], {"drop_height_m": 2})],
            segments=[SegmentSpec(0, 0, 4, request.text)],
        )


class FakeAuditor(PhysicsAuditor):
    """脚本化双通道（按调用次序）：
    lane1: 第 1 次调用报帧 1，之后通过
    lane2: 第 1 次（初始巡检）通过；第 2 次（首轮修复后复审）报帧 3；之后通过
    """

    def __init__(self):
        self.lane1_calls = 0
        self.lane2_calls = 0
        self.background_enabled = True

    @staticmethod
    def _report(metric, frame):
        if frame is None:
            return AuditReport(0, [], [])
        return AuditReport(0, [CheckItem("scene", metric, None, 1, 1, 0, True)],
                           [Anomaly(0, frame, "stone", metric, "high",
                                    f"{metric} 异常", "修复它")])

    def audit_expectations(self, clip, spec):
        self.lane1_calls += 1
        return self._report("acceptance_check", 1 if self.lane1_calls == 1 else None)

    def audit_background(self, clip, spec):
        self.lane2_calls += 1
        if not self.background_enabled:
            raise AssertionError("background_check=False 时不应调用通道②")
        return self._report("physics_patrol", 3 if self.lane2_calls == 2 else None)


class SpyRepairer(Repairer):
    """记录每次修复的帧号顺序，帧内容原样复用。"""

    def __init__(self):
        self.calls = []

    def repair_frames(self, clip, sim, frame_indices, out_dir):
        self.calls.append(list(frame_indices))
        return clip

    def repair_segment(self, clip, sim, report, out_dir):
        return clip


# ---------- fixtures ----------

def _make_frames(tmp_path, n=5):
    d = tmp_path / "frames"
    d.mkdir()
    for i in range(n):
        Image.new("RGB", (320, 240), (i * 30, 100, 80)).save(d / f"frame_{i:04d}.png")
    return str(d)


def _cfg(background_check=True):
    return AppConfig(
        planner=ProviderCfg(provider="mock"),
        simulator=ProviderCfg(provider="mock"),
        generator=ProviderCfg(provider="mock"),
        auditor=ProviderCfg(provider="mock"),
        repairer=ProviderCfg(provider="mock"),
        thresholds=Thresholds(), budget=Budget(max_frame_repair_rounds=3),
        video=VideoCfg(fps=6, segment_seconds=1.0, resolution=[320, 240]),
        mock=MockCfg(), audit=AuditCfg(background_check=background_check))


# ---------- 测试 ----------

def test_dual_lane_priority_repair(tmp_path):
    """通道① 异常先修，通道② 后修，预算内收敛；报告落盘。"""
    frames_dir = _make_frames(tmp_path)
    run_dir = str(tmp_path / "run")
    ap = AuditPipeline(_cfg(background_check=True), run_dir)
    ap.planner = FakePlanner()
    ap.auditor = FakeAuditor()
    repairer = SpyRepairer()
    ap.repairer = repairer

    report = ap.run(frames_dir, "一块10×10米的土地上放一个2×2×1米的石块")

    # 优先级：第一次修通道① 的帧 1，第二次修通道② 的帧 3
    assert repairer.calls == [[1], [3]]
    # 审查次数：初始 1 次 + 每轮修复后复审（2 轮修复 → 共 3 次）
    assert ap.auditor.lane1_calls == 3 and ap.auditor.lane2_calls == 3
    assert os.path.exists(report) and os.path.exists(
        os.path.join(run_dir, "report.html"))
    # 最终产物
    assert os.path.exists(os.path.join(run_dir, "final"))


def test_background_check_off(tmp_path):
    """background_check=False：通道② 不被调用，只修通道①。"""
    frames_dir = _make_frames(tmp_path)
    run_dir = str(tmp_path / "run")
    ap = AuditPipeline(_cfg(background_check=False), run_dir)
    ap.planner = FakePlanner()
    ap.auditor = FakeAuditor()
    repairer = SpyRepairer()
    ap.repairer = repairer

    ap.run(frames_dir, "测试请求")

    assert ap.auditor.lane2_calls == 0      # 通道② 未启用
    assert repairer.calls == [[1]]          # 只修通道① 的帧
    assert not any("②" in line for line in [])  # 无通道② 修复记录


def test_missing_video_raises(tmp_path):
    ap = AuditPipeline(_cfg(), str(tmp_path / "run"))
    with pytest.raises(FileNotFoundError):
        ap.run(str(tmp_path / "nope.mp4"), "测试")
