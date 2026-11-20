"""audit 模式（外部视频质检）测试：双通道审查 + 优先级修复 + 开关。"""

import os

import pytest
from PIL import Image

from physgen.config import (AppConfig, AuditCfg, Budget, MockCfg, ProviderCfg,
                            Thresholds, VideoCfg)
from physgen.types import (Anomaly, AuditReport, CheckItem, GroundSpec,
                           ObjectFrameState, SceneEvent, SceneObject,
                           SceneSpec, SegmentSpec, SimResult, UserRequest)
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
    strict 模式下 lane1 改走 audit()（GT 基），计数在 gt_calls。
    """

    def __init__(self):
        self.lane1_calls = 0
        self.lane2_calls = 0
        self.gt_calls = 0
        self.background_enabled = True

    def audit(self, clip, sim, fix_factor=1.0):
        """GT 基审查（strict 模式的通道①）。"""
        self.gt_calls += 1
        assert sim.gt, "strict 模式下 sim 必须带物理真值"
        return self._report("gt_expectations", 1 if self.gt_calls == 1 else None)

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


class FakeSimulator:
    """记录 build/simulate 调用，返回带 GT 的 SimResult（帧目录输入）。"""

    def __init__(self, first_frame_path: str):
        self.built = False
        self.ran = 0
        self.run_dir = None
        self._first = first_frame_path

    def build(self, spec):
        self.built = True

    def simulate_segment(self, index: int) -> SimResult:
        self.ran += 1
        return SimResult(
            segment_index=index, first_frame_path=self._first, fps=6,
            n_frames=5,
            gt={"stone": [ObjectFrameState("stone", (1, 0.5, 1), (0, 0, 0),
                                           (2, 2, 1))] * 5})


def test_strict_uses_local_simulation_gt(tmp_path):
    """strict=True：先本地模拟出真值，通道①走 GT 基审查；sim 传给修复器。"""
    frames_dir = _make_frames(tmp_path)
    run_dir = str(tmp_path / "run")
    ap = AuditPipeline(_cfg(background_check=True), run_dir)
    ap.planner = FakePlanner()
    ap.auditor = auditor = FakeAuditor()
    repairer = SpyRepairer()
    ap.repairer = repairer

    import physgen.orchestrator.audit_only as ao
    fake_sim = FakeSimulator(f"{frames_dir}/frame_0000.png")
    monkey = __import__("pytest").MonkeyPatch()
    monkey.setattr(ao, "create_simulator", lambda cfg: fake_sim)
    try:
        ap.run(frames_dir, "一块10×10米的土地上放一个2×2×1米的石块", strict=True)
    finally:
        monkey.undo()

    assert fake_sim.built and fake_sim.ran == 1        # 模拟只跑一次
    assert auditor.gt_calls == 3 and auditor.lane1_calls == 0  # 通道①走 GT 基
    # 通道①(GT) 报帧 1 先修；通道② 巡检报帧 3 后修（背景巡检仍开启）
    assert repairer.calls == [[1], [3]]


def test_non_strict_skips_simulation(tmp_path):
    """默认模式：完全不触碰模拟器，通道①走期望清单审查。"""
    frames_dir = _make_frames(tmp_path)
    run_dir = str(tmp_path / "run")
    ap = AuditPipeline(_cfg(background_check=False), run_dir)
    ap.planner = FakePlanner()
    ap.auditor = auditor = FakeAuditor()
    ap.repairer = SpyRepairer()

    ap.run(frames_dir, "测试请求", strict=False)

    assert auditor.gt_calls == 0          # 没走 audit()（GT 基）
    assert auditor.lane1_calls == 2       # 走的是期望清单审查
    assert auditor.lane2_calls == 0


def test_missing_video_raises(tmp_path):
    ap = AuditPipeline(_cfg(), str(tmp_path / "run"))
    with pytest.raises(FileNotFoundError):
        ap.run(str(tmp_path / "nope.mp4"), "测试")
