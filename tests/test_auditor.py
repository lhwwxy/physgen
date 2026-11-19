import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from physgen.auditor.mock import MockAuditor
from physgen.config import Thresholds
from physgen.types import (GroundSpec, SceneEvent, SceneObject, SceneSpec,
                           SegmentSpec, UserRequest, mass_from_volume)


def _make(spec_text="一块10×10米的土地上盖一间8×6×3米的房子", seg_index=0):
    from physgen.planner.mock import MockPlanner
    from physgen.simulator.mock import MockSimulator

    spec: SceneSpec = MockPlanner().plan(UserRequest(text=spec_text, duration_s=6.0))
    sim = MockSimulator(fps=24, resolution=(320, 240))
    sim.run_dir = os.path.join(os.path.dirname(__file__), "_tmp_render")
    sim.build(spec)
    sim_result = sim.simulate_segment(seg_index)
    clip_frames = [sim_result.first_frame_path] * sim_result.n_frames
    from physgen.types import VideoClip
    clip = VideoClip(segment_index=seg_index, frame_paths=clip_frames,
                     fps=sim_result.fps)
    return sim, sim_result, clip


def test_audit_clean_when_no_injection():
    sim, result, clip = _make(seg_index=0)
    auditor = MockAuditor(noise_seed=1, inject_anomaly=False)
    report = auditor.audit(clip, result)
    assert report.ok, f"低噪声下不应有异常: {report.anomalies[:3]}"


def test_audit_detects_injected_anomaly():
    """注入超标噪声 → 审查器必须检出（演示异常被抓住）。"""
    sim, result, clip = _make(seg_index=1)
    auditor = MockAuditor(noise_seed=1, inject_anomaly=True)
    report = auditor.audit(clip, result)
    assert not report.ok
    assert report.anomalous_frames()


def test_repair_loop_reduces_anomalies():
    """修复轮次提高 fix_factor 衰减 → 异常收敛。"""
    sim, result, clip = _make(seg_index=1)
    auditor = MockAuditor(noise_seed=1, inject_anomaly=True)
    r1 = auditor.audit(clip, result, fix_factor=1.0)
    r2 = auditor.audit(clip, result, fix_factor=0.3)
    assert len(r2.anomalies) <= len(r1.anomalies)


def test_fall_event_present():
    from physgen.planner.mock import MockPlanner
    spec = MockPlanner().plan(UserRequest(text="10×10米场地放一个2×2×1米的石头"))
    assert any(e.type == "fall" for e in spec.events)
