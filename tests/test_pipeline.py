import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from physgen.config import load_config
from physgen.orchestrator import Pipeline
from physgen.types import UserRequest


def _fresh_pipeline(tmp: str, **mock_overrides) -> Pipeline:
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    # 单测一律走 Mock 后端（config.yaml 默认指向真实后端）
    for m in ("planner", "simulator", "generator", "auditor", "repairer"):
        getattr(cfg, m).provider = "mock"
    cfg.video.fps = 12          # 测试提速
    cfg.video.segment_seconds = 2.0
    cfg.video.resolution = [320, 240]
    for k, v in mock_overrides.items():
        setattr(cfg.mock, k, v)
    return Pipeline(cfg, os.path.join(tmp, "run1"))


def test_end_to_end_mock(tmp_path=None):
    tmp = tempfile.mkdtemp(prefix="physgen_e2e_")
    try:
        p = _fresh_pipeline(tmp, inject_anomaly=True)
        report_path = p.run(UserRequest(text="一块10×10米的土地上盖一间8×6×3米的房子"))
        assert os.path.exists(report_path)
        # 各段产物与 checkpoint
        assert os.path.exists(os.path.join(p.run_dir, "state.json"))
        segs = [d for d in os.listdir(p.run_dir) if d.startswith("seg_")]
        assert len(segs) >= 2
        # 全局阶段产物
        assert os.path.exists(os.path.join(p.run_dir, "final"))
        # 有修复记录（注入了演示异常）
        assert any("修复" in x for x in p.repair_log)
        html = open(report_path, encoding="utf-8").read()
        assert "PhysGen" in html and "审查报告" in html
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_checkpoint_resume(tmp_path=None):
    tmp = tempfile.mkdtemp(prefix="physgen_resume_")
    try:
        p = _fresh_pipeline(tmp, inject_anomaly=False)
        cfg_data = p._config_data()
        from physgen.planner.mock import MockPlanner
        from physgen.types import UserRequest
        spec = MockPlanner().plan(
            UserRequest(text="一块10×10米的土地上盖一间8×6×3米的房子", duration_s=6.0))
        # 手动存一个"完成段0"的 checkpoint，模拟中断
        p.ckpt.save(spec, cfg_data, 1, "segments")
        report_path = p.resume(p.run_dir)
        assert os.path.exists(report_path)
        assert len(p._last) >= len(spec.segments)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
