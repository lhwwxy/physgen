"""M4 修复器单元测试（不打网络，全部 mock）。"""

import base64
import io
import os

import pytest
from PIL import Image

from physgen.repairer.seedance import SeedanceRepairer
from physgen.types import Anomaly, SimResult, VideoClip


def _png(path, size=(320, 240), color=(10, 20, 30)):
    Image.new("RGB", size, color).save(path)
    return path


def test_clamp_size():
    # 320×240 < 921600 → 放大到合法像素数且保持宽高比
    w, h = map(int, SeedanceRepairer._clamp_size(320, 240).split("x"))
    assert w * h >= 921600
    assert abs(w / h - 320 / 240) < 0.05
    # 1280×720 恰好为下限，原样返回
    assert SeedanceRepairer._clamp_size(1280, 720) == "1280x720"
    # 超大 → 缩小到上限内
    w, h = map(int, SeedanceRepairer._clamp_size(4000, 3000).split("x"))
    assert w * h <= 4624220


def test_match_original_crops_and_resizes(tmp_path):
    src = _png(str(tmp_path / "src.png"), size=(320, 240))
    edited = Image.new("RGB", (1024, 1024), (200, 100, 50))
    out = SeedanceRepairer._match_original(edited, src)
    assert out.size == (320, 240)  # 方形编辑图裁成 4:3 后缩放


def test_repair_frames_replaces_only_targets(tmp_path, monkeypatch):
    paths = [_png(str(tmp_path / f"frame_{i:04d}.png")) for i in range(5)]
    rep = SeedanceRepairer(api_key="fake", model="test-video",
                           image_model="test-image",
                           base_url="https://example.com/v1")

    # 伪造 Seedream 返回：一张纯色图
    buf = io.BytesIO()
    Image.new("RGB", (1024, 1024), (99, 99, 99)).save(buf, format="PNG")
    fake_b64 = base64.b64encode(buf.getvalue()).decode()

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"b64_json": fake_b64}]}

    monkeypatch.setattr("physgen.repairer.seedance.requests.post",
                        lambda *a, **kw: _Resp())
    rep.set_anomalies([Anomaly(segment_index=0, frame_index=2, obj_id="stone_2",
                               metric="vlm_commonsense", severity="high",
                               detail="石块过大", suggest_fix="缩小石块")])
    clip = VideoClip(segment_index=0, frame_paths=paths, fps=12)
    sim = SimResult(segment_index=0, first_frame_path=paths[0], fps=12,
                    n_frames=5, gt={})
    out_dir = str(tmp_path / "fixed")
    new_clip = rep.repair_frames(clip, sim, [2, 4], out_dir)
    assert new_clip.frame_paths[0] == paths[0]
    assert new_clip.frame_paths[2] != paths[2]
    assert new_clip.frame_paths[4] != paths[4]
    assert os.path.exists(new_clip.frame_paths[2])
    assert Image.open(new_clip.frame_paths[2]).size == (320, 240)


def test_repair_frames_caps_per_round(tmp_path, monkeypatch):
    paths = [_png(str(tmp_path / f"frame_{i:04d}.png")) for i in range(10)]
    rep = SeedanceRepairer(api_key="fake", model="test-video",
                           image_model="test-image",
                           base_url="https://example.com/v1")
    calls = []

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            buf = io.BytesIO()
            Image.new("RGB", (1024, 1024), (1, 2, 3)).save(buf, format="PNG")
            return {"data": [{"b64_json": base64.b64encode(buf.getvalue()).decode()}]}

    def _post(*a, **kw):
        calls.append(1)
        return _Resp()

    monkeypatch.setattr("physgen.repairer.seedance.requests.post", _post)
    rep.set_anomalies([])
    clip = VideoClip(segment_index=0, frame_paths=paths, fps=12)
    sim = SimResult(segment_index=0, first_frame_path=paths[0], fps=12,
                    n_frames=10, gt={})
    rep.repair_frames(clip, sim, list(range(9)), str(tmp_path / "fixed2"))
    assert len(calls) == rep.max_frames_per_round  # 每轮预算内封顶
