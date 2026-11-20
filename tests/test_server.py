"""HTTP 服务化包装测试（全 Mock 流水线，不打网络）。"""

import json
import time

import pytest

fastapi = pytest.importorskip("fastapi", reason="未安装 server extra，跳过")
from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402
from server import app  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # 隔离 runs 目录 + 单 worker + 精简 Mock 配置
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(server, "WORKERS", 1)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "planner: {provider: mock}\n"
        "simulator: {provider: mock}\n"
        "generator: {provider: mock}\n"
        "auditor: {provider: mock}\n"
        "repairer: {provider: mock}\n"
        "video: {fps: 6, segment_seconds: 0.5, resolution: [320, 240]}\n")
    with TestClient(app) as c:
        c.cfg_path = str(cfg)
        yield c


def _wait_done(client, task_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = client.get(f"/tasks/{task_id}").json()
        if s["status"] in ("done", "failed"):
            return s
        time.sleep(1)
    raise AssertionError("任务超时未完成")


def test_submit_run_and_result(client):
    r = client.post("/tasks", json={"prompt": "一块4×4米的场地上放一个1×1×1米的石块",
                                    "config": client.cfg_path})
    assert r.status_code == 202
    task_id = r.json()["task_id"]
    s = _wait_done(client, task_id)
    assert s["status"] == "done", s.get("error")
    assert s["progress"]["segments_total"] >= 1

    res = client.get(f"/tasks/{task_id}/result").json()
    assert res["report"] and res["report_url"]
    # Mock 生成器只有帧序列（无 ffmpeg 编码的 clip.mp4），验证产物文件可下载
    assert client.get(f"/tasks/{task_id}/files",
                      params={"path": "state.json"}).status_code == 200

    # 报告可访问 + 文件下载带防穿越
    rep = client.get(f"/tasks/{task_id}/report")
    assert rep.status_code == 200 and "html" in rep.headers["content-type"]
    assert client.get(f"/tasks/{task_id}/files",
                      params={"path": "../outside.txt"}).status_code == 404


def test_auth_required_when_token_set(client, monkeypatch):
    monkeypatch.setattr(server, "TOKEN", "secret")
    assert client.post("/tasks", json={"prompt": "x"}).status_code == 401
    ok = client.post("/tasks", json={"prompt": "x", "config": client.cfg_path},
                     headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 202


def test_audit_endpoint_with_frames_dir(client, tmp_path):
    """POST /audit：帧目录输入 → audit 任务 → 报告产出。"""
    from PIL import Image as PILImage
    d = tmp_path / "frames"
    d.mkdir()
    for i in range(4):
        PILImage.new("RGB", (320, 240), (50, 100, 150)).save(d / f"frame_{i:04d}.png")
    r = client.post("/audit", json={"video": str(d),
                                    "prompt": "一块4×4米的场地上放一个1×1×1米的石块",
                                    "config": client.cfg_path})
    assert r.status_code == 202
    task_id = r.json()["task_id"]
    assert r.json()["type"] == "audit"
    s = _wait_done(client, task_id)
    assert s["status"] == "done", s.get("error")
    res = client.get(f"/tasks/{task_id}/result").json()
    assert res["report_url"]


def test_empty_prompt_rejected(client):
    assert client.post("/tasks", json={"prompt": "  ",
                                       "config": client.cfg_path}).status_code == 400


def test_persisted_task_recovered(client):
    """服务重启后：完成的任务仍在列表；中断任务标记 interrupted 可续跑。"""
    r = client.post("/tasks", json={"prompt": "测试", "config": client.cfg_path})
    task_id = r.json()["task_id"]
    _wait_done(client, task_id)
    # 模拟重启：重载持久化任务
    server._tasks.clear()
    server._load_persisted_tasks()
    assert task_id in server._tasks
    assert server._tasks[task_id]["status"] == "done"
    # 中断任务被标记
    meta = json.loads((server.RUNS_DIR / task_id / "task_meta.json")
                      .read_text(encoding="utf-8"))
    meta["status"] = "running"
    (server.RUNS_DIR / task_id / "task_meta.json").write_text(
        json.dumps(meta), encoding="utf-8")
    server._tasks.clear()
    server._load_persisted_tasks()
    assert server._tasks[task_id]["status"] == "interrupted"
    # 断点续跑接口可用（Mock 流水线，秒级完成）
    s = client.post(f"/tasks/{task_id}/resume")
    assert s.status_code == 202
    _wait_done(client, task_id)
