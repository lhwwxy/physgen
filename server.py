from __future__ import annotations

"""PhysGen HTTP 服务化包装（供视频制作平台等外部系统集成）。

核心流水线零改动，本模块只提供：
- POST /tasks            提交生成任务（异步，立即返回 task_id）
- GET  /tasks            任务列表
- GET  /tasks/{id}       任务状态与进度（来自 checkpoint：phase/段进度）
- POST /tasks/{id}/resume 断点续跑失败/中断的任务
- GET  /tasks/{id}/result  产物清单（成片/报告/各段视频）
- GET  /tasks/{id}/report  审查报告 HTML
- GET  /tasks/{id}/files?path=...  下载任意产物文件（防路径穿越）

鉴权：设置环境变量 PHYSGEN_SERVER_TOKEN 后，所有接口要求
`Authorization: Bearer <token>`；未设置则不鉴权（内网/本地用）。

运行：uvicorn server:app --host 0.0.0.0 --port 8000
或：  physgen-server

注意：输入内容的平台侧安全过滤由集成方负责，本服务不审核 prompt。
"""

import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
RUNS_DIR = Path(os.environ.get("PHYSGEN_RUNS_DIR", ROOT / "runs")).resolve()
WORKERS = int(os.environ.get("PHYSGEN_SERVER_WORKERS", "2"))
TOKEN = os.environ.get("PHYSGEN_SERVER_TOKEN", "")  # 空则不鉴权

app = FastAPI(title="PhysGen API",
              description="物理一致性视频生成流水线（规划→模拟→生成→审查→修复）",
              version="0.1.0")

_pool = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="physgen")
_tasks: dict[str, dict] = {}
_lock = threading.Lock()


# ---------- 任务元数据 ----------

def _save_meta(t: dict) -> None:
    meta = {k: v for k, v in t.items() if not k.startswith("_")}
    try:
        with open(Path(t["run_dir"]) / "task_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
    except Exception:  # noqa: BLE001 — 持久化失败不影响任务执行
        pass


def _load_persisted_tasks() -> None:
    """服务重启后从 runs/ 恢复任务列表；进行中的标记为 interrupted。"""
    for meta in sorted(RUNS_DIR.glob("*/task_meta.json")):
        try:
            t = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — 坏文件跳过
            continue
        if t.get("status") in ("queued", "running"):
            t["status"] = "interrupted"
            t["error"] = "服务重启导致中断，可调用 resume 续跑"
        _tasks[t["id"]] = t


_load_persisted_tasks()


def _public(t: dict) -> dict:
    return {k: v for k, v in t.items() if k != "_future"}


# ---------- 执行 ----------

def _progress_of(run_dir: str) -> dict:
    """从 checkpoint 读进度（流水线零改动）。"""
    state_path = Path(run_dir) / "state.json"
    if not state_path.exists():
        return {"phase": "queued", "segments_done": 0, "segments_total": 0}
    try:
        s = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"phase": "unknown", "segments_done": 0, "segments_total": 0}
    phase = s.get("phase", "planning")
    total = len((s.get("spec") or {}).get("segments") or [])
    nxt = s.get("next_segment", 0)
    done = total if phase in ("global_audit", "done") else min(nxt, total)
    return {"phase": phase, "segments_done": done, "segments_total": total}


def _execute(task_id: str, resume: bool) -> None:
    with _lock:
        t = _tasks[task_id]
    try:
        from physgen.config import load_config
        from physgen.orchestrator import Pipeline
        from physgen.types import UserRequest

        cfg = load_config(t["config"])
        pipeline = Pipeline(cfg, t["run_dir"])
        if resume:
            pipeline.resume(t["run_dir"])
        else:
            pipeline.run(UserRequest(text=t["prompt"],
                                     duration_s=t.get("duration_s")))
        t["status"] = "done"
    except Exception as e:  # noqa: BLE001 — 失败落到任务状态，不炸服务
        t["status"] = "failed"
        t["error"] = f"{type(e).__name__}: {e}"[:500]
    finally:
        t["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _save_meta(t)


def _enqueue(task_id: str, resume: bool = False) -> None:
    fut = _pool.submit(_execute, task_id, resume)
    with _lock:
        _tasks[task_id]["_future"] = fut
        _tasks[task_id]["status"] = "running"


# ---------- 鉴权 ----------

def _auth(authorization: str | None = Header(default=None)) -> None:
    if TOKEN and authorization != f"Bearer {TOKEN}":
        raise HTTPException(401, "缺少或错误的鉴权 token（Header: "
                                 "Authorization: Bearer <PHYSGEN_SERVER_TOKEN>）")


# ---------- 模型 ----------

class TaskIn(BaseModel):
    prompt: str
    config: str | None = None       # config.yaml 路径，缺省用仓库默认
    duration_s: float | None = None  # 视频总时长（秒），缺省用配置值


# ---------- 接口 ----------

@app.post("/tasks", status_code=202)
def submit(task: TaskIn, authorization: str | None = Header(default=None)) -> dict:
    _auth(authorization)
    if not task.prompt.strip():
        raise HTTPException(400, "prompt 不能为空")
    task_id = time.strftime("t%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    run_dir = str(RUNS_DIR / task_id)
    os.makedirs(run_dir, exist_ok=True)
    t = {"id": task_id, "status": "queued", "prompt": task.prompt,
         "config": task.config, "duration_s": task.duration_s,
         "run_dir": run_dir, "error": None,
         "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
         "finished_at": None}
    with _lock:
        _tasks[task_id] = t
    _save_meta(t)
    _enqueue(task_id)
    return {"task_id": task_id, "status": "running",
            "detail": f"GET /tasks/{task_id} 查询进度"}


@app.post("/tasks/{task_id}/resume", status_code=202)
def resume(task_id: str, authorization: str | None = Header(default=None)) -> dict:
    _auth(authorization)
    t = _tasks.get(task_id)
    if not t:
        raise HTTPException(404, "任务不存在")
    if not (Path(t["run_dir"]) / "state.json").exists():
        raise HTTPException(409, "该任务没有 checkpoint，无法续跑")
    if t["status"] == "running":
        raise HTTPException(409, "任务正在运行中")
    t["error"] = None
    _enqueue(task_id, resume=True)
    return {"task_id": task_id, "status": "running"}


@app.get("/tasks")
def list_tasks(authorization: str | None = Header(default=None)) -> list[dict]:
    _auth(authorization)
    return [_public(t) | {"progress": _progress_of(t["run_dir"])}
            for t in _tasks.values()]


@app.get("/tasks/{task_id}")
def task_status(task_id: str, authorization: str | None = Header(default=None)) -> dict:
    _auth(authorization)
    t = _tasks.get(task_id)
    if not t:
        raise HTTPException(404, "任务不存在")
    return _public(t) | {"progress": _progress_of(t["run_dir"])}


@app.get("/tasks/{task_id}/result")
def task_result(task_id: str, authorization: str | None = Header(default=None)) -> dict:
    _auth(authorization)
    t = _tasks.get(task_id)
    if not t:
        raise HTTPException(404, "任务不存在")
    rd = Path(t["run_dir"])
    clips = sorted(str(p) for p in rd.glob("seg_*/clip.mp4"))
    finals = sorted(str(p) for p in (rd / "final").glob("*")) if (rd / "final").exists() else []
    report = rd / "report.html"
    return {"status": t["status"],
            "report": str(report) if report.exists() else None,
            "report_url": f"/tasks/{task_id}/report" if report.exists() else None,
            "clips": clips, "final": finals}


@app.get("/tasks/{task_id}/report", response_class=HTMLResponse)
def task_report(task_id: str, authorization: str | None = Header(default=None)):
    _auth(authorization)
    t = _tasks.get(task_id)
    if not t:
        raise HTTPException(404, "任务不存在")
    report = Path(t["run_dir"]) / "report.html"
    if not report.exists():
        raise HTTPException(404, "报告尚未生成")
    return FileResponse(report, media_type="text/html")


@app.get("/tasks/{task_id}/files")
def task_file(task_id: str, path: str = Query(..., description="run 目录内相对路径"),
              authorization: str | None = Header(default=None)):
    _auth(authorization)
    t = _tasks.get(task_id)
    if not t:
        raise HTTPException(404, "任务不存在")
    root = Path(t["run_dir"]).resolve()
    target = (root / path).resolve()
    if not str(target).startswith(str(root) + os.sep) or not target.is_file():
        raise HTTPException(404, "文件不存在或路径越界")
    return FileResponse(target)


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PHYSGEN_SERVER_PORT", "8000")))


if __name__ == "__main__":
    main()
