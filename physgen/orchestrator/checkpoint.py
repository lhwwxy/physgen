from __future__ import annotations

"""Checkpoint：运行状态落盘，支持断点续跑。

run_dir 结构：
runs/<run_id>/
├── state.json          # 进度 + 配置 + SceneSpec
├── seg_00/ ...         # 每段产物（首帧/帧序列/clip.mp4）
├── final/              # 全局修复后的帧与成片
└── report.html         # 审查报告
"""

import json
import os
import time
from dataclasses import asdict, is_dataclass
from typing import Optional

from ..config import dump_config, from_pipeline_dict
from ..types import SceneSpec


def _to_jsonable(obj):
    if is_dataclass(obj):
        return {"__dc__": obj.__class__.__name__, **asdict(obj)}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


class Checkpoint:

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)

    @property
    def state_path(self) -> str:
        return os.path.join(self.run_dir, "state.json")

    def save(self, spec: SceneSpec, config_data: dict, next_segment: int,
             phase: str, extra: dict | None = None) -> None:
        state = {
            "run_id": os.path.basename(self.run_dir),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phase": phase,               # planning / segments / global_audit / done
            "next_segment": next_segment,
            "spec": _to_jsonable(spec),
            "config": config_data,
            "extra": extra or {},
        }
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        dump_config(config_data, os.path.join(self.run_dir, "config_snapshot.yaml"))

    def load(self) -> Optional[dict]:
        if not os.path.exists(self.state_path):
            return None
        with open(self.state_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def restore_config(self, state: dict):
        return from_pipeline_dict(state["config"])
