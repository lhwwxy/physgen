from __future__ import annotations

"""配置加载：config.yaml + .env + 环境变量。

密钥管理约定（仓库可公开）：
- config.yaml 中只写 `${ENV_NAME}` 占位引用，不含任何明文密钥
- 实际密钥放在 `.env`（被 .gitignore 排除）或 shell 环境变量
- 加载顺序：shell 环境变量 > .env（config 文件同目录 → CWD 向上查找）
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class Thresholds:
    dim_error_pct: float = 15.0
    trajectory_rmse_m: float = 0.30
    fall_time_err_pct: float = 20.0
    overlap_penetration: float = 0.05


@dataclass
class Budget:
    repair_rounds_per_segment: int = 2
    max_frame_repair_rounds: int = 3
    frame_anomaly_seg_ratio: float = 0.5
    max_frames_per_round: int = 4   # 全局阶段单轮最多修复的帧数


@dataclass
class VideoCfg:
    fps: int = 24
    segment_seconds: float = 3.0
    resolution: list = field(default_factory=lambda: [1280, 720])
    renderer: str = "3d"        # 3d（matplotlib 透视）| 2d（立面占位）


@dataclass
class MockCfg:
    noise_seed: int = 42
    inject_anomaly: bool = True


@dataclass
class ProviderCfg:
    provider: str = "mock"
    model: Optional[str] = None
    poll_interval_s: float = 2.0
    base_url: Optional[str] = None     # 中转网关 / 自定义端点（真实后端必填）
    api_key: Optional[str] = None      # 一般填 ${PHYSGEN_API_KEY} 引用，key 本体在 .env
    image_model: Optional[str] = None  # 修复器用：帧级重绘的图像编辑模型
    max_poll_time_s: Optional[int] = None  # 视频生成任务最长等待时间


@dataclass
class AuditCfg:
    """audit 模式（外部视频质检）设置。"""
    background_check: bool = True   # 通道②：画面物理一致性巡检（背景物体间冲突等）


@dataclass
class AppConfig:
    planner: ProviderCfg
    simulator: ProviderCfg
    generator: ProviderCfg
    auditor: ProviderCfg
    repairer: ProviderCfg
    thresholds: Thresholds
    budget: Budget
    video: VideoCfg
    mock: MockCfg
    audit: AuditCfg = field(default_factory=AuditCfg)

    def api_key(self, module: str = "") -> Optional[str]:
        """统一密钥入口：config.yaml 之外，key 只认 PHYSGEN_API_KEY（.env / 环境变量）。

        module 参数仅为兼容保留，不再区分模块专属变量。
        """
        return os.environ.get("PHYSGEN_API_KEY")


def _load_dotenv(search_dirs: list[Path]) -> None:
    """加载 .env 文件中尚未在环境里设置的变量（不覆盖已存在的）。"""
    for d in search_dirs:
        f = d / ".env" if d.is_dir() else d
        if not f.exists():
            continue
        for raw in f.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


def _interp(v: Any) -> Any:
    """递归展开字符串中的 ${VAR} 引用；未定义的变量保持原样。"""
    if isinstance(v, str):
        return re.sub(r"\$\{(\w+)\}",
                      lambda m: os.environ.get(m.group(1), m.group(0)), v)
    if isinstance(v, dict):
        return {k: _interp(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_interp(x) for x in v]
    return v


def _provider(d: dict) -> ProviderCfg:
    return ProviderCfg(
        provider=d.get("provider", "mock"),
        model=d.get("model"),
        poll_interval_s=float(d.get("poll_interval_s", 2.0)),
        base_url=d.get("base_url"),
        api_key=d.get("api_key"),
        image_model=d.get("image_model"),
        max_poll_time_s=(int(d["max_poll_time_s"])
                         if d.get("max_poll_time_s") is not None else None),
    )


def load_config(path: Optional[str | Path] = None) -> AppConfig:
    p = Path(path) if path else Path(__file__).resolve().parent.parent / "config.yaml"
    data: dict = {}
    if p.exists():
        # .env 优先级低于 shell 环境变量：从 config 文件目录与 CWD 查找
        _load_dotenv([p.parent, p.parent.parent, Path.cwd()])
        with open(p, "r", encoding="utf-8") as f:
            data = _interp(yaml.safe_load(f) or {})

    def section(name: str) -> dict:
        return data.get(name) or {}

    t = section("thresholds")
    b = section("budget")
    v = section("video")
    m = section("mock")
    return AppConfig(
        planner=_provider(section("planner")),
        simulator=_provider(section("simulator")),
        generator=_provider(section("generator")),
        auditor=_provider(section("auditor")),
        repairer=_provider(section("repairer")),
        thresholds=Thresholds(**{k: t[k] for k in t if k in Thresholds.__dataclass_fields__}),
        budget=Budget(**{k: b[k] for k in b if k in Budget.__dataclass_fields__}),
        video=VideoCfg(
            fps=int(v.get("fps", 24)),
            segment_seconds=float(v.get("segment_seconds", 3.0)),
            resolution=list(v.get("resolution", [1280, 720])),
            renderer=str(v.get("renderer", "3d")),
        ),
        mock=MockCfg(
            noise_seed=int(m.get("noise_seed", 42)),
            inject_anomaly=bool(m.get("inject_anomaly", True)),
        ),
        audit=AuditCfg(
            background_check=bool(section("audit").get("background_check", True)),
        ),
    )


def restore_secrets(cfg: AppConfig) -> AppConfig:
    """resume 场景：checkpoint 快照不含密钥，这里从环境变量还原。

    支持多供应商：各模块 api_key 可引用不同的 ${ENV_VAR}（如 LLM/VLM 一家、
    视频生成另一家）。还原规则：
    - None / REDACTED → 回落默认共享密钥 PHYSGEN_API_KEY
    - ${VAR} 占位 → 按变量名还原；未定义则保留占位（调用时报明确错误）
    """
    placeholder = ("REDACTED",)  # 历史 state.json 清洗后的占位
    for module in ("planner", "simulator", "generator", "auditor", "repairer"):
        pc: ProviderCfg = getattr(cfg, module)
        v = pc.api_key
        if not v or v in placeholder:
            pc.api_key = os.environ.get("PHYSGEN_API_KEY")
        elif v.startswith("${") and v.endswith("}"):
            resolved = os.environ.get(v[2:-1])
            if resolved:
                pc.api_key = resolved
    return cfg


def dump_config(cfg: "AppConfig | dict", path: str | Path) -> None:
    """把配置回写为 YAML（checkpoint 用）。接受 AppConfig 或已序列化的 dict。"""
    if isinstance(cfg, dict):
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        return
    d = {
        "planner": {"provider": cfg.planner.provider},
        "simulator": {"provider": cfg.simulator.provider},
        "generator": {"provider": cfg.generator.provider},
        "auditor": {"provider": cfg.auditor.provider},
        "repairer": {"provider": cfg.repairer.provider},
        "thresholds": cfg.thresholds.__dict__,
        "budget": cfg.budget.__dict__,
        "video": cfg.video.__dict__,
        "mock": cfg.mock.__dict__,
        "audit": cfg.audit.__dict__,
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, allow_unicode=True, sort_keys=False)


def from_pipeline_dict(d: dict[str, Any]) -> AppConfig:
    """从 checkpoint 的 config 字典恢复 AppConfig（密钥需另行 restore_secrets）。"""
    return AppConfig(
        planner=_provider(d.get("planner", {})),
        simulator=_provider(d.get("simulator", {})),
        generator=_provider(d.get("generator", {})),
        auditor=_provider(d.get("auditor", {})),
        repairer=_provider(d.get("repairer", {})),
        thresholds=Thresholds(**d.get("thresholds", {})),
        budget=Budget(**d.get("budget", {})),
        video=VideoCfg(**{k: v for k, v in d.get("video", {}).items()
                          if k in VideoCfg.__dataclass_fields__}),
        mock=MockCfg(**d.get("mock", {})),
        audit=AuditCfg(**{k: v for k, v in d.get("audit", {}).items()
                          if k in AuditCfg.__dataclass_fields__}),
    )
