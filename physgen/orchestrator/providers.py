from __future__ import annotations

"""Provider 工厂：按 config 选择 Mock / 真实后端实现。API key 一律走环境变量。"""


from ..auditor.base import PhysicsAuditor
from ..auditor.mock import MockAuditor
from ..auditor.vlm import VLMAuditor
from ..config import AppConfig, ProviderCfg
from ..generator.base import VideoGenerator
from ..generator.http_generic import GenericHTTPGenerator
from ..generator.kling import KlingVideoGenerator
from ..generator.mock import MockVideoGenerator
from ..generator.seedance import SeedanceGenerator
from ..planner.base import Planner
from ..planner.llm import LLMPlanner
from ..planner.mock import MockPlanner
from ..repairer.base import Repairer
from ..repairer.http_generic import GenericHTTPRepairer
from ..repairer.mock import MockRepairer
from ..simulator.base import Simulator
from ..simulator.mock import MockSimulator


def _require_model(p: ProviderCfg, module: str, field: str = "model") -> str:
    """真实后端的模型名一律以 config.yaml 为准，缺配置直接报错指路。"""
    v = getattr(p, field)
    if not v:
        raise ValueError(
            f"config.yaml 缺少配置：[{module}] 段的 {field} 字段"
            f"（provider={p.provider} 需要指定模型名），请补齐后重试")
    return v


def create_planner(cfg: AppConfig) -> Planner:
    p: ProviderCfg = cfg.planner
    if p.provider == "mock":
        return MockPlanner()
    if p.provider == "llm":
        return LLMPlanner(model=_require_model(p, "planner"),
                          api_key=p.api_key or cfg.api_key(),
                          base_url=p.base_url)
    raise ValueError(f"未知 planner provider: {p.provider}")


def create_simulator(cfg: AppConfig) -> Simulator:
    p = cfg.simulator
    if p.provider == "genesis":
        try:
            from ..simulator.genesis import GenesisSimulator
            return GenesisSimulator(fps=cfg.video.fps,
                                    resolution=tuple(cfg.video.resolution))
        except RuntimeError as e:
            print(f"[providers] genesis 不可用（{e}），降级为 MockSimulator")
            return MockSimulator(fps=cfg.video.fps,
                                 resolution=tuple(cfg.video.resolution),
                                 renderer=getattr(cfg.video, "renderer", "3d"))
    if p.provider == "mock":
        return MockSimulator(fps=cfg.video.fps,
                             resolution=tuple(cfg.video.resolution),
                             renderer=getattr(cfg.video, "renderer", "3d"))
    raise ValueError(f"未知 simulator provider: {p.provider}")


def create_generator(cfg: AppConfig) -> VideoGenerator:
    p = cfg.generator
    if p.provider == "mock":
        return MockVideoGenerator()
    if p.provider in ("seedance", "ark"):
        return SeedanceGenerator(api_key=p.api_key or cfg.api_key(),
                                 model=_require_model(p, "generator"),
                                 base_url=p.base_url,
                                 poll_interval_s=p.poll_interval_s,
                                 max_poll_time_s=p.max_poll_time_s or 900)
    if p.provider == "kling":
        return KlingVideoGenerator(api_key=p.api_key or cfg.api_key(),
                                   poll_interval_s=p.poll_interval_s,
                                   base_url=p.base_url)
    if p.provider == "http_generic":
        return GenericHTTPGenerator(api_key=p.api_key or cfg.api_key(),
                                    poll_interval_s=p.poll_interval_s)
    raise ValueError(f"未知 generator provider: {p.provider}")


def create_auditor(cfg: AppConfig) -> PhysicsAuditor:
    p = cfg.auditor
    if p.provider == "mock":
        return MockAuditor(thresholds=cfg.thresholds,
                           noise_seed=cfg.mock.noise_seed,
                           inject_anomaly=cfg.mock.inject_anomaly)
    if p.provider == "vlm":
        return VLMAuditor(thresholds=cfg.thresholds,
                          api_key=p.api_key or cfg.api_key(),
                          model=_require_model(p, "auditor"),
                          base_url=p.base_url)
    raise ValueError(f"未知 auditor provider: {p.provider}")


def create_repairer(cfg: AppConfig) -> Repairer:
    p = cfg.repairer
    if p.provider == "mock":
        return MockRepairer()
    if p.provider == "seedance":
        from ..repairer.seedance import SeedanceRepairer
        return SeedanceRepairer(
            api_key=p.api_key or cfg.api_key(),
            model=_require_model(p, "repairer"),
            image_model=_require_model(p, "repairer", "image_model"),
            base_url=p.base_url,
            max_frames_per_round=cfg.budget.max_frames_per_round)
    if p.provider == "http_generic":
        return GenericHTTPRepairer(api_key=p.api_key or cfg.api_key())
    raise ValueError(f"未知 repairer provider: {p.provider}")
