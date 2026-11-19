"""部署体检与配置驱动模型名的测试。"""

import pytest

from physgen.config import load_config, restore_secrets
from physgen.orchestrator.setup import preflight


def test_preflight_mock_config_clean(tmp_path):
    """全 Mock 配置下体检应通过且不提示安装 real 依赖。"""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "planner: {provider: mock}\n"
        "simulator: {provider: mock}\n"
        "generator: {provider: mock}\n"
        "auditor: {provider: mock}\n"
        "repairer: {provider: mock}\n")
    cfg = load_config(cfg_path)
    lines = preflight(cfg)
    assert any("全部模块为 Mock，无需密钥" in l for l in lines)
    assert not any("安装命令" in l for l in lines)


def test_preflight_real_provider_needs_key(tmp_path, monkeypatch):
    """真实 provider 但无 .env / 密钥 → 明确提示。"""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "planner: {provider: mock}\n"
        "simulator: {provider: mock}\n"
        "generator: {provider: seedance, model: test-model}\n"
        "auditor: {provider: mock}\n"
        "repairer: {provider: mock}\n")
    cfg = load_config(cfg_path)
    monkeypatch.chdir(tmp_path)  # .env 检查按 CWD，避免命中仓库根的 .env
    lines = preflight(cfg)
    assert any("未找到 .env" in l for l in lines)


def test_real_provider_requires_model_in_config(tmp_path, monkeypatch):
    """模型名必须来自配置文件：缺配置时给出指路报错，而非静默用内置默认。"""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "planner: {provider: mock}\n"
        "simulator: {provider: mock}\n"
        "generator: {provider: seedance}\n"   # 没写 model
        "auditor: {provider: mock}\n"
        "repairer: {provider: mock}\n")
    cfg = load_config(cfg_path)
    from physgen.orchestrator.providers import create_generator
    with pytest.raises(ValueError, match="config.yaml 缺少配置.*generator.*model"):
        create_generator(cfg)


def test_real_modules_require_model_and_base_url():
    """仓库公开约束：真实后端类不内置任何模型 id 或端点，必须由配置提供。"""
    import pytest
    with pytest.raises(ValueError, match="model 和 base_url"):
        from physgen.planner.llm import LLMPlanner
        LLMPlanner(model=None, base_url=None)
    with pytest.raises(ValueError, match="model 和 base_url"):
        from physgen.auditor.vlm import VLMAuditor
        VLMAuditor(model=None, base_url=None)
    with pytest.raises(ValueError, match="base_url"):
        from physgen.repairer.seedance import SeedanceRepairer
        SeedanceRepairer(api_key="k", model="m", image_model="i", base_url=None)
    with pytest.raises(ValueError, match="model 和"):
        from physgen.generator.seedance import SeedanceGenerator
        SeedanceGenerator(api_key="k", model="m", base_url=None)


def test_multi_vendor_keys(monkeypatch, tmp_path):
    """多供应商：各模块 api_key 可引用不同环境变量，独立解析；未定义时保留占位。"""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "planner: {provider: llm, model: m1, base_url: 'https://a/v1', api_key: '${LLM_KEY}'}\n"
        "simulator: {provider: mock}\n"
        "generator: {provider: seedance, model: m2, base_url: 'https://b/v1', api_key: '${VIDEO_KEY}'}\n"
        "auditor: {provider: vlm, model: m3, base_url: 'https://a/v1', api_key: '${LLM_KEY}'}\n"
        "repairer: {provider: seedance, model: m4, image_model: i, base_url: 'https://b/v1', api_key: '${VIDEO_KEY}'}\n")
    monkeypatch.setenv("LLM_KEY", "llm-vendor-key")
    monkeypatch.setenv("VIDEO_KEY", "video-vendor-key")
    monkeypatch.delenv("PHYSGEN_API_KEY", raising=False)
    cfg = load_config(cfg_path)
    assert cfg.planner.api_key == "llm-vendor-key"
    assert cfg.generator.api_key == "video-vendor-key"
    assert cfg.auditor.api_key == "llm-vendor-key"
    assert cfg.repairer.api_key == "video-vendor-key"
    # resume 场景：已解析的 key 不被默认共享 key 覆盖；未定义变量保留占位
    monkeypatch.setenv("PHYSGEN_API_KEY", "shared-default")
    cfg.generator.api_key = "REDACTED"          # 快照脱敏占位 → 回落共享 key
    cfg.repairer.api_key = "${MISSING_VAR}"     # 未定义变量 → 保留占位
    restore_secrets(cfg)
    assert cfg.generator.api_key == "shared-default"
    assert cfg.repairer.api_key == "${MISSING_VAR}"


def test_single_key_env_only(monkeypatch):
    """仓库公开约束：除 PHYSGEN_API_KEY 外，代码不引用任何模块专属密钥变量。"""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    banned = ("PLANNER_API_KEY", "GENERATOR_API_KEY", "AUDITOR_API_KEY",
              "REPAIRER_API_KEY", "ARK_API_KEY")
    for py in (root / "physgen").rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for b in banned:
            assert b not in text, f"{py.name} 残留废弃密钥变量 {b}"


def test_no_hardcoded_model_ids():
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "physgen" / "orchestrator" / "providers.py").read_text(encoding="utf-8")
    for bad in ("gpt-4o", "doubao-seedance", "doubao-seedream", "glm-"):
        assert bad not in src, f"providers.py 中残留硬编码模型名: {bad}"
