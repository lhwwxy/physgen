"""密钥管理：.env 加载 / ${VAR} 插值 / checkpoint 快照脱敏。"""

import os

import pytest

from physgen.config import load_config, restore_secrets


def test_env_file_and_interpolation(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_KEY_X", raising=False)
    (tmp_path / ".env").write_text('TEST_KEY_X="abc123"\n# 注释\n空行\n')
    (tmp_path / "cfg.yaml").write_text(
        "planner:\n  provider: llm\n  api_key: ${TEST_KEY_X}\n"
        "simulator:\n  provider: mock\n")
    cfg = load_config(tmp_path / "cfg.yaml")
    assert cfg.planner.api_key == "abc123"


def test_env_var_priority_over_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_KEY_Y", "from-env")
    (tmp_path / ".env").write_text("TEST_KEY_Y=from-dotenv\n")
    (tmp_path / "cfg.yaml").write_text(
        "planner:\n  provider: mock\n  api_key: ${TEST_KEY_Y}\n"
        "simulator:\n  provider: mock\n")
    cfg = load_config(tmp_path / "cfg.yaml")
    assert cfg.planner.api_key == "from-env"   # shell 环境变量优先


def test_missing_var_stays_literal(tmp_path):
    (tmp_path / "cfg.yaml").write_text(
        "planner:\n  api_key: ${DEFINITELY_NOT_SET_VAR}\n"
        "simulator:\n  provider: mock\n")
    cfg = load_config(tmp_path / "cfg.yaml")
    assert cfg.planner.api_key == "${DEFINITELY_NOT_SET_VAR}"


def test_restore_secrets_uses_single_key(monkeypatch):
    """密钥策略：唯一来源 PHYSGEN_API_KEY，不再支持模块专属变量。"""
    monkeypatch.setenv("PHYSGEN_API_KEY", "shared")
    monkeypatch.setenv("GENERATOR_API_KEY", "gen-specific")  # 废弃变量应被忽略
    cfg = load_config()  # 真实 config.yaml，其中 api_key 为 ${PHYSGEN_API_KEY}
    assert cfg.generator.api_key == "shared"  # config.yaml 插值已生效
    # 模拟 checkpoint 快照脱敏后的恢复
    cfg.generator.api_key = None
    cfg.auditor.api_key = None
    restore_secrets(cfg)
    assert cfg.generator.api_key == "shared"
    assert cfg.auditor.api_key == "shared"


def test_config_yaml_has_no_plaintext_key():
    """仓库公开的硬性约束：config.yaml 不允许出现任何 sk- 开头的明文密钥。"""
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parent.parent
    for name in ("config.yaml", ".env.example"):
        text = (root / name).read_text(encoding="utf-8")
        assert not re.search(r"sk-[A-Za-z0-9]{20,}", text), f"{name} 含明文密钥"
