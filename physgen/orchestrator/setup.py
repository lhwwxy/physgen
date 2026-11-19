from __future__ import annotations

"""部署体检：检查运行环境、可选依赖与配置完整性。

`physgen setup` 调用 preflight()；真实后端依赖（Genesis/torch/ffmpeg 等）
不在基础安装里，由部署阶段按需下载（pip install -e ".[real]"）。
"""

import importlib.util
import os
import shutil
import sys

REAL_MODULES = {
    "matplotlib": "Mock/Genesis 首帧 3D 渲染（缺失时降级 2D）",
    "genesis": "本地 3D 物理模拟（缺失时降级 MockSimulator）",
    "torch": "Genesis 运行时依赖",
    "imageio": "视频下载与抽帧（真实生成必需）",
}

REQUIRED_MODULES = {
    "yaml": "配置解析",
    "typer": "命令行",
    "requests": "真实后端 API 调用",
    "PIL": "图像处理",
}


def preflight(cfg, install: bool = False) -> list[str]:
    """返回逐行体检报告。install=True 时自动安装缺失的可选依赖。"""
    lines: list[str] = []
    add = lines.append

    # 1. Python 版本
    v = sys.version_info
    ok_py = v >= (3, 10)
    add(f"[{'OK' if ok_py else 'WARN'}] Python {v.major}.{v.minor}"
        + ("" if ok_py else "（Genesis 需要 >=3.10，基础 Mock 功能不受影响）"))

    # 2. 必需依赖
    for mod, desc in REQUIRED_MODULES.items():
        ok = importlib.util.find_spec(mod) is not None
        add(f"[{'OK' if ok else 'FAIL'}] {mod} — {desc}"
            + ("" if ok else "（请重装：pip install -e .）"))

    # 3. 可选依赖（真实后端）
    missing_real = [m for m in REAL_MODULES
                    if importlib.util.find_spec(m) is None]
    for mod, desc in REAL_MODULES.items():
        ok = importlib.util.find_spec(mod) is not None
        add(f"[{'OK' if ok else 'SKIP'}] {mod} — {desc}")
    real_needed = (cfg.simulator.provider == "genesis"
                   or any(getattr(cfg, m).provider not in ("mock",)
                          for m in ("planner", "generator", "auditor", "repairer")))
    if missing_real and real_needed:
        cmd = 'pip install -e ".[real]"'
        if install:
            add(f"[RUN ] 正在安装真实后端依赖: {cmd}")
            import subprocess
            r = subprocess.run([sys.executable, "-m", "pip", "install",
                                "-e", ".[real]"], capture_output=True, text=True)
            add(f"[{'OK' if r.returncode == 0 else 'FAIL'}] 安装"
                + ("" if r.returncode == 0 else f"（{r.stderr[-300:]}）"))
        else:
            add(f"[HINT ] 缺 {len(missing_real)} 个可选依赖，安装命令: {cmd}"
                f"（或 physgen setup --install）")

    # 4. ffmpeg（可选：imageio-ffmpeg 会自动下载备用二进制）
    has_ffmpeg = shutil.which("ffmpeg") is not None
    add(f"[{'OK' if has_ffmpeg else 'SKIP'}] ffmpeg 系统命令"
        + ("" if has_ffmpeg else
           "（未装也可用：imageio-ffmpeg 会在首次抽帧时自动获取）"))

    # 5. 密钥与端点
    real_providers = [m for m in ("planner", "generator", "auditor", "repairer")
                      if getattr(cfg, m).provider not in ("mock",)]
    if not real_providers:
        add("[OK  ] 全部模块为 Mock，无需密钥")
    else:
        if os.path.exists(".env"):
            add("[OK  ] .env 存在")
        elif any(getattr(cfg, m).api_key for m in real_providers):
            add("[OK  ] 密钥已通过环境变量/配置提供")
        else:
            add("[WARN ] 检测到真实 provider 但未找到 .env / 密钥，"
                "请复制 .env.example 为 .env 并填写 PHYSGEN_API_KEY")
        for m in real_providers:
            pc = getattr(cfg, m)
            if not pc.base_url:
                add(f"[FAIL ] config.yaml [{m}] 缺 base_url（真实后端必填，"
                    f"代码不内置任何端点）")
            key = pc.api_key
            if not key:
                add(f"[WARN ] [{m}] 未配置密钥（.env / 环境变量 PHYSGEN_API_KEY）")
            elif key.startswith("${"):
                add(f"[WARN ] [{m}] 密钥引用 {key}，但该变量未在 .env 中定义"
                    f"（多供应商场景请补齐对应变量）")
            else:
                add(f"[OK  ] [{m}] 密钥已解析（{key[:6]}***）")

    # 6. 配置摘要
    for m in ("planner", "simulator", "generator", "auditor", "repairer"):
        pc = getattr(cfg, m)
        model = pc.model or pc.image_model or ""
        add(f"      {m:9s} provider={pc.provider:9s} {model}")
    return lines
