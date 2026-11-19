from __future__ import annotations

"""PhysGen CLI。

用法：
    physgen setup [--install]       # 部署体检：依赖/ffmpeg/密钥/配置
    physgen run "一块10×10米的土地上盖一间8×6×3米的房子"
    physgen resume runs/<run_id>
    physgen report runs/<run_id>     # 重新渲染审查报告
"""

import os
import time

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(help="PhysGen — 物理一致性视频生成流水线", no_args_is_help=True)
console = Console()


def _load_config(path: str | None):
    from physgen.config import load_config
    return load_config(path)


@app.command()
def run(prompt: str,
        config: str = typer.Option(None, "--config", "-c", help="配置文件路径"),
        out: str = typer.Option(None, "--out", "-o", help="运行目录（默认 runs/<时间戳>）"),
        duration: float = typer.Option(None, "--duration", "-d", help="视频总时长(秒)")):
    """运行完整流水线：规划→模拟→生成→审查→修复→全局逐帧审查。"""
    cfg = _load_config(config)
    run_dir = out or os.path.join("runs", time.strftime("run_%Y%m%d_%H%M%S"))
    from physgen.orchestrator import Pipeline
    from physgen.types import UserRequest

    pipeline = Pipeline(cfg, run_dir)
    console.print(Panel(f"需求: {prompt}\n输出目录: {run_dir}\n"
                        f"providers: planner={cfg.planner.provider} "
                        f"simulator={cfg.simulator.provider} "
                        f"generator={cfg.generator.provider} "
                        f"auditor={cfg.auditor.provider} "
                        f"repairer={cfg.repairer.provider}",
                        title="PhysGen"))
    request = UserRequest(text=prompt, duration_s=duration)
    report_path = pipeline.run(request)
    console.print(f"[green]完成[/green] 审查报告: {report_path}")


@app.command()
def resume(run_dir: str, config: str = typer.Option(None, "--config", "-c")):
    """从 checkpoint 断点续跑。"""
    cfg = _load_config(config)
    from physgen.orchestrator import Pipeline
    pipeline = Pipeline(cfg, run_dir)
    report_path = pipeline.resume(run_dir)
    console.print(f"[green]续跑完成[/green] 审查报告: {report_path}")


@app.command()
def setup(config: str = typer.Option(None, "--config", "-c"),
          install: bool = typer.Option(False, "--install",
                                       help="自动安装缺失的真实后端依赖")):
    """部署体检：Python 版本、依赖、ffmpeg、密钥与配置完整性。"""
    from physgen.orchestrator.setup import preflight
    cfg = _load_config(config)
    for line in preflight(cfg, install=install):
        console.print(line)


@app.command()
def report(run_dir: str):
    """打开某次运行的审查报告（打印路径）。"""
    path = os.path.join(run_dir, "report.html")
    if os.path.exists(path):
        console.print(f"审查报告: {os.path.abspath(path)}")
    else:
        console.print(f"[red]未找到报告: {path}[/red]")
        raise typer.Exit(1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
