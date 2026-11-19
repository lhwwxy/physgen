from __future__ import annotations

"""HTML 审查报告生成：逐段检查项 + 异常清单 + 修复记录 + 全局逐帧审查结果。"""

import html
import os
import time

from ..types import AuditReport


def _esc(s: str) -> str:
    return html.escape(str(s))


def render_report(run_dir: str, request_text: str, segment_reports: list[AuditReport],
                  global_report: AuditReport | None, repair_log: list[str],
                  final_clips: list[dict]) -> str:
    parts = [f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>PhysGen 物理审查报告</title>
<style>
body {{ font-family: -apple-system, "PingFang SC", sans-serif; margin: 24px; background:#f7f8fa; }}
h1,h2 {{ color:#1a2b4c; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0; background:#fff; }}
th, td {{ border: 1px solid #dde2ea; padding: 6px 10px; font-size: 13px; text-align:left; }}
th {{ background:#eef2f7; }}
.pass {{ color:#188038; }} .fail {{ color:#c5221f; font-weight:bold; }}
.anomaly {{ background:#fdecea; }}
.log {{ background:#fff; border:1px solid #dde2ea; padding:12px; font-size:13px; }}
</style></head><body>
<h1>PhysGen 物理一致性审查报告</h1>
<p>需求：{_esc(request_text)}<br>生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}</p>"""]

    def table_for(rep: AuditReport, title: str) -> str:
        rows = []
        for a in rep.anomalies:
            rows.append(
                f'<tr class="anomaly"><td>{_esc(a.obj_id)}</td><td>{_esc(a.metric)}</td>'
                f'<td>帧 {a.frame_index if a.frame_index is not None else "-"}</td>'
                f'<td colspan="2">{_esc(a.detail)}</td><td>{_esc(a.suggest_fix)}</td></tr>')
        anomaly_html = ("<h3>异常清单</h3><table><tr><th>物体</th><th>指标</th>"
                        "<th>帧</th><th colspan='2'>详情</th><th>建议修复</th></tr>"
                        + "".join(rows) + "</table>") if rows else \
                       "<p class='pass'>本段无异常</p>"
        n_fail = sum(1 for c in rep.checks if not c.passed)
        return (f"<h2>{title}</h2><p>检查项 {len(rep.checks)} 条，"
                f"未通过 <span class='fail'>{n_fail}</span> 条</p>{anomaly_html}")

    for rep in segment_reports:
        parts.append(table_for(rep, f"段 {rep.segment_index}（生成阶段审查）"))
    if global_report is not None:
        parts.append(table_for(global_report, "全局逐帧审查（成片后统一审查）"))

    parts.append("<h2>修复记录</h2><div class='log'>")
    parts.append("<br>".join(_esc(x) for x in repair_log) or "无修复动作")
    parts.append("</div>")

    parts.append("<h2>成片</h2><table><tr><th>段</th><th>帧数</th><th>视频文件</th>"
                 "<th>GIF 预览</th></tr>")
    for c in final_clips:
        parts.append(f"<tr><td>{c['segment']}</td><td>{c['frames']}</td>"
                     f"<td>{_esc(c['video'] or '（无 ffmpeg，仅帧序列 PNG）')}</td>"
                     f"<td>{_esc(c.get('gif') or '-')}</td></tr>")
    parts.append("</table></body></html>")

    path = os.path.join(run_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    return path
