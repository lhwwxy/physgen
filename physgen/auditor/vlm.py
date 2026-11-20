from __future__ import annotations

"""VLMAuditor（M3）：画面级物理审查 —— VLM 看生成视频抽帧 + GT 期望值对比。

与数值审查器（MockAuditor）的分工：
- 数值层：GT 轨迹/尺寸逐帧比对（精确、无 API 成本）
- 本层（常识判断）：物体是否出现在画面里、比例是否与 GT 一致、
  重物是否有重量感（下落加速/落地冲击）、悬浮/穿模/凭空消失等违规

接入方式：
    config.yaml: auditor: {provider: vlm, model: ..., base_url: ..., api_key: ...}
    密钥经 .env（PHYSGEN_API_KEY）注入
"""

import base64
import io
import json
import os

import requests
from PIL import Image

from ..config import Thresholds
from ..types import (Anomaly, AuditReport, CheckItem, SimResult, VideoClip)
from .base import PhysicsAuditor

AUDIT_PROMPT = """你是物理一致性审查员。第 1 张图是物理模拟器渲染的标准首帧（含人物比例参照），\
其余是 AI 生成视频的抽帧。期望场景：
{expected}

请逐项审查生成帧，输出 JSON（不要解释）：
{{
  "objects_visible": {{"物体id": bool, ...}},          // 各物体是否在生成帧中可见
  "scale_consistent": bool,                            // 物体相对比例是否符合期望（房子应明显大于石块等）
  "anomalies": [
    {{"frame_index": 整数或null, "obj_id": "物体id或scene",
      "severity": "low|medium|high",
      "detail": "具体异常（如：石块凭空消失/房子比例明显过大/物体悬浮/穿模）",
      "suggest_fix": "修复建议"}}
  ]
}}
审查要点：物体缺失、尺寸比例失真（对照人与房子/石块的比例）、重物下落缺少加速感、\
落地无冲击、物体悬浮、穿模。没有问题 anomalies 给空数组。"""

# audit 模式 通道①：验收核对（用户请求 = 验收标准，白名单原则）
EXPECT_PROMPT = """你是交付验收审查员。以下是视频抽帧，请对照【期望属性清单】核对画面。

【期望属性清单】（来自用户请求，只有这些算验收标准）：
{expected}

【检查规则（严格遵守）】
1. 只核对清单项：尺寸比例、数量、材质、空间关系、是否出现
2. 未提及的外观细节（具体形状、纹理、画风、画质、背景内容）一律不判异常
3. 明显违反物理的画面（物体悬浮/穿模/重物轻放）独立于清单也要报告

输出 JSON（不要解释）：
{{
  "expectations_met": bool,                 // 清单项是否全部满足
  "anomalies": [
    {{"frame_index": 整数或null, "obj_id": "物体id或scene",
      "severity": "low|medium|high",
      "detail": "与清单哪一项不符 + 画面实际是什么",
      "suggest_fix": "修复建议"}}
  ]
}}
没有问题 anomalies 给空数组。"""

# audit 模式 通道②：物理巡检（全画面，含背景物体间冲突）
PATROL_PROMPT = """你是物理一致性巡检员。以下是视频抽帧，请检查画面中【所有】物体的\
物理合理性，包括主体之外背景里的物体以及它们之间的相互关系。

检查要点：
- 物体悬浮/穿模/嵌进地面或墙面
- 物体之间的互相穿插、支撑关系错误
- 阴影方向与光源矛盾、倒影/反射错乱
- 物体凭空出现/消失（跨帧对比）
- 大质量物体下落缺少加速感、落地无冲击

注意：画风/画质/背景内容本身不是异常，只报告物理上的矛盾。
输出 JSON（不要解释）：
{{
  "anomalies": [
    {{"frame_index": 整数或null, "obj_id": "物体id或scene",
      "severity": "low|medium|high",
      "detail": "具体物理矛盾",
      "suggest_fix": "修复建议"}}
  ]
}}
没有问题 anomalies 给空数组。"""


def _frame_to_data_uri(path: str, max_px: int = 640) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_px, max_px), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


class VLMAuditor(PhysicsAuditor):

    def __init__(self, thresholds: Thresholds | None = None,
                 api_key: str | None = None, model: str | None = None,
                 base_url: str | None = None,
                 max_frames: int = 4):
        # 模型名与端点均由 config.yaml [auditor] 提供（模型需支持 image_input）
        if not model or not base_url:
            raise ValueError(
                "VLMAuditor 需要 config.yaml [auditor] 段配置 model 和 base_url，"
                "代码不内置任何端点")
        super().__init__(thresholds)
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_frames = max_frames
        self.inject_anomaly = False  # 与 MockAuditor 的属性对齐（pipeline 会关掉它）
        self.spec = None             # Pipeline 在 plan 后注入 SceneSpec

    # ---------- 期望值描述 ----------

    @staticmethod
    def _expected_text(spec, sim: SimResult) -> str:
        lines = []
        g = spec.ground
        lines.append(f"地块 {g.length_m}×{g.width_m} 米")
        for o in spec.objects:
            l, w, h = o.dims_m
            dyn = "，动态物体（会下落）" if o.is_dynamic else ""
            lines.append(f"- {o.id}（{o.name}）：{l}×{w}×{h} 米，"
                         f"约 {o.mass_kg:.0f} kg{dyn}")
        for ev in spec.events or []:
            if ev.type == "fall":
                drop = ev.params.get("drop_height_m", 2.0)
                lines.append(f"- 事件：{ev.actors} 在 {ev.t_start}s 后从约 {drop} 米"
                             f"高度下落到地面（应有加速下落+落地冲击）")
        lines.append(f"本段时长约 {sim.n_frames / sim.fps:.0f} 秒")
        return "\n".join(lines)

    # ---------- 主流程 ----------

    def audit(self, clip: VideoClip, sim: SimResult,
              fix_factor: float = 1.0) -> AuditReport:
        if not self.api_key:
            raise RuntimeError("缺少密钥：请在 .env 配置 PHYSGEN_API_KEY（或 config.yaml [auditor].api_key）")
        try:
            return self._audit_inner(clip, sim)
        except Exception as e:  # noqa: BLE001 — 网络/服务异常不中断流水线
            print(f"[vlm-auditor] 审查调用失败（{e}），本段降级为通过")
            return AuditReport(segment_index=sim.segment_index,
                               checks=[], anomalies=[])

    def _audit_inner(self, clip: VideoClip, sim: SimResult) -> AuditReport:
        spec = self._spec_of(sim)
        if spec is None:
            return AuditReport(segment_index=sim.segment_index, checks=[], anomalies=[])

        # 抽帧：首帧 + 均匀采样（最多 max_frames 张生成帧）
        frames = clip.frame_paths
        n = max(1, self.max_frames)
        idxs = sorted({0, len(frames) // 3, len(frames) // 2,
                       2 * len(frames) // 3, len(frames) - 1})[:n + 1]
        sampled = [frames[i] for i in idxs if 0 <= i < len(frames)]

        content: list[dict] = [{
            "type": "text",
            "text": AUDIT_PROMPT.format(expected=self._expected_text(spec, sim))}]
        gt_first = sim.first_frame_path
        if gt_first and os.path.exists(gt_first):
            content.append({"type": "image_url",
                            "image_url": {"url": _frame_to_data_uri(gt_first)}})
        for fp in sampled:
            content.append({"type": "image_url",
                            "image_url": {"url": _frame_to_data_uri(fp)}})

        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json={"model": self.model, "temperature": 0.1,
                  "messages": [{"role": "user", "content": content}]},
            timeout=180)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]

        return self._parse_report(text, sim.segment_index, spec)

    # ---------- 解析 ----------

    @staticmethod
    def _extract_json(text: str) -> dict:
        s = text.strip()
        if "```" in s:
            s = s.split("```")[1]
            if s.startswith("json"):
                s = s[4:]
        return json.loads(s.strip())

    def _parse_report(self, text: str, seg_idx: int, spec,
                      metric: str = "vlm_commonsense") -> AuditReport:
        checks: list[CheckItem] = []
        anomalies: list[Anomaly] = []
        try:
            data = self._extract_json(text)
        except Exception as e:  # noqa: BLE001 — VLM 输出坏 JSON 不能阻断流水线
            print(f"[vlm-auditor] 回复解析失败（{e}），本段按通过处理")
            return AuditReport(segment_index=seg_idx, checks=checks,
                               anomalies=anomalies)

        visible = data.get("objects_visible") or {}
        for oid, ok in visible.items():
            checks.append(CheckItem(obj_id=str(oid), metric="vlm_visible",
                                    frame_index=None, expected=1.0,
                                    observed=1.0 if ok else 0.0,
                                    error_pct=0.0 if ok else 1.0,
                                    passed=bool(ok),
                                    detail="物体可见" if ok else "物体不可见"))
        checks.append(CheckItem(obj_id="scene", metric="vlm_scale_consistent",
                                frame_index=None, expected=1.0,
                                observed=1.0 if data.get("scale_consistent") else 0.0,
                                error_pct=0.0 if data.get("scale_consistent") else 1.0,
                                passed=bool(data.get("scale_consistent")),
                                detail="比例一致性"))

        for a in (data.get("anomalies") or []):
            fi = a.get("frame_index")
            anomalies.append(Anomaly(
                segment_index=seg_idx,
                frame_index=int(fi) if isinstance(fi, (int, float)) else None,
                obj_id=str(a.get("obj_id") or "scene"),
                metric=metric,
                severity=str(a.get("severity") or "medium"),
                detail=str(a.get("detail") or ""),
                suggest_fix=str(a.get("suggest_fix") or "")))
        return AuditReport(segment_index=seg_idx, checks=checks, anomalies=anomalies)

    # ---------- audit 模式双通道（外部视频质检） ----------

    def _sample_frames(self, clip: VideoClip) -> list[str]:
        frames = clip.frame_paths
        n = max(1, self.max_frames)
        idxs = sorted({0, len(frames) // 3, len(frames) // 2,
                       2 * len(frames) // 3, len(frames) - 1})[:n + 1]
        return [frames[i] for i in idxs if 0 <= i < len(frames)]

    @staticmethod
    def _expectations_text(spec) -> str:
        """从 SceneSpec 生成期望属性清单（白名单原则：只列请求中明确的项）。"""
        lines = [f"地块 {spec.ground.length_m}×{spec.ground.width_m} 米"]
        for o in spec.objects:
            l, w, h = o.dims_m
            lines.append(f"- {o.id}（{o.name}）：{l}×{w}×{h} 米，"
                         f"材质 {o.material}，约 {o.mass_kg:.0f} kg")
        for ev in spec.events or []:
            if ev.type == "fall":
                drop = ev.params.get("drop_height_m", 2.0)
                lines.append(f"- 事件：{ev.actors} 在 {ev.t_start}s 后从约 {drop} 米"
                             f"高度下落到地面")
        return "\n".join(lines)

    def _audit_lane(self, clip: VideoClip, spec, prompt_tpl: str,
                    metric: str, include_expected: bool) -> AuditReport:
        if not self.api_key:
            raise RuntimeError("缺少密钥：请在 .env 配置 PHYSGEN_API_KEY（或 config.yaml [auditor].api_key）")
        try:
            text = (prompt_tpl.format(expected=self._expectations_text(spec))
                    if include_expected else prompt_tpl)
            content: list[dict] = [{"type": "text", "text": text}]
            for fp in self._sample_frames(clip):
                content.append({"type": "image_url",
                                "image_url": {"url": _frame_to_data_uri(fp)}})
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json={"model": self.model, "temperature": 0.1,
                      "messages": [{"role": "user", "content": content}]},
                timeout=180)
            r.raise_for_status()
            reply = r.json()["choices"][0]["message"]["content"]
            return self._parse_report(reply, clip.segment_index, spec, metric=metric)
        except Exception as e:  # noqa: BLE001 — 审查失败不中断质检流程
            print(f"[vlm-auditor] {metric} 调用失败（{e}），本通道按通过处理")
            return AuditReport(segment_index=clip.segment_index,
                               checks=[], anomalies=[])

    def audit_expectations(self, clip: VideoClip, spec) -> AuditReport:
        """通道① 验收核对：画面 vs 用户请求期望清单。"""
        return self._audit_lane(clip, spec, EXPECT_PROMPT, "acceptance_check", True)

    def audit_background(self, clip: VideoClip, spec) -> AuditReport:
        """通道② 物理巡检：全画面物理一致性（含背景物体间冲突）。"""
        return self._audit_lane(clip, spec, PATROL_PROMPT, "physics_patrol", False)

    # ---------- 工具 ----------

    def _spec_of(self, sim: SimResult):
        """SceneSpec 由 Pipeline 在 plan() 后注入（self.spec）。"""
        return self.spec
