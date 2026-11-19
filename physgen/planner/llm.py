from __future__ import annotations

"""LLMPlanner（M3）：调用 OpenAI 兼容接口做语义解析，确定性代码兜底物理计算。

职责切分（防 LLM 幻觉的关键设计）：
- LLM 只做语义工作：识别物体类型/尺寸/材质/数量、事件类型
- 确定性代码负责：质量计算（体积×密度）、摆放（防互穿）、分段、
  物理合法性校验（footprint/质量）与自动修正
- 任何 API/解析失败 → 自动降级 MockPlanner（流水线不中断）

接入方式：
    config.yaml: planner: {provider: llm, model: ..., base_url: ..., api_key: ...}
    密钥经 .env（PHYSGEN_API_KEY）注入
"""

import json

import requests

from ..types import (GroundSpec, MATERIAL_DENSITY, SceneEvent, SceneObject,
                     SceneSpec, SegmentSpec, UserRequest, mass_from_volume)
from .base import Planner
from .mock import MockPlanner, _DEFAULT_DIMS

VALID_MATERIALS = [m for m in MATERIAL_DENSITY if m != "default"]

SYSTEM_PROMPT = """你是物理场景规划器。把用户的中文需求解析为 JSON：
{
  "ground": {"length_m": float, "width_m": float},   // 地块尺寸，未提则默认 20×20
  "objects": [                                       // 用户提到的每个物体一项
    {"type": "house|building|stone|box|table|wall|vehicle|container",
     "name": "中文名", "dims_m": [长, 宽, 高],
     "material": "wood|brick|concrete|steel|stone|water|plastic|glass|soil|foam",
     "is_dynamic": bool},                            // 只有会被移动/下落的物体才 true
    ...
  ],
  "events": [{"type": "fall|move|collide|none", "actors_index": [物体下标],
              "t_start": 秒, "t_end": 秒, "drop_height_m": 米}]   // 没有则给 [{"type":"none"}]
}
硬性规则：
1. 只用用户给定的尺寸；没给尺寸就用 __DEFAULT_DIMS__ 的默认值。
2. material 从列表里选最接近的：__MATERIALS__。
3. 建筑物/墙等结构 is_dynamic 必须 false。
4. 静态物体占地不得超出地块，超出就按比例缩小 dims_m。
只输出 JSON，不要解释。""".replace("__DEFAULT_DIMS__", json.dumps(_DEFAULT_DIMS, ensure_ascii=False)) \
    .replace("__MATERIALS__", ", ".join(VALID_MATERIALS))


class LLMPlanner(Planner):

    def __init__(self, model: str | None, api_key: str | None = None,
                 base_url: str | None = None):
        # 模型名与端点均由 config.yaml [planner] 提供，代码不内置任何默认值
        if not model or not base_url:
            raise ValueError(
                "LLMPlanner 需要 config.yaml [planner] 段配置 model 和 base_url"
                "（如 base_url: https://your-gateway/v1），代码不内置任何端点")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._fallback = MockPlanner()

    # ---------- LLM 调用 ----------

    def _chat(self, user_text: str) -> str:
        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json={"model": self.model, "temperature": 0.1,
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                               {"role": "user", "content": user_text}]},
            timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _extract_json(text: str) -> dict:
        """容忍 markdown 代码块包裹。"""
        s = text.strip()
        if "```" in s:
            s = s.split("```")[1]
            if s.startswith("json"):
                s = s[4:]
        return json.loads(s.strip())

    # ---------- 主流程 ----------

    def plan(self, request: UserRequest) -> SceneSpec:
        try:
            data = self._extract_json(self._chat(request.text))
            spec = self._to_spec(data, request)
            problems = spec.validate()
            for p in problems:
                print(f"[llm-planner] SceneSpec 违规已自动修正: {p}")
            return self._fallback._auto_fix(spec, problems)
        except Exception as e:  # noqa: BLE001 — 任何失败都降级，流水线不中断
            print(f"[llm-planner] LLM 规划失败（{e}），降级 MockPlanner")
            return self._fallback.plan(request)

    def replan(self, spec: SceneSpec, problems: list[str]) -> SceneSpec:
        """把违规清单发回 LLM 修正；失败则用确定性 auto_fix 兜底。"""
        try:
            brief = {"ground": vars(spec.ground),
                     "objects": [{"type": o.id.rsplit("_", 1)[0], "name": o.name,
                                  "dims_m": list(o.dims_m), "material": o.material,
                                  "is_dynamic": o.is_dynamic} for o in spec.objects]}
            data = self._extract_json(self._chat(
                f"当前场景规划如下，存在这些物理违规：{problems}。"
                f"请修正后输出完整 JSON（同 schema）。\n{json.dumps(brief, ensure_ascii=False)}"))
            return self._to_spec(data, UserRequest(text=spec.request_text))
        except Exception as e:  # noqa: BLE001
            print(f"[llm-planner] replan 失败（{e}），用确定性修正兜底")
            return self._fallback._auto_fix(spec, spec.validate())

    # ---------- JSON -> SceneSpec（确定性兜底部分） ----------

    _TYPE_ALIASES = {
        "房子": "house", "房屋": "house", "建筑": "building", "大楼": "building",
        "石头": "stone", "石块": "stone", "岩石": "stone",
        "箱子": "box", "纸箱": "box", "桌子": "table", "墙": "wall",
        "车": "vehicle", "汽车": "vehicle", "卡车": "vehicle",
        "集装箱": "container",
    }

    def _to_spec(self, data: dict, request: UserRequest) -> SceneSpec:
        g = data.get("ground") or {}
        ground = GroundSpec(float(g.get("length_m", 20.0)),
                            float(g.get("width_m", 20.0)))
        objects: list[SceneObject] = []
        for i, od in enumerate(data.get("objects") or []):
            t = str(od.get("type", "")).lower()
            if t not in _DEFAULT_DIMS:
                t = self._TYPE_ALIASES.get(str(od.get("name", "")).lower()
                                           .strip(), "stone")
            dims = [float(v) for v in (od.get("dims_m") or _DEFAULT_DIMS[t])]
            if len(dims) != 3 or min(dims) <= 0:
                dims = list(_DEFAULT_DIMS[t])
            material = str(od.get("material", "default")).lower()
            if material not in MATERIAL_DENSITY:
                material = "wood" if t in ("house", "box", "table") else "stone"
            objects.append(SceneObject(
                id=f"{t}_{i + 1}", name=str(od.get("name") or t),
                dims_m=tuple(dims),
                mass_kg=round(mass_from_volume(dims, material), 1),  # 密度表兜底，不信 LLM
                material=material, position=(0.0, 0.0, 0.0),
                is_dynamic=bool(od.get("is_dynamic", False))))
        if not objects:
            raise ValueError("LLM 未产出任何物体")

        # 动态标记复核：结构物强制静态（LLM 偶尔会标错）
        for o in objects:
            if o.id.startswith(("house", "building", "wall")):
                o.is_dynamic = False
        if not any(o.is_dynamic for o in objects):
            movable = [o for o in objects
                       if o.id.startswith(("stone", "box", "container", "vehicle"))]
            if movable:
                max(movable, key=lambda o: o.mass_kg).is_dynamic = True

        # 摆放/事件/分段全部交给确定性代码（与 MockPlanner 同一套）
        static_seq: list[str] = []
        dyn_idx = 0
        for o in objects:
            o.position = self._fallback._place(ground, o.dims_m, o.is_dynamic,
                                               static_seq, dyn_idx)
            if o.is_dynamic:
                dyn_idx += 1
            else:
                static_seq.append(o.id)

        events = [SceneEvent(0.0, 1e9, "none", [], {})]
        for ev in (data.get("events") or []):
            if ev.get("type") not in ("fall", "move", "collide"):
                continue
            actors = [objects[int(j)].id
                      for j in (ev.get("actors_index") or [])
                      if int(j) < len(objects)]
            if not actors:
                continue
            events.append(SceneEvent(
                t_start=float(ev.get("t_start", 1.0)),
                t_end=float(ev.get("t_end", 2.5)),
                type=ev["type"], actors=actors,
                params={"drop_height_m": float(ev.get("drop_height_m", 2.0))}))
        if len(events) == 1:  # LLM 没给有效事件 → 默认下落演示
            events += self._fallback._default_events(
                [o for o in objects if o.is_dynamic])[1:]

        segments = self._fallback._make_segments(
            request.text, request.duration_s, len(objects))
        return SceneSpec(request_text=request.text, ground=ground,
                         objects=objects, events=events, segments=segments,
                         style={"light": "daylight", "camera": "eye-level"})
