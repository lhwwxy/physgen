from __future__ import annotations

"""MockPlanner：规则引擎解析需求，无 API key 也能产出合法 SceneSpec。

解析能力（正则 + 单位换算）：
- 地块尺寸：如 "10×10米的土地" / "10x10m 地块"
- 物体尺寸：如 "8×6×3米的房子"
- 数量词：如 "两块石头"
- 事件（演示用）：默认给一个重物下落事件，展示重量感审查
"""

import itertools
import re

from ..types import (GroundSpec, MATERIAL_DENSITY, SceneEvent, SceneObject,
                     SceneSpec, SegmentSpec, UserRequest, mass_from_volume)
from .base import Planner

_NUM = r"(\d+(?:\.\d+)?)"
_DIM_SEP = r"[×xX*]\s*"
_UNIT = r"\s*(?:米|m|M)"

_OBJ_KEYWORDS = [
    ("房子", "house", "wood"), ("房屋", "house", "wood"), ("建筑", "building", "concrete"),
    ("石头", "stone", "stone"), ("石块", "stone", "stone"),
    ("箱子", "box", "wood"), ("桌子", "table", "wood"), ("墙", "wall", "brick"),
    ("车", "vehicle", "steel"), ("集装箱", "container", "steel"),
]

# 模糊输入（无尺寸）时的标准默认尺寸（米）
_DEFAULT_DIMS = {
    "house": (8.0, 6.0, 3.0), "building": (12.0, 10.0, 15.0),
    "stone": (1.0, 1.0, 1.0), "box": (1.0, 1.0, 1.0),
    "table": (1.5, 0.8, 0.75), "wall": (6.0, 0.3, 2.5),
    "vehicle": (5.0, 2.0, 1.8), "container": (12.0, 2.4, 2.6),
}
_CN_NUM = {"两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7,
           "八": 8, "九": 9}


class MockPlanner(Planner):

    def plan(self, request: UserRequest) -> SceneSpec:
        text = request.text

        ground, ground_span = self._parse_ground(text)
        objects = self._parse_objects(text, ground, ground_span)
        events = self._default_events(objects)
        segments = self._make_segments(text, request.duration_s, len(objects))

        spec = SceneSpec(
            request_text=text, ground=ground, objects=objects,
            events=events, segments=segments,
            style={"light": "daylight", "camera": "eye-level"},
        )
        # 物理合法性复核： footprint 超地块等 → 规划阶段直接拦截并自动缩到合法值
        problems = spec.validate()
        for p in problems:
            print(f"[planner] SceneSpec 违规已修正: {p}")
        return self._auto_fix(spec, problems)

    # ---------- 解析 ----------

    _DIM_PAT = re.compile(_NUM + _DIM_SEP + _NUM + "(?:" + _DIM_SEP + _NUM + ")?" + _UNIT)
    _KW_POS = r"[\s，。;；、的地在上了]"

    def _parse_ground(self, text: str) -> tuple[GroundSpec, tuple[int, int] | None]:
        """返回 (地块规格, 匹配span)。span 用于从物体候选中排除地块尺寸。"""
        m = re.search(_NUM + _DIM_SEP + _NUM + _UNIT, text)
        if m:
            return GroundSpec(float(m.group(1)), float(m.group(2))), m.span()
        m = re.search(_NUM + _UNIT, text)
        side = float(m.group(1)) if m else 20.0
        return GroundSpec(side, side), (m.span() if m else None)

    def _place(self, ground: GroundSpec, dims, is_dynamic: bool,
               static_seq: list, dyn_idx: int = 0) -> tuple[float, float, float]:
        """摆放策略：静态物体沿中线排列；动态物体放靠近观察者的空地带，
        且必须与已摆放物体保持间距（初始互穿会导致求解器行为异常）。"""
        l, w, h = dims
        if is_dynamic:
            x = ground.length_m * (0.5 + 0.28 * dyn_idx)
            x = max(l / 2, min(x, ground.length_m - l / 2))
            z = min(ground.width_m - w / 2, ground.width_m * 0.9)  # 前侧空地带
            return (round(x, 2), round(h / 2, 2), round(max(z, w / 2), 2))
        x = ground.length_m * (0.5 if not static_seq else min(0.5 + 0.15 * len(static_seq), 0.85))
        return (round(x, 2), round(h / 2, 2), round(ground.width_m / 2, 2))

    def _parse_objects(self, text: str, ground: GroundSpec,
                       ground_span: tuple[int, int] | None) -> list[SceneObject]:
        objects: list[SceneObject] = []
        counter = itertools.count(1)

        # 关键词出现位置（可重复出现）
        kw_hits = [(m.start(), kw, en, material)
                   for kw, en, material in _OBJ_KEYWORDS
                   for m in re.finditer(re.escape(kw), text)]
        used_kw: set[int] = set()

        for dm in self._DIM_PAT.finditer(text):
            if ground_span and dm.start() == ground_span[0]:
                continue  # 这是地块尺寸，不是物体
            # 关联最近的、未占用的关键词（尺寸可在关键词前或后，窗口 15 字符）
            best, best_d = None, 16
            for i, (pos, kw, en, material) in enumerate(kw_hits):
                if i in used_kw:
                    continue
                d = abs(pos - dm.end())
                if d < best_d:
                    best, best_d = i, d
            if best is None:
                continue
            used_kw.add(best)
            _, kw, en, material = kw_hits[best]
            nums = [float(dm.group(1)), float(dm.group(2))]
            h = float(dm.group(3)) if dm.group(3) else 2.0
            dims = (nums[0], nums[1], h)
            objects.append(SceneObject(
                id=f"{en}_{next(counter)}", name=kw, dims_m=dims,
                mass_kg=round(mass_from_volume(dims, material), 1),
                material=material, position=(0, 0, 0),
            ))

        # 模糊输入兜底：出现关键词但没解析到尺寸 → 按类型用标准默认尺寸补建
        for i, (pos, kw, en, material) in enumerate(kw_hits):
            if i in used_kw:
                continue
            dims = _DEFAULT_DIMS.get(en, (1.0, 1.0, 1.0))
            count = self._parse_count(text, pos)
            for k in range(count):
                objects.append(SceneObject(
                    id=f"{en}_{next(counter)}", name=kw, dims_m=dims,
                    mass_kg=round(mass_from_volume(dims, material), 1),
                    material=material, position=(0, 0, 0),
                ))

        if not objects:
            # 兜底：无明确物体时放一个默认石块，保证流水线有内容
            dims = (1.0, 1.0, 1.0)
            objects.append(SceneObject(
                id="stone_1", name="石头", dims_m=dims,
                mass_kg=round(mass_from_volume(dims, "stone"), 1),
                material="stone", position=(0, 0, 0),
            ))
        # 动态标记：只有可移动物体才参与下落演示（建筑/墙等结构物永远是静态的）
        movable_prefix = ("stone", "box", "container", "vehicle")
        dyn_candidates = [o for o in objects if o.id.startswith(movable_prefix)]
        if dyn_candidates:
            max(dyn_candidates, key=lambda o: o.mass_kg).is_dynamic = True
        # 全部确定动/静之后再统一摆放（动态物体让位到前侧空地带）
        static_seq: list[str] = []
        dyn_idx = 0
        for o in objects:
            o.position = self._place(ground, o.dims_m, o.is_dynamic, static_seq,
                                     dyn_idx)
            if o.is_dynamic:
                dyn_idx += 1
            else:
                static_seq.append(o.id)
        return objects

    @staticmethod
    def _parse_count(text: str, kw_pos: int) -> int:
        """解析关键词附近的数量词：'三个箱子' / '2个箱子'，'几个' 默认 2。"""
        window = text[max(0, kw_pos - 4):kw_pos]
        m = re.search(r"([0-9一二两三四五六七八九])个?", window)
        if m:
            c = m.group(1)
            return _CN_NUM.get(c, int(c)) if not c.isdigit() else int(c)
        if "几" in window:
            return 2  # "几个"的保守默认
        return 1

    def _default_events(self, objects: list[SceneObject]) -> list[SceneEvent]:
        dyn = [o for o in objects if o.is_dynamic]
        events: list[SceneEvent] = [SceneEvent(0, 1e9, "none", [], {})]
        if dyn:
            o = dyn[0]
            events.append(SceneEvent(
                t_start=1.0, t_end=2.5, type="fall", actors=[o.id],
                params={"drop_height_m": 2.0}))
        return events

    def _make_segments(self, text: str, duration_s, n_objects: int) -> list[SegmentSpec]:
        total = duration_s if duration_s else min(max(6.0, 2.0 * n_objects), 12.0)
        seg_len = 3.0
        n = max(2, int(total / seg_len + (0 if total % seg_len == 0 else 1)))
        segments = []
        for i in range(n):
            t0, t1 = i * seg_len, min((i + 1) * seg_len, total)
            segments.append(SegmentSpec(
                index=i, t_start=t0, t_end=t1,
                prompt=f"{text}（第{i + 1}/{n}段）",
            ))
        return segments

    # ---------- 修正 ----------

    def _auto_fix(self, spec: SceneSpec, problems: list[str]) -> SceneSpec:
        """自动修正典型物理违规：占地超地块 → 等比缩小。"""
        for obj in spec.objects:
            l, w, h = obj.dims_m
            if not obj.is_dynamic and (l > spec.ground.length_m or w > spec.ground.width_m):
                scale = min(spec.ground.length_m / l, spec.ground.width_m / w)
                obj.dims_m = (round(l * scale, 2), round(w * scale, 2), round(h, 2))
                obj.mass_kg = round(mass_from_volume(obj.dims_m, obj.material), 1)
                obj.position = (spec.ground.length_m / 2, obj.dims_m[2] / 2,
                                obj.position[2])
        return spec
