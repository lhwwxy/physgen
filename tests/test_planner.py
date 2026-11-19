import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from physgen.planner.mock import MockPlanner
from physgen.types import UserRequest


def test_parse_ground_and_house():
    p = MockPlanner()
    spec = p.plan(UserRequest(text="一块10×10米的土地上盖一间8×6×3米的房子"))
    assert spec.ground.length_m == 10 and spec.ground.width_m == 10
    houses = [o for o in spec.objects if "house" in o.id]
    assert houses and houses[0].dims_m == (8.0, 6.0, 3.0)
    assert spec.validate() == []


def test_footprint_auto_fix():
    """房子占地超地块 → 规划阶段自动缩到合法（物理违规在源头被拦下）。"""
    p = MockPlanner()
    spec = p.plan(UserRequest(text="一块10×10米的土地上盖一间20×10×5米的房子"))
    house = spec.objects[0]
    assert house.dims_m[0] <= 10.01 and house.dims_m[1] <= 10.01
    assert spec.validate() == []


def test_segments_generated():
    p = MockPlanner()
    spec = p.plan(UserRequest(text="10×10米的场地放一个1×1×1的箱子", duration_s=8.0))
    assert len(spec.segments) >= 2
    assert abs(spec.segments[-1].t_end - 8.0) < 1e-6
