from __future__ import annotations

"""核心数据结构：SceneSpec / SimResult / AuditReport 等。

所有模块间传递的数据一律使用本文件中的 dataclass，保证 Mock 与真实后端
产出/消费相同的数据形状。
"""

import math
from dataclasses import dataclass, field
from typing import Optional

Vec3 = tuple[float, float, float]  # (x, y, z) 米；y 为竖直方向

G = 9.8  # 重力加速度 m/s^2

# 常见材料密度 kg/m^3，供规划器核算质量
MATERIAL_DENSITY = {
    "wood": 600.0, "brick": 1800.0, "concrete": 2400.0, "steel": 7850.0,
    "stone": 2600.0, "water": 1000.0, "plastic": 950.0, "glass": 2500.0,
    "soil": 1600.0, "foam": 80.0, "default": 1000.0,
}


def mass_from_volume(dims: Vec3, material: str) -> float:
    d = MATERIAL_DENSITY.get(material, MATERIAL_DENSITY["default"])
    return dims[0] * dims[1] * dims[2] * d


def fall_time(height_m: float) -> float:
    """自由落体理论时长。"""
    return math.sqrt(2.0 * max(height_m, 0.0) / G)


@dataclass
class GroundSpec:
    length_m: float
    width_m: float


@dataclass
class SceneObject:
    id: str
    name: str
    dims_m: Vec3                # 长 宽 高
    mass_kg: float
    material: str
    position: Vec3              # 物体中心
    is_dynamic: bool = False


@dataclass
class SceneEvent:
    t_start: float
    t_end: float
    type: str                   # move / fall / collide / none
    actors: list[str]
    params: dict = field(default_factory=dict)


@dataclass
class SegmentSpec:
    index: int
    t_start: float
    t_end: float
    prompt: str                 # 该段的视频生成提示词


@dataclass
class SceneSpec:
    request_text: str
    ground: GroundSpec
    objects: list[SceneObject]
    events: list[SceneEvent]
    segments: list[SegmentSpec]
    style: dict = field(default_factory=dict)

    def validate(self) -> list[str]:
        """物理合法性复核：返回违规描述列表（空列表 = 通过）。"""
        problems: list[str] = []
        for obj in self.objects:
            l, w, _h = obj.dims_m
            if min(obj.dims_m) <= 0:
                problems.append(f"{obj.id}: 尺寸必须为正, 得到 {obj.dims_m}")
            expected = mass_from_volume(obj.dims_m, obj.material)
            if obj.mass_kg <= 0:
                problems.append(f"{obj.id}: 质量必须为正")
            elif abs(obj.mass_kg - expected) / expected > 0.5:
                problems.append(
                    f"{obj.id}: 质量 {obj.mass_kg:.0f}kg 与 体积×密度({obj.material}) "
                    f"{expected:.0f}kg 偏差超过 50%")
            # footprint 不得超出地块（静态物体）
            if not obj.is_dynamic:
                if l > self.ground.length_m + 1e-6 or w > self.ground.width_m + 1e-6:
                    problems.append(
                        f"{obj.id}: 占地 {l}×{w}m 超出地块 {self.ground.length_m}×"
                        f"{self.ground.width_m}m —— 正是用户描述的典型物理违规")
        return problems


@dataclass
class ObjectFrameState:
    obj_id: str
    position: Vec3
    velocity: Vec3
    bbox_m: Vec3


@dataclass
class SimResult:
    segment_index: int
    first_frame_path: str
    fps: int
    n_frames: int
    # gt[obj_id] -> 每帧的物理真值
    gt: dict[str, list[ObjectFrameState]]
    control_meta: dict = field(default_factory=dict)


@dataclass
class VideoClip:
    segment_index: int
    frame_paths: list[str]
    fps: int
    video_path: Optional[str] = None   # ffmpeg 可用时生成 mp4，否则 None


@dataclass
class CheckItem:
    obj_id: str
    metric: str                 # dim_l / dim_w / dim_h / traj_rmse / fall_time / mass_feel
    frame_index: Optional[int]  # None = 全段汇总指标
    expected: float
    observed: float
    error_pct: float
    passed: bool
    detail: str = ""


@dataclass
class Anomaly:
    segment_index: int
    frame_index: Optional[int]
    obj_id: str
    metric: str
    severity: str               # low / medium / high
    detail: str
    suggest_fix: str = ""


@dataclass
class AuditReport:
    segment_index: int
    checks: list[CheckItem]
    anomalies: list[Anomaly]

    @property
    def ok(self) -> bool:
        return not self.anomalies

    def anomalous_frames(self) -> set[int]:
        return {a.frame_index for a in self.anomalies if a.frame_index is not None}


@dataclass
class RepairResult:
    segment_index: int
    repaired: bool
    clip: VideoClip
    actions: list[str] = field(default_factory=list)
    rounds: int = 0


@dataclass
class UserRequest:
    text: str
    reference_image: Optional[str] = None
    duration_s: Optional[float] = None
