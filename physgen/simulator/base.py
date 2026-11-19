from __future__ import annotations

"""Simulator 接口：SceneSpec -> 物理模拟（首帧渲染 + 控制信号 + 物理真值）。"""

from ..types import SceneSpec, SimResult


class Simulator:
    """所有模拟器的基类。"""

    def build(self, spec: SceneSpec) -> None:
        self.spec = spec

    def simulate_segment(self, seg_idx: int) -> SimResult:
        raise NotImplementedError
