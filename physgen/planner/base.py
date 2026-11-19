from __future__ import annotations

"""Planner 接口：自然语言需求 -> 结构化 SceneSpec。"""

from ..types import SceneSpec, UserRequest


class Planner:
    """所有规划器的基类。"""

    def plan(self, request: UserRequest) -> SceneSpec:
        raise NotImplementedError

    def replan(self, spec: SceneSpec, problems: list[str]) -> SceneSpec:
        """全局审查发现 SceneSpec 违规时调用（默认原样返回）。"""
        return spec
