from __future__ import annotations

"""MockAuditor：数值真值比对层（纯计算，无需 LLM/VLM）。

审查项（对齐设计文档 §3.4A）：
- dim_l/w/h   物体尺寸误差 %（逐帧）
- traj_rmse   位置轨迹 RMSE（全段汇总）
- pos_err     逐帧位置偏差（动态物体）
- fall_time   落地时长 vs 理论 t=√(2h/g)
- mass_feel   重量感：大质量物体落地冲击速度必须符合 v=√(2gh)（重物不能"轻飘飘"）

视觉估计在 Mock 中 = GT + 受控噪声（noise_seed 保证可复现）。
inject_anomaly=True 时对第 1 段注入超标噪声，演示"审查→修复"闭环。
真实 VLM 审查器接入后替换本实现，产出相同 AuditReport。
"""

import random

from ..config import Thresholds
from ..types import (Anomaly, AuditReport, CheckItem, SimResult,
                     VideoClip, fall_time)
from .base import PhysicsAuditor


class MockAuditor(PhysicsAuditor):

    def __init__(self, thresholds: Thresholds | None = None,
                 noise_seed: int = 42, inject_anomaly: bool = False):
        super().__init__(thresholds)
        self.rng = random.Random(noise_seed)
        self.inject_anomaly = inject_anomaly

    def _observed(self, value: float, rel_noise: float) -> float:
        """模拟视觉估计：真值 × (1 + 噪声)。"""
        return value * (1.0 + self.rng.uniform(-rel_noise, rel_noise))

    def audit(self, clip: VideoClip, sim: SimResult,
              fix_factor: float = 1.0) -> AuditReport:
        th = self.thresholds
        checks: list[CheckItem] = []
        anomalies: list[Anomaly] = []

        # 注入演示异常：只对第 1 段且尚未修复时生效
        inject = (self.inject_anomaly and sim.segment_index == 1 and fix_factor >= 0.999)

        for obj_id, states in sim.gt.items():
            rel = 0.40 if inject else 0.10          # 视觉估计相对噪声
            rel *= fix_factor                        # 修复后噪声衰减
            base_dims = states[0].bbox_m
            frame_pos_err: list[float] = []

            for fi, st in enumerate(states):
                # 尺寸逐帧审查
                for axis, name in ((0, "dim_l"), (1, "dim_w"), (2, "dim_h")):
                    expected = base_dims[axis]
                    observed = self._observed(expected, rel)
                    err = abs(observed - expected) / max(expected, 1e-6) * 100
                    passed = err <= th.dim_error_pct
                    checks.append(CheckItem(obj_id, name, fi, expected, observed,
                                            round(err, 2), passed))
                    if not passed:
                        anomalies.append(Anomaly(
                            sim.segment_index, fi, obj_id, name,
                            "high" if err > 2 * th.dim_error_pct else "medium",
                            f"尺寸{name}偏差 {err:.1f}% 超阈值 {th.dim_error_pct}%",
                            "局部重绘该帧并携带首帧尺寸约束"))
                frame_pos_err.append((fi, st))

            # 动态物体：轨迹 RMSE + 逐帧位置 + 落地时长/重量感
            dyn_states = frame_pos_err
            if self._is_dynamic(obj_id, sim):
                errs = []
                for fi, st in dyn_states:
                    observed_pos = tuple(
                        c + self.rng.uniform(-1, 1) * (0.10 if not inject else 0.8) * fix_factor
                        for c in st.position)
                    e = sum((a - b) ** 2 for a, b in zip(st.position, observed_pos)) ** 0.5
                    errs.append(e)
                    passed = e <= th.trajectory_rmse_m
                    checks.append(CheckItem(obj_id, "pos_err", fi,
                                            0.0, round(e, 3), round(e / max(th.trajectory_rmse_m, 1e-6) * 100, 2),
                                            passed, "位置偏差(m)"))
                    if not passed:
                        anomalies.append(Anomaly(
                            sim.segment_index, fi, obj_id, "pos_err",
                            "high",
                            f"轨迹位置偏差 {e:.2f}m 超阈值 {th.trajectory_rmse_m}m",
                            "携带 GT 轨迹控制信号整段重生成"))
                rmse = (sum(e * e for e in errs) / max(len(errs), 1)) ** 0.5
                checks.append(CheckItem(obj_id, "traj_rmse", None,
                                        th.trajectory_rmse_m, round(rmse, 3),
                                        round(rmse / th.trajectory_rmse_m * 100, 2),
                                        rmse <= th.trajectory_rmse_m, "全段轨迹RMSE(m)"))

                # 落地时长 vs 理论值
                drop = self._drop_height(obj_id, sim)
                if drop is not None:
                    expected_t = fall_time(drop)
                    observed_t = self._observed(expected_t, rel)
                    err = abs(observed_t - expected_t) / expected_t * 100
                    passed = err <= th.fall_time_err_pct
                    checks.append(CheckItem(obj_id, "fall_time", None,
                                            round(expected_t, 3), round(observed_t, 3),
                                            round(err, 2), passed,
                                            f"落地时长 vs √(2h/g), h={drop}m"))
                    if not passed:
                        anomalies.append(Anomaly(
                            sim.segment_index, None, obj_id, "fall_time", "medium",
                            f"落地时长偏差 {err:.1f}% 超阈值 {th.fall_time_err_pct}%",
                            "调整下落时间轴，携带速度曲线控制信号重生成"))

                # 重量感：落地冲击速度应符合自由落体，重物不得"轻飘飘"
                v_expected = (2 * 9.8 * drop) ** 0.5 if drop else 0.0
                v_observed = self._observed(v_expected, rel) if v_expected else 0.0
                if v_expected > 0:
                    err = abs(v_observed - v_expected) / v_expected * 100
                    mass = self._mass(obj_id, sim)
                    heavy = mass > 200
                    passed = err <= th.fall_time_err_pct and (not heavy or err <= 2 * th.fall_time_err_pct)
                    checks.append(CheckItem(obj_id, "mass_feel", None,
                                            round(v_expected, 2), round(v_observed, 2),
                                            round(err, 2), passed,
                                            f"重量感: 质量{mass:.0f}kg 冲击速度 vs √(2gh)"))
                    if not passed:
                        anomalies.append(Anomaly(
                            sim.segment_index, None, obj_id, "mass_feel", "high",
                            f"重物({mass:.0f}kg)落地冲击速度偏差 {err:.1f}% —— 典型的'重物轻放'违规",
                            "增加落地尘土/震动/形变提示词，重生成该段"))

        return AuditReport(segment_index=sim.segment_index,
                           checks=checks, anomalies=anomalies)

    # ---------- helpers ----------

    def _is_dynamic(self, obj_id: str, sim: SimResult) -> bool:
        return any(len(states) > 1 and any(
            s.velocity != states[0].velocity for s in states)
            for states in [sim.gt[obj_id]])

    def _drop_height(self, obj_id: str, sim: SimResult) -> float | None:
        states = sim.gt[obj_id]
        ys = [s.position[1] for s in states]
        if max(ys) - min(ys) < 0.3:
            return None
        return round(max(ys) - min(ys), 2)

    def _mass(self, obj_id: str, sim: SimResult) -> float:
        # Mock 阶段质量从 GT bbox 反查（真实后端由 SceneSpec 直传）
        st = sim.gt[obj_id][0]
        l, w, h = st.bbox_m
        return l * w * h * 2600  # stone 密度近似
