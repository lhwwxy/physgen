import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from physgen.types import (GroundSpec, SceneEvent, SceneObject, SceneSpec,
                           fall_time, mass_from_volume)


def test_fall_time():
    assert abs(fall_time(9.8) - 2 ** 0.5) < 1e-6  # t=sqrt(2h/g), h=g -> sqrt(2)
    assert fall_time(0) == 0


def test_mass_from_volume():
    # 1m^3 石头 ≈ 2600kg
    assert abs(mass_from_volume((1, 1, 1), "stone") - 2600) < 1e-6


def _spec(ground=(10, 10), dims=(8, 6, 3), material="wood") -> SceneSpec:
    return SceneSpec(
        request_text="test", ground=GroundSpec(*ground),
        objects=[SceneObject("house_1", "房子", dims,
                             mass_from_volume(dims, material), material,
                             (ground[0] / 2, dims[2] / 2, 0.5))],
        events=[SceneEvent(0, 10, "none", [], {})],
        segments=[],
    )


def test_validate_ok():
    assert _spec().validate() == []


def test_validate_footprint_violation():
    # 房子 20×10 米盖在 10×10 地块上 —— 用户描述的典型违规
    problems = _spec(dims=(20, 10, 3)).validate()
    assert problems and "超出地块" in problems[0]


def test_validate_mass_violation():
    spec = _spec(dims=(8, 6, 3))
    spec.objects[0].mass_kg = 1.0  # 明显失真的质量
    problems = spec.validate()
    assert problems and "质量" in problems[0]
