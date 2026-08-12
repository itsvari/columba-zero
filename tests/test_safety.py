from __future__ import annotations

from conftest import make_config

from app.models import Detection, FramePacket, Target
from app.safety import SafetyGate


def test_stale_frame_is_critical() -> None:
    config = make_config()
    gate = SafetyGate(config.safety, config.model, 0.5, clock=lambda: 1.0)
    result = gate.evaluate(FramePacket(0, 0.0, None), [], None)
    assert not result.allowed
    assert result.critical
    assert result.reason == "stale_frame"


def test_person_blocks_even_when_target_is_valid() -> None:
    config = make_config()
    gate = SafetyGate(config.safety, config.model, 0.5, clock=lambda: 0.0)
    person = Detection("person", 0.8, 1, 1, 20, 30)
    result = gate.evaluate(FramePacket(0, 0.0, None), [person], Target(100, 100, 1, 10))
    assert not result.allowed
    assert not result.critical
    assert result.reason == "blocked_object:person"


def test_open_interlock_is_critical_when_required() -> None:
    config = make_config()
    gate = SafetyGate(
        config.safety,
        config.model,
        0.5,
        interlock_closed=lambda: False,
        clock=lambda: 0.0,
    )
    result = gate.evaluate(FramePacket(0, 0.0, None), [], None, require_interlock=True)
    assert not result.allowed
    assert result.critical
    assert result.reason == "hardware_interlock_open"
