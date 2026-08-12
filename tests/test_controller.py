from __future__ import annotations

from pathlib import Path

from conftest import make_config

from app.actuators import DeterrentActuator, MemoryOutput
from app.controller import RuntimeController
from app.geometry import TargetGeometry
from app.logger import EventLogger
from app.models import Detection, FramePacket, RuntimeState
from app.safety import SafetyGate
from app.turret import SimulationTurret


def build_controller(tmp_path: Path, now: list[float]) -> tuple[RuntimeController, MemoryOutput]:
    config = make_config(logging={"path": str(tmp_path / "events.jsonl")})
    safety = SafetyGate(
        config.safety,
        config.model,
        config.camera.stale_after_s,
        clock=lambda: now[0],
    )
    turret = SimulationTurret(config.turret, clock=lambda: now[0], sleeper=lambda _: None)
    output = MemoryOutput()
    actuator = DeterrentActuator(config.actuator, output, MemoryOutput(), sleeper=lambda _: None)
    controller = RuntimeController(
        config,
        safety,
        TargetGeometry(config.geometry),
        turret,
        actuator,
        EventLogger(tmp_path / "events.jsonl"),
        clock=lambda: now[0],
        sleeper=lambda _: None,
    )
    assert controller.arm()
    return controller, output


def pigeon() -> Detection:
    return Detection("pigeon", 0.9, 90, 90, 110, 110)


def packet(sequence: int, now: float) -> FramePacket:
    return FramePacket(sequence, now, None)


def test_single_frame_cannot_activate(tmp_path: Path) -> None:
    now = [0.0]
    controller, output = build_controller(tmp_path, now)
    assert controller.process(packet(0, now[0]), [pigeon()]) == RuntimeState.VERIFYING
    assert True not in output.transitions


def test_confirmation_aims_then_requires_fresh_frame_to_activate(tmp_path: Path) -> None:
    now = [0.0]
    controller, output = build_controller(tmp_path, now)
    for sequence in range(2):
        controller.process(packet(sequence, now[0]), [pigeon()])
    assert controller.process(packet(2, now[0]), [pigeon()]) == RuntimeState.AIMING
    assert True not in output.transitions
    now[0] = 0.01
    assert controller.process(packet(3, now[0]), [pigeon()]) == RuntimeState.COOLDOWN
    assert True in output.transitions
    assert not output.value


def test_person_rejects_activation_and_stale_frame_fails_safe(tmp_path: Path) -> None:
    now = [0.0]
    controller, output = build_controller(tmp_path, now)
    controller.process(packet(0, 0), [pigeon()])
    person = Detection("person", 0.9, 0, 0, 50, 100)
    assert controller.process(packet(1, 0), [pigeon(), person]) == RuntimeState.IDLE
    assert True not in output.transitions
    now[0] = 1.0
    assert controller.process(packet(2, 0), []) == RuntimeState.SAFE
    assert not output.value
