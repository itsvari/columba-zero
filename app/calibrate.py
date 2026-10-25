"""Interactive, manually stepped calibration: python -m app.calibrate."""
from __future__ import annotations

import argparse
from dataclasses import replace

from .config import ConfigError
from .pantilt import PCA9685, AxisCalibration, Calibration


def calibrate(profile: Calibration, mount: PCA9685, path: str,
              read=input, write=print) -> bool:
    pulses = {"pan": 1500.0, "tilt": 1500.0}
    marks: dict[str, dict] = {"pan": {}, "tilt": {}}
    for name in pulses:
        axis = getattr(profile, name)
        if not axis.pulse_min_us <= 1500 <= axis.pulse_max_us:
            raise ConfigError("calibration pulse envelopes must contain the initial 1500 us")
    for name in pulses:
        mount.pulse(getattr(profile, name).channel, 1500)
    write("Both axes commanded to 1500 us. No automatic sweep.\n"
          "Commands: pan + | pan - | tilt + | tilt - (10 us per step);\n"
          "pan min|home|max (mark current position); pan direction 1|-1;\n"
          "use tilt for the other axis; status; save; quit.\n"
          "Direction 1 means increasing pulse moves toward a target right/down in the image.\n"
          "Verify each axis direction and mark both travel limits and home before saving.")
    while True:
        try:
            parts = read("calibrate> ").strip().lower().split()
        except EOFError:
            return False
        if not parts:
            continue
        if parts == ["quit"]:
            return False
        if parts == ["status"]:
            write(str({"pulses_us": pulses, "marks": marks}))
            continue
        if parts == ["save"]:
            if any(set(axis) != {"min", "home", "max", "direction"} for axis in marks.values()):
                write("Record min, home, max, and direction for both axes first.")
                continue
            result = profile
            for name, values in marks.items():
                axis = getattr(profile, name)
                scale = 180 / (axis.pulse_max_us - axis.pulse_min_us)
                result = replace(result, **{name: replace(
                    axis, min_angle=(values["min"] - axis.pulse_min_us) * scale,
                    home_angle=(values["home"] - axis.pulse_min_us) * scale,
                    max_angle=(values["max"] - axis.pulse_min_us) * scale,
                    direction=values["direction"])})
            try:
                result.save(path)
            except ConfigError as exc:
                write(str(exc))
                continue
            write(f"Saved {path}")
            return True
        if len(parts) not in (2, 3) or parts[0] not in pulses:
            write("Unknown command; use pan/tilt +, -, min, home, max, direction 1|-1.")
            continue
        name, command = parts[:2]
        axis = getattr(profile, name)
        if command in ("+", "-") and len(parts) == 2:
            pulse = pulses[name] + (10 if command == "+" else -10)
            if not axis.pulse_min_us <= pulse <= axis.pulse_max_us:
                write("Pulse envelope reached; movement refused.")
                continue
            mount.pulse(axis.channel, pulse)
            pulses[name] = pulse
            write(f"{name}: {pulse:g} us")
        elif command in ("min", "home", "max") and len(parts) == 2:
            marks[name][command] = pulses[name]
            write(f"Marked {name} {command}: {pulses[name]:g} us")
        elif command == "direction" and len(parts) == 3 and parts[2] in ("1", "-1"):
            marks[name]["direction"] = int(parts[2])
        else:
            write("Unknown command.")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Manually calibrate the Waveshare camera mount")
    parser.add_argument("--output", default="calibration.local.yaml")
    parser.add_argument("--bus", type=int, default=1)
    parser.add_argument("--address", type=lambda v: int(v, 0), default=0x40)
    for name, channel in (("pan", 1), ("tilt", 0)):
        parser.add_argument(f"--{name}-channel", type=int, default=channel)
        parser.add_argument(f"--{name}-pulse-min-us", type=float, default=1000)
        parser.add_argument(f"--{name}-pulse-max-us", type=float, default=2000)
    args = parser.parse_args()
    mount = None
    try:
        axes = {name: AxisCalibration(
            channel=getattr(args, f"{name}_channel"),
            pulse_min_us=getattr(args, f"{name}_pulse_min_us"),
            pulse_max_us=getattr(args, f"{name}_pulse_max_us")) for name in ("pan", "tilt")}
        profile = Calibration(**axes, bus=args.bus, address=args.address)
        profile.validate()
        if any(not axis.pulse_min_us <= 1500 <= axis.pulse_max_us for axis in axes.values()):
            raise ConfigError("pulse envelopes must contain 1500 us")
        print("Prepare for both servos to move to 1500 us. Follow the assembly guide first.")
        mount = PCA9685(profile)
        calibrate(profile, mount, args.output)
    except (ConfigError, OSError, RuntimeError) as exc:
        parser.exit(1, f"calibration failed: {exc}\n")
    except KeyboardInterrupt:
        print("\nCalibration cancelled.")
    finally:
        if mount is not None:
            mount.close()


if __name__ == "__main__":
    cli()
