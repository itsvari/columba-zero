"""Select pinned Python environments from CPU architecture and GPU allocation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Environment:
    architecture: str
    acceleration: str
    torch_version: str
    torchvision_version: str

    @property
    def directory_name(self) -> str:
        return f"{self.architecture}-{self.acceleration}"


def environment_for(architecture: str, has_gpu: bool) -> Environment:
    if architecture not in {"x86_64", "aarch64"}:
        raise ValueError(f"unsupported architecture: {architecture}")
    acceleration = "cu126" if has_gpu else "cpu"
    if architecture == "aarch64" and has_gpu:
        return Environment(architecture, acceleration, "2.9.0", "0.24.0")
    return Environment(architecture, acceleration, "2.8.0", "0.23.0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--gpu", action="store_true")
    arguments = parser.parse_args()
    environment = environment_for(arguments.architecture, arguments.gpu)
    print(environment.acceleration, environment.torch_version, environment.torchvision_version)


if __name__ == "__main__":
    main()
