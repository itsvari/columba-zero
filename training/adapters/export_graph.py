"""Export a fixed-shape packed head and invoke the pinned PNNX converter."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import torch

THREAD_COUNT = 4
COLOUR_CHANNELS = 3
PNNX_VERSION = "20260526"


class Packed(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, dictionary_output: bool) -> None:
        super().__init__()
        self.model = model
        self.dictionary_output = dictionary_output

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = self.model(image)
        if self.dictionary_output:
            predictions = cast(dict[str, torch.Tensor], output)
            return torch.cat(
                (predictions["pred_boxes"], predictions["pred_logits"]), dim=-1
            ).squeeze(0)
        return cast(torch.Tensor, output).squeeze(0)


def arguments(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-hw", type=int, nargs=2, default=(384, 640))
    result = parser.parse_args()
    if any(size <= 0 or size % 32 for size in result.input_hw):
        parser.error("input dimensions must be positive multiples of 32")
    return result


def export(model: torch.nn.Module, output: Path, input_hw: tuple[int, int]) -> None:
    torch.set_num_threads(THREAD_COUNT)
    output.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, COLOUR_CHANNELS, *input_hw)
    model = model.eval()
    with torch.no_grad():
        prediction = cast(torch.Tensor, model(example))
        if prediction.ndim != 2:
            raise ValueError("packed head must return exactly one 2D tensor")
        traced = torch.jit.trace(model, example, check_trace=False)  # type: ignore[no-untyped-call]
        traced.save(str(output / "model.pt"))
    import importlib.metadata

    if importlib.metadata.version("pnnx") != PNNX_VERSION:
        raise ValueError(f"requires pnnx=={PNNX_VERSION}")
    converter = Path(sys.executable).parent / "pnnx"
    subprocess.run(
        [
            str(converter),
            str(output / "model.pt"),
            f"inputshape=[1,3,{input_hw[0]},{input_hw[1]}]",
            "fp16=0",
            f"ncnnparam={output / 'model.ncnn.param'}",
            f"ncnnbin={output / 'model.ncnn.bin'}",
        ],
        check=True,
    )
    (output / "metadata.yaml").write_text(f"imgsz: [{input_hw[0]}, {input_hw[1]}]\n")
    (output / "export.json").write_text(
        json.dumps(
            {
                "input_hw": input_hw,
                "output_shape": list(prediction.shape),
                "torch": torch.__version__,
                "pnnx": PNNX_VERSION,
                "weight_storage": "fp32",
                "correctness": "unverified; run verify_challenger",
            },
            indent=2,
        )
        + "\n"
    )
