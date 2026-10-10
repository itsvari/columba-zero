"""Compare held-out accuracy and CPU validation timings at matched input settings."""
import argparse
import json
import platform
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_yaml')
    parser.add_argument('--models', default='models/pigeon_baseline_v1-2')
    parser.add_argument('--output', default='outputs/pc_benchmark')
    args = parser.parse_args()
    run = Path(args.models)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {'platform': platform.platform(), 'imgsz': 640, 'device': 'cpu',
              'batch': 1, 'conf': 0.001, 'iou': 0.7, 'rect': False, 'results': {}}
    for label, model_path in [('pytorch', run / 'best.pt'),
                              ('ncnn', run / 'best_ncnn_model')]:
        metrics = YOLO(str(model_path), task='detect').val(
            data=args.data_yaml,
            split='test', imgsz=640, device='cpu', batch=1, workers=0,
            conf=0.001, iou=0.7, rect=False, plots=True,
            project=str(output), name=label, verbose=False)
        speed = {key: float(value) for key, value in metrics.speed.items()}
        report['results'][label] = {
            'precision': float(metrics.box.mp), 'recall': float(metrics.box.mr),
            'map50': float(metrics.box.map50), 'map50_95': float(metrics.box.map),
            'speed_ms_per_image': speed,
            'inference_fps': 1000 / speed['inference'],
            'pipeline_fps': 1000 / sum(speed[key] for key in
                                       ('preprocess', 'inference', 'postprocess'))}
        (output / 'results.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report['results'][label], indent=2))


if __name__ == '__main__':
    main()
