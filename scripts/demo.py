#!/usr/bin/env python3
"""
EDGE-DR — Quick Demo Script
==============================
Run inference on a sample image using the trained model
(no dataset required — uses a synthetic test image).

Usage:
    python scripts/demo.py
    python scripts/demo.py --image path/to/fundus.jpg
"""

import os
import sys
import numpy as np
import cv2
import torch
import click
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_synthetic_fundus(size: int = 512) -> np.ndarray:
    """
    Generate a synthetic orange fundus-like image for quick testing
    when no real fundus image is available.
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Orange background (retinal fundus colour)
    img[:, :] = [30, 100, 200]   # BGR: orange

    # Draw optic disc (bright yellow-white circle)
    cx, cy = int(size * 0.65), size // 2
    cv2.circle(img, (cx, cy), size // 10, (100, 220, 255), -1)
    cv2.circle(img, (cx, cy), size // 15, (200, 240, 255), -1)

    # Simulate blood vessels (dark branching lines)
    for angle in range(0, 360, 30):
        import math
        rad   = math.radians(angle)
        ex    = int(cx + size * 0.4 * math.cos(rad))
        ey    = int(cy + size * 0.4 * math.sin(rad))
        cv2.line(img, (cx, cy), (ex, ey), (10, 60, 120), 2)

    # Simulate microaneurysm dots (DR lesions)
    rng = np.random.default_rng(42)
    for _ in range(15):
        x  = rng.integers(size // 4, 3 * size // 4)
        y  = rng.integers(size // 4, 3 * size // 4)
        cv2.circle(img, (x, y), rng.integers(2, 5), (10, 30, 180), -1)

    # Circular mask (simulate fundus boundary)
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), size // 2 - 10, 255, -1)
    img[mask == 0] = 0

    return img


@click.command()
@click.option("--image",      default=None,                  help="Path to fundus image")
@click.option("--config",     default="configs/config.yaml", help="Config YAML path")
@click.option("--checkpoint", default="checkpoints/best_model.pt")
@click.option("--output-dir", default="outputs/demo/")
@click.option("--use-tflite", is_flag=True, default=False,
              help="Use TFLite model instead of PyTorch")
@click.option("--tflite-path",default="edge_model/edge_dr.tflite")
def main(image, config, checkpoint, output_dir, use_tflite, tflite_path):
    import yaml
    from src.data.preprocess import preprocess_image

    with open(config) as f:
        cfg = yaml.safe_load(f)

    # Determine image source
    if image is None:
        print("No image provided — generating synthetic fundus for demo…")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        synth = create_synthetic_fundus(cfg["data"]["image_size"])
        image = str(Path(output_dir) / "synthetic_fundus.png")
        cv2.imwrite(image, synth)
        print(f"Synthetic image saved → {image}")
    else:
        if not Path(image).exists():
            print(f"ERROR: Image not found: {image}")
            sys.exit(1)

    # Check model availability
    model_path = tflite_path if use_tflite else checkpoint
    if not Path(model_path).exists():
        print(f"\nModel not found at: {model_path}")
        print("Run the full training pipeline first:")
        print("  1. python scripts/download_datasets.py --dataset aptos")
        print("  2. python -m src.data.preprocess")
        print("  3. python -m src.models.train")
        print("  4. python -m src.models.quantize  (for TFLite)")
        print("\nRunning preprocessing + XAI demo without trained model…\n")

        # Demo preprocessing only
        processed = preprocess_image(image, output_size=cfg["data"]["image_size"],
                                      skip_quality_gate=True)
        if processed is not None:
            out = str(Path(output_dir) / "preprocessed_demo.png")
            cv2.imwrite(out, processed)
            print(f"Preprocessed image saved → {out}")
            print("Preprocessing pipeline works correctly ✓")
        return

    # Full inference
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    from src.inference import EdgeDRInference
    engine = EdgeDRInference(model_path, config, use_tflite=use_tflite)
    report = engine.predict(image, save_heatmap=True, output_dir=output_dir)

    print(f"\n{'='*60}")
    print(f"  EDGE-DR Demo Result")
    print(f"{'='*60}")
    print(f"  Image        : {image}")
    print(f"  Grade        : {report.grade} - {report.grade_label}")
    print(f"  Confidence   : {report.confidence:.1f}%")
    print(f"  Action       : {report.triage_action}")
    print(f"  Urgency      : {report.urgency.upper()}")
    print(f"  Refer?       : {'YES [REFERRAL]' if report.refer else 'No'}")
    print(f"  Latency      : {report.inference_ms:.1f} ms")
    print(f"  Hindi        : {report.hindi_summary}")
    if report.heatmap_path:
        print(f"  Grad-CAM     : {report.heatmap_path}")
    print(f"{'='*60}\n")

    # Print lesion localization
    loc = report.localization
    print("Lesion Localization:")
    print(f"  Active quadrants : {loc.get('quadrants', [])}")
    print(f"  Peak activation  : {loc.get('peak_activation', 0):.3f}")
    print(f"  Lesion hints     : {loc.get('lesion_hints', [])}")


if __name__ == "__main__":
    main()

