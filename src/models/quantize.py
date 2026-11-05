"""
EDGE-DR - TFLite INT8 Quantization Pipeline
=============================================
Converts trained PyTorch model -> ONNX -> TFLite INT8
Target: <12 MB footprint, <1200 ms latency on MediaTek/Snapdragon

Supports:
  - Post-Training Quantization (PTQ)  - fast, slight accuracy drop
  - Quantization-Aware Training (QAT) - better accuracy (run from train.py)

Usage:
    python -m src.models.quantize --config configs/config.yaml
"""

import os
import yaml
import click
import numpy as np
import torch
from pathlib import Path


# ──────────────────────────────────────────────
# Step 1: PyTorch -> ONNX
# ──────────────────────────────────────────────

def export_to_onnx(model: torch.nn.Module,
                   onnx_path: str,
                   input_size: int = 224) -> str:
    """Export PyTorch model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, 3, input_size, input_size)
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"ONNX exported -> {onnx_path}  ({size_mb:.2f} MB)")
    return onnx_path


# ──────────────────────────────────────────────
# Step 2: ONNX -> TFLite INT8 via tf2onnx + TF
# ──────────────────────────────────────────────

def build_representative_dataset(csv_path: str,
                                   n_samples: int = 500,
                                   image_size: int = 224):
    """
    Generator for TFLite PTQ calibration.
    Yields preprocessed float32 numpy arrays from the held-out dataset.
    """
    import pandas as pd
    import cv2
    from src.data.dataset import get_transforms

    df = pd.read_csv(csv_path).sample(n=min(n_samples, len(pd.read_csv(csv_path))),
                                       random_state=42)
    transform = get_transforms("val", image_size)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def generator():
        for _, row in df.iterrows():
            img = cv2.imread(str(row["image_path"]))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transform(image=img)["image"]                     # tensor CHW
            img = img.numpy().transpose(1, 2, 0)                    # HWC
            img = img.astype(np.float32)
            yield [np.expand_dims(img, axis=0)]                     # [1, H, W, 3]

    return generator


def convert_onnx_to_tflite_int8(onnx_path: str,
                                  tflite_path: str,
                                  representative_dataset_gen,
                                  input_size: int = 224) -> str:
    """
    Convert ONNX -> TensorFlow SavedModel -> TFLite INT8.
    Requires: tensorflow, tf2onnx (pip install tf2onnx tensorflow)
    """
    import subprocess, sys

    # Step 2a: ONNX -> TF SavedModel using tf2onnx CLI
    saved_model_dir = str(Path(tflite_path).parent / "saved_model")
    result = subprocess.run(
        [sys.executable, "-m", "tf2onnx.convert",
         "--onnx", onnx_path,
         "--output", saved_model_dir,
         "--target", "tensorflow"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"tf2onnx conversion failed:\n{result.stderr}")
    print(f"SavedModel written -> {saved_model_dir}")

    # Step 2b: TF SavedModel -> TFLite INT8
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    Path(tflite_path).parent.mkdir(parents=True, exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(tflite_path) / 1e6
    print(f"TFLite INT8 model -> {tflite_path}  ({size_mb:.2f} MB)")
    if size_mb > 12:
        print(f"  [!] Model exceeds 12 MB target ({size_mb:.1f} MB). "
              f"Consider pruning or using MobileNetV3 backbone.")
    else:
        print(f"  * Model within 12 MB budget.")
    return tflite_path


# ──────────────────────────────────────────────
# Step 3: Validate TFLite model on test set
# ──────────────────────────────────────────────

def benchmark_tflite(tflite_path: str,
                      csv_path: str,
                      image_size: int = 224,
                      n_samples: int = 200) -> dict:
    """
    Run TFLite interpreter on sample images and report:
    - Average latency
    - Accuracy vs PyTorch model
    """
    import tensorflow as tf
    import pandas as pd
    import cv2
    import time
    from src.data.dataset import get_transforms
    from src.models.model import quadratic_weighted_kappa, compute_sensitivity_specificity

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # INT8 scale & zero-point for de-quantization
    in_scale, in_zero_point   = (input_details[0]["quantization"]
                                  if input_details[0]["dtype"] == np.int8
                                  else (1.0, 0))
    out_scale, out_zero_point = (output_details[0]["quantization"]
                                  if output_details[0]["dtype"] == np.int8
                                  else (1.0, 0))

    df = pd.read_csv(csv_path).sample(n=min(n_samples, len(pd.read_csv(csv_path))),
                                       random_state=99)
    transform = get_transforms("val", image_size)
    latencies, preds, targets = [], [], []

    for _, row in df.iterrows():
        img = cv2.imread(str(row["image_path"]))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = transform(image=img)["image"].numpy().transpose(1, 2, 0)   # HWC float32

        # Quantize input
        img_q = (img_t / in_scale + in_zero_point).astype(np.int8)
        img_q = np.expand_dims(img_q, axis=0)

        interpreter.set_tensor(input_details[0]["index"], img_q)
        t0 = time.perf_counter()
        interpreter.invoke()
        latencies.append((time.perf_counter() - t0) * 1000)

        out = interpreter.get_tensor(output_details[0]["index"])
        logits = (out.astype(np.float32) - out_zero_point) * out_scale
        pred_class = int(np.argmax(logits))
        preds.append(pred_class)
        targets.append(int(row["label"]))

    preds_arr   = np.array(preds)
    targets_arr = np.array(targets)
    qwk         = quadratic_weighted_kappa(preds_arr, targets_arr)
    sens_spec   = compute_sensitivity_specificity(preds_arr, targets_arr)

    results = {
        "avg_latency_ms":  np.mean(latencies),
        "p95_latency_ms":  np.percentile(latencies, 95),
        "accuracy":        (preds_arr == targets_arr).mean(),
        "qwk":             qwk,
        "sensitivity":     sens_spec["sensitivity"],
        "specificity":     sens_spec["specificity"],
    }

    print("\nTFLite INT8 Benchmark Results:")
    print(f"  Avg latency : {results['avg_latency_ms']:.1f} ms  "
          f"(target <1200 ms)")
    print(f"  P95 latency : {results['p95_latency_ms']:.1f} ms")
    print(f"  QWK         : {results['qwk']:.4f}  (target >0.85)")
    print(f"  Sensitivity : {results['sensitivity']:.4f}  (target >0.92)")
    print(f"  Specificity : {results['specificity']:.4f}  (target >0.88)")

    return results


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

@click.command()
@click.option("--config", default="configs/config.yaml")
@click.option("--checkpoint", default=None,
              help="Path to .pt checkpoint (defaults to config checkpoint_dir/best_model.pt)")
@click.option("--benchmark", is_flag=True, default=True,
              help="Run TFLite latency + accuracy benchmark after conversion")
def main(config, checkpoint, benchmark):
    with open(config) as f:
        cfg = yaml.safe_load(f)

    from src.models.model import EdgeDRModel

    # Load checkpoint
    ckpt_path = checkpoint or (Path(cfg["training"]["checkpoint_dir"]) / "best_model.pt")
    device = torch.device("cpu")   # quantise on CPU for portability
    ckpt   = torch.load(str(ckpt_path), map_location=device)

    model = EdgeDRModel(
        backbone=cfg["model"]["backbone"],
        num_classes=cfg["data"]["num_classes"],
        dropout=0.0,                 # disable dropout for inference
        pretrained=False,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Export ONNX
    onnx_path   = cfg["quantization"]["onnx_output"]
    tflite_path = cfg["quantization"]["tflite_output"]
    export_to_onnx(model, onnx_path, cfg["data"]["model_input_size"])

    # Representative dataset
    csv_path = str(Path(cfg["data"]["processed_dir"]) / "dataset.csv")
    rep_gen  = build_representative_dataset(
        csv_path,
        n_samples=cfg["quantization"]["representative_samples"],
        image_size=cfg["data"]["model_input_size"],
    )

    # Convert to TFLite INT8
    convert_onnx_to_tflite_int8(onnx_path, tflite_path, rep_gen,
                                  cfg["data"]["model_input_size"])

    # Optional benchmark
    if benchmark:
        benchmark_tflite(tflite_path, csv_path,
                          cfg["data"]["model_input_size"])


if __name__ == "__main__":
    main()

