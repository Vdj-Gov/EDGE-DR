"""
EDGE-DR — Data Preprocessing Pipeline
=======================================
Implements the full on-device-style preprocessing chain:
  1. Blur / quality rejection  (Laplacian variance)
  2. CLAHE contrast enhancement
  3. Illumination normalisation (Ben Graham method)
  4. Circular ROI crop  → 512 × 512
  5. Augmentation pipeline for training

Usage:
    python -m src.data.preprocess --config configs/config.yaml
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import yaml
import click


# ──────────────────────────────────────────────
# 1.  QUALITY GATE — Laplacian blur detection
# ──────────────────────────────────────────────

def laplacian_variance(img_gray: np.ndarray) -> float:
    """Return the Laplacian variance of a grayscale image."""
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())


def is_quality_ok(img: np.ndarray, blur_threshold: float = 30.0) -> bool:
    """Return True if image passes basic quality gate."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Standardize scale to 512x512 for consistent resolution-independent variance
    gray_std = cv2.resize(gray, (512, 512))
    
    # Blur check on standardized image
    if laplacian_variance(gray_std) < blur_threshold:
        return False
        
    # Exposure check on non-black retina pixels
    non_zero = gray_std[gray_std > 15]
    if len(non_zero) == 0:
        return False
    mean_intensity = float(non_zero.mean())
    if mean_intensity < 20 or mean_intensity > 240:
        return False
    return True


# ──────────────────────────────────────────────
# 2.  BEN GRAHAM NORMALISATION
# ──────────────────────────────────────────────

def ben_graham_normalize(img: np.ndarray, sigmaX: int = 10) -> np.ndarray:
    """Subtract local average colour to remove illumination gradient
    (Ben Graham's well-known Kaggle pre-processing technique).
    """
    img = cv2.addWeighted(
        img, 4,
        cv2.GaussianBlur(img, (0, 0), sigmaX), -4,
        128
    )
    return img


# ──────────────────────────────────────────────
# 3.  CLAHE enhancement
# ──────────────────────────────────────────────

def apply_clahe(img: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid: tuple = (8, 8)) -> np.ndarray:
    """Apply CLAHE per channel in LAB colour space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l)
    merged = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ──────────────────────────────────────────────
# 4.  CIRCULAR CROP — isolate fundus disc
# ──────────────────────────────────────────────

def crop_circle(img: np.ndarray,
                output_size: int = 512,
                tol: float = 0.02) -> np.ndarray:
    """Detect fundus circle boundary, mask background, square-crop, resize."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    # Find bounding box of non-black area
    coords = cv2.findNonZero(mask)
    if coords is None:
        # Fallback: return centre-square crop
        h, w = img.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        cropped = img[y0:y0+s, x0:x0+s]
        return cv2.resize(cropped, (output_size, output_size))

    x, y, bw, bh = cv2.boundingRect(coords)
    # Add tolerance
    tol_px = int(tol * min(bw, bh))
    x = max(0, x - tol_px)
    y = max(0, y - tol_px)
    bw = min(img.shape[1] - x, bw + 2 * tol_px)
    bh = min(img.shape[0] - y, bh + 2 * tol_px)

    # Square-ify around centre
    cx, cy = x + bw // 2, y + bh // 2
    s = max(bw, bh)
    x0 = max(0, cx - s // 2)
    y0 = max(0, cy - s // 2)
    x1 = min(img.shape[1], x0 + s)
    y1 = min(img.shape[0], y0 + s)
    cropped = img[y0:y1, x0:x1]

    # Apply circular mask
    h, w = cropped.shape[:2]
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, (w // 2, h // 2), min(w, h) // 2, 255, -1)
    result = cropped.copy()
    result[circle_mask == 0] = 0

    return cv2.resize(result, (output_size, output_size))


# ──────────────────────────────────────────────
# 5.  MASTER PREPROCESSING FUNCTION
# ──────────────────────────────────────────────

def preprocess_image(img_path: str,
                     output_size: int = 512,
                     blur_threshold: float = 50.0,
                     skip_quality_gate: bool = False) -> np.ndarray | None:
    """
    Full preprocessing pipeline for a single fundus image.
    Returns processed np.ndarray (BGR, uint8) or None if rejected.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    # Quality gate
    if not skip_quality_gate and not is_quality_ok(img, blur_threshold):
        return None

    # Circular crop first (removes irrelevant border)
    img = crop_circle(img, output_size=output_size)

    # Ben Graham normalisation
    img = ben_graham_normalize(img, sigmaX=output_size // 50)

    # CLAHE
    img = apply_clahe(img)

    return img


# ──────────────────────────────────────────────
# 6.  DATASET BUILDER
# ──────────────────────────────────────────────

def build_aptos_dataset(aptos_dir: str,
                        output_dir: str,
                        image_size: int = 512,
                        blur_threshold: float = 50.0) -> pd.DataFrame:
    """
    Process the APTOS 2019 dataset.
    Expected structure:
        aptos_dir/
            train.csv       (id_code, diagnosis)
            train_images/   (*.png)
    """
    aptos_dir = Path(aptos_dir)
    output_dir = Path(output_dir) / "aptos"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(aptos_dir / "train.csv")
    records = []
    rejected = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="APTOS preprocessing"):
        src = aptos_dir / "train_images" / f"{row['id_code']}.png"
        dst = output_dir / f"{row['id_code']}.png"

        if dst.exists():
            records.append({"image_path": str(dst), "label": row["diagnosis"], "source": "aptos"})
            continue

        processed = preprocess_image(str(src), output_size=image_size, blur_threshold=blur_threshold)
        if processed is None:
            rejected += 1
            continue

        cv2.imwrite(str(dst), processed)
        records.append({"image_path": str(dst), "label": row["diagnosis"], "source": "aptos"})

    print(f"APTOS: {len(records)} accepted, {rejected} rejected (blur/quality)")
    return pd.DataFrame(records)


def build_idrid_dataset(idrid_dir: str,
                         output_dir: str,
                         image_size: int = 512,
                         blur_threshold: float = 50.0,
                         skip_quality_gate: bool = False) -> pd.DataFrame:
    """
    Process the IDRiD dataset.
    Expected structure:
        idrid_dir/
            idrid_labels.csv    (id_code, diagnosis)
            Imagenes/Imagenes/  (*.jpg)
    """
    idrid_dir = Path(idrid_dir)
    output_dir = Path(output_dir) / "idrid"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_candidates = list(idrid_dir.glob("*.csv"))
    if not csv_candidates:
        print(f"No CSV labels found in {idrid_dir}")
        return pd.DataFrame()

    df = pd.read_csv(csv_candidates[0])
    img_dirs = [p for p in idrid_dir.rglob("*") if p.is_dir() and any(p.glob("*.jpg"))]
    if not img_dirs:
        print(f"No image directory found in {idrid_dir}")
        return pd.DataFrame()
    img_dir = img_dirs[0]

    records = []
    rejected = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="IDRiD preprocessing"):
        id_code = str(row["id_code"]).strip()
        label = int(row["diagnosis"])

        src = None
        for ext in [".jpg", ".jpeg", ".png"]:
            cand = img_dir / f"{id_code}{ext}"
            if cand.exists():
                src = cand
                break

        if src is None:
            continue

        dst = output_dir / f"{id_code}.png"
        if dst.exists():
            records.append({"image_path": str(dst), "label": label, "source": "idrid"})
            continue

        processed = preprocess_image(
            str(src),
            output_size=image_size,
            blur_threshold=blur_threshold,
            skip_quality_gate=skip_quality_gate
        )
        if processed is None:
            rejected += 1
            continue

        cv2.imwrite(str(dst), processed)
        records.append({"image_path": str(dst), "label": label, "source": "idrid"})

    print(f"IDRiD: {len(records)} accepted, {rejected} rejected (blur/quality)")
    return pd.DataFrame(records)


def build_eyepacs_dataset(eyepacs_dir: str,
                          output_dir: str,
                          image_size: int = 512,
                          sample_per_class: int = 7000,
                          blur_threshold: float = 50.0) -> pd.DataFrame:
    """
    Process the EyePACS dataset.
    Expected structure:
        eyepacs_dir/
            trainLabels.csv     (image, level)
            train/              (*.jpeg)
    """
    eyepacs_dir = Path(eyepacs_dir)
    output_dir = Path(output_dir) / "eyepacs"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(eyepacs_dir / "trainLabels.csv")

    # Balanced sampling to avoid extreme class imbalance
    df_sampled = (df.groupby("level", group_keys=False)
                    .apply(lambda x: x.sample(min(len(x), sample_per_class), random_state=42)))

    records = []
    rejected = 0

    for _, row in tqdm(df_sampled.iterrows(), total=len(df_sampled), desc="EyePACS preprocessing"):
        img_name = row["image"]
        src = eyepacs_dir / "train" / f"{img_name}.jpeg"
        if not src.exists():
            src = eyepacs_dir / "train" / f"{img_name}.jpg"
        dst = output_dir / f"{img_name}.png"

        if dst.exists():
            records.append({"image_path": str(dst), "label": row["level"], "source": "eyepacs"})
            continue

        processed = preprocess_image(str(src), output_size=image_size, blur_threshold=blur_threshold)
        if processed is None:
            rejected += 1
            continue

        cv2.imwrite(str(dst), processed)
        records.append({"image_path": str(dst), "label": row["level"], "source": "eyepacs"})

    print(f"EyePACS: {len(records)} accepted, {rejected} rejected")
    return pd.DataFrame(records)


# ──────────────────────────────────────────────
# 7.  CLI ENTRY POINT
# ──────────────────────────────────────────────

@click.command()
@click.option("--config", default="configs/config.yaml", help="Path to config YAML")
@click.option("--dataset", default="all", type=click.Choice(["aptos", "idrid", "eyepacs", "all"]))
@click.option("--skip-quality-gate", is_flag=True, default=False)
def main(config, dataset, skip_quality_gate):
    with open(config) as f:
        cfg = yaml.safe_load(f)

    dfs = []

    if dataset in ("idrid", "all") and os.path.exists(cfg["data"]["idrid_dir"]):
        df = build_idrid_dataset(
            cfg["data"]["idrid_dir"],
            cfg["data"]["processed_dir"],
            image_size=cfg["data"]["image_size"],
            skip_quality_gate=skip_quality_gate,
        )
        dfs.append(df)

    if dataset in ("aptos", "all") and os.path.exists(cfg["data"]["aptos_dir"]):
        df = build_aptos_dataset(
            cfg["data"]["aptos_dir"],
            cfg["data"]["processed_dir"],
            image_size=cfg["data"]["image_size"],
        )
        dfs.append(df)

    if dataset in ("eyepacs", "all") and os.path.exists(cfg["data"]["eyepacs_dir"]):
        df = build_eyepacs_dataset(
            cfg["data"]["eyepacs_dir"],
            cfg["data"]["processed_dir"],
            image_size=cfg["data"]["image_size"],
        )
        dfs.append(df)

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        out_csv = Path(cfg["data"]["processed_dir"]) / "dataset.csv"
        combined.to_csv(out_csv, index=False)
        print(f"\nSaved combined manifest -> {out_csv}")
        print(combined["label"].value_counts().sort_index())
    else:
        print("No datasets found. Download IDRiD / EyePACS / APTOS and update configs/config.yaml paths.")


if __name__ == "__main__":
    main()

