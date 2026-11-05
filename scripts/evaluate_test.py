"""
EDGE-DR — Test Dataset Evaluator & Batch Predictor
===================================================
1. Evaluates on the held-out labeled test split (333 images)
   and outputs a full classification report + confusion matrix.
2. Supports batch prediction on unlabelled test images (e.g. data/aptos2019/test_images).

Usage:
    python scripts/evaluate_test.py --mode labeled
    python scripts/evaluate_test.py --mode unlabelled --input-dir data/aptos2019/test_images
"""

import os
import sys
import click
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

# Add root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.model import EdgeDRModel, compute_sensitivity_specificity
from src.data.dataset import get_dataloaders, DR_CLASSES, get_transforms
from src.data.preprocess import preprocess_image


def evaluate_labeled_test(checkpoint_path: str = "checkpoints/best_model.pt",
                          manifest_path: str = "data/processed/dataset.csv",
                          export_csv: str = "data/processed/test_split.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {checkpoint_path} on {device}")

    # Build dataloaders
    loaders = get_dataloaders(
        csv_path=manifest_path,
        batch_size=32,
        val_split=0.15,
        test_split=0.10,
        num_workers=0,
        seed=42
    )

    # Save test dataset split explicitly for user inspection
    test_ds = loaders["test"].dataset
    test_df = test_ds.df
    test_df.to_csv(export_csv, index=False)
    print(f"Saved labeled test split ({len(test_df)} samples) -> {export_csv}")

    # Load model
    model = EdgeDRModel(pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loaders["test"], desc="Evaluating test set"):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_targets.extend(labels.numpy())
            all_probs.extend(probs)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    sens_spec = compute_sensitivity_specificity(y_pred, y_true)

    print("\n" + "="*60)
    print("  EDGE-DR TEST SET EVALUATION BENCHMARKS")
    print("="*60)
    print(f"  Test Samples Evaluated: {len(y_true)}")
    print(f"  Quadratic Weighted Kappa (QWK): {qwk:.4f}  (Target: >0.85)")
    print(f"  Clinical Sensitivity (Recall) : {sens_spec['sensitivity']*100:.2f}%  (Target: >92.0%)")
    print(f"  Clinical Specificity          : {sens_spec['specificity']*100:.2f}%  (Target: >88.0%)")
    print("="*60)

    print("\nPer-Class Classification Report:")
    target_names = [f"Grade {k} ({v})" for k, v in DR_CLASSES.items()]
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    print("Confusion Matrix (Rows: Ground Truth, Cols: Predicted):")
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=[f"Actual G{i}" for i in range(5)], columns=[f"Pred G{i}" for i in range(5)])
    print(cm_df)


def predict_unlabelled(input_dir: str,
                       checkpoint_path: str = "checkpoints/best_model.pt",
                       output_csv: str = "outputs/unlabelled_predictions.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EdgeDRModel(pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    transform = get_transforms("test", 224)
    images = list(Path(input_dir).glob("*.png")) + list(Path(input_dir).glob("*.jpg"))
    print(f"Found {len(images)} images in {input_dir}")

    records = []
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    for img_p in tqdm(images[:100], desc="Predicting (sample 100)"):
        processed = preprocess_image(str(img_p), output_size=224, skip_quality_gate=True)
        if processed is None:
            continue
        import cv2
        img_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))

        records.append({
            "image": img_p.name,
            "predicted_grade": pred,
            "grade_label": DR_CLASSES[pred],
            "confidence": round(float(probs[pred]) * 100, 1),
            "refer": pred in [2, 3, 4]
        })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\nPredictions saved -> {output_csv}")
    print("\nPredicted Class Distribution:")
    print(df["grade_label"].value_counts())


@click.command()
@click.option("--mode", default="labeled", type=click.Choice(["labeled", "unlabelled"]))
@click.option("--checkpoint", default="checkpoints/best_model.pt")
@click.option("--input-dir", default="data/aptos2019/test_images")
def main(mode, checkpoint, input_dir):
    if mode == "labeled":
        evaluate_labeled_test(checkpoint)
    else:
        predict_unlabelled(input_dir, checkpoint)


if __name__ == "__main__":
    main()

