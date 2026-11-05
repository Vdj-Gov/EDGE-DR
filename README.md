# EDGE-DR: Explainable AI for Diabetic Retinopathy Screening
> **100% offline · Software-only · Edge AI · Multilingual**

## Project Structure
```
EDGEAIproj/
├── configs/
│   └── config.yaml              # Central config (data paths, model, training)
├── src/
│   ├── data/
│   │   ├── preprocess.py        # CLAHE, blur detection, circle crop, Ben Graham norm
│   │   └── dataset.py           # PyTorch Dataset, albumentations augmentation pipeline
│   ├── models/
│   │   ├── model.py             # EfficientNet-B0 / MobileNetV3, Focal Loss, QWK metric
│   │   ├── train.py             # Full training loop (AMP, cosine LR, checkpoint)
│   │   └── quantize.py          # PyTorch → ONNX → TFLite INT8 (<12 MB)
│   ├── xai/
│   │   └── gradcam.py           # Grad-CAM explainability engine + lesion localization
│   └── inference.py             # End-to-end inference + multilingual triage report
├── scripts/
│   ├── download_datasets.py     # Kaggle API downloader (APTOS / EyePACS / IDRiD)
│   └── demo.py                  # Quick demo (synthetic fundus if no dataset yet)
├── edge_dr_app/                 # Flutter offline mobile app
│   ├── lib/
│   │   ├── main.dart            # App entry, routing, localisation
│   │   ├── models/
│   │   │   ├── triage_result.dart
│   │   │   └── app_state.dart
│   │   ├── services/
│   │   │   ├── inference_service.dart  # TFLite INT8 inference on-device
│   │   │   └── database_service.dart   # SQLite encrypted local storage
│   │   └── screens/
│   │       ├── home_screen.dart
│   │       ├── camera_screen.dart      # Capture + trigger inference
│   │       ├── result_screen.dart      # Triage card + Grad-CAM + PDF referral
│   │       ├── history_screen.dart
│   │       └── settings_screen.dart
│   └── pubspec.yaml
├── requirements.txt
└── README.md
```

## 5-Step Pipeline (Software-Only)

| Step | Component | File |
|------|-----------|------|
| 1 | Image preprocessing (CLAHE, blur gate, circle crop) | `src/data/preprocess.py` |
| 2 | Dataset + augmentation pipeline | `src/data/dataset.py` |
| 3 | EfficientNet-B0 training with Focal Loss | `src/models/train.py` |
| 4 | Grad-CAM XAI explainability | `src/xai/gradcam.py` |
| 5 | INT8 TFLite quantization + edge deployment | `src/models/quantize.py` |

## Quickstart

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Download datasets (requires Kaggle API key)
```bash
python scripts/download_datasets.py --dataset aptos
# Optional: --dataset eyepacs (large, ~90 GB)
```

### 3. Preprocess images
```bash
python -m src.data.preprocess --config configs/config.yaml --dataset aptos
```

### 4. Train model
```bash
python -m src.models.train --config configs/config.yaml
```

### 5. Convert to TFLite INT8
```bash
python -m src.models.quantize --config configs/config.yaml
```

### 6. Run inference / demo
```bash
# With trained model:
python scripts/demo.py --image path/to/fundus.jpg

# Quick demo (synthetic image, no model needed):
python scripts/demo.py
```

### 7. Build Flutter mobile app
```bash
cd edge_dr_app
flutter pub get
flutter run          # Android device / emulator
flutter build apk    # Release APK
```

## Target Performance Metrics

| Metric | Target | Source |
|--------|--------|--------|
| Sensitivity | > 92.0% | WHO standard |
| Specificity | > 88.0% | ATA standard |
| Quadratic Weighted Kappa | > 0.85 | DR grading consensus |
| On-device latency | < 1,200 ms | MediaTek / Snapdragon |
| TFLite model size | < 12 MB | Offline storage budget |

## Languages Supported
Hindi · Tamil · Telugu · Marathi · English

## Datasets
- **APTOS 2019** — 3,662 clinician-annotated fundus images (Kaggle)
- **EyePACS** — 35,126 labeled images (Kaggle)
- **IDRiD** — 516 pixel-level annotated images for XAI verification

## Hardware Note
This is a **software-only** implementation. The 3D-printed optical adapter described
in the original project proposal has been excluded. Fundus images can be:
- Captured using existing smartphone cameras
- Loaded from gallery (clinical photos, telemedicine datasets)
- Used directly from APTOS/EyePACS dataset for validation

