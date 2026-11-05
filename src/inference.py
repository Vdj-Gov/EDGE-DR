"""
EDGE-DR — Inference Engine
============================
Loads the TFLite INT8 model and Grad-CAM explainer,
runs end-to-end inference on a single fundus image,
and returns a structured clinical triage result.

This module is the bridge between the Python ML backend
and the Flutter mobile app (via Python-Flutter bridge or REST API).

Usage:
    python -m src.inference --image path/to/fundus.jpg --config configs/config.yaml
"""

import sys
import cv2
import numpy as np
import torch
import yaml
import click
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict

# Ensure project root in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.preprocess import preprocess_image
from src.xai.gradcam import GradCAMExplainer, GradCAMResult, save_explanation
from src.models.model import EdgeDRModel
from src.data.dataset import DR_CLASSES, REFERRAL_GRADES, get_transforms


# ──────────────────────────────────────────────
# Clinical triage report
# ──────────────────────────────────────────────

DR_GRADE_INFO = {
    0: {
        "label":   "No Diabetic Retinopathy",
        "action":  "MONITOR: Annual screening recommended",
        "urgency": "none",
        "hindi":   "कोई डायबिटिक रेटिनोपैथी नहीं मिली। वार्षिक जाँच जारी रखें।",
        "tamil":   "நீரிழிவு விழித்திரை நோய் இல்லை. ஆண்டு பரிசோதனை தொடரவும்.",
        "telugu":  "డయాబెటిక్ రెటినోపతి లేదు. వార్షిక స్క్రీనింగ్ కొనసాగించండి.",
    },
    1: {
        "label":   "Mild NPDR",
        "action":  "MONITOR: Recheck in 6–12 months",
        "urgency": "low",
        "hindi":   "हल्की डायबिटिक रेटिनोपैथी मिली है। 6-12 महीनों में पुनः जाँच कराएँ।",
        "tamil":   "மென்மையான நோய் கண்டறியப்பட்டது. 6-12 மாதங்களில் மறுபரிசோதனை செய்யவும்.",
        "telugu":  "తేలికపాటి వ్యాధి కనుగొనబడింది. 6-12 నెలల్లో తిరిగి తనిఖీ చేయించుకోండి.",
    },
    2: {
        "label":   "Moderate NPDR",
        "action":  "REFER: District Eye Hospital within 4–6 weeks",
        "urgency": "moderate",
        "hindi":   "आँख की जाँच में मध्यम डायबिटिक रेटिनोपैथी का पता चला है। कृपया नेत्र चिकित्सक से मिलें।",
        "tamil":   "மிதமான நோய் உள்ளது. 4-6 வாரங்களுக்குள் கண் மருத்துவரை அணுகவும்.",
        "telugu":  "మధ్యస్థ వ్యాధి ఉంది. 4-6 వారాల్లో కంటి వైద్యుడిని సందర్శించండి.",
    },
    3: {
        "label":   "Severe NPDR",
        "action":  "REFER URGENTLY: District Eye Hospital within 1–2 weeks",
        "urgency": "high",
        "hindi":   "गंभीर रेटिनोपैथी मिली है। तुरंत नेत्र चिकित्सक से मिलें।",
        "tamil":   "தீவிர நோய் உள்ளது. 1-2 வாரங்களுக்குள் கண் மருத்துவரை அணுகவும்.",
        "telugu":  "తీవ్రమైన వ్యాధి ఉంది. 1-2 వారాల్లో కంటి వైద్యుడిని సందర్శించండి.",
    },
    4: {
        "label":   "Proliferative DR (PDR)",
        "action":  "REFER IMMEDIATELY: Specialist within 48–72 hours — vision at risk",
        "urgency": "critical",
        "hindi":   "बहुत गंभीर रेटिनोपैथी मिली है। 48-72 घंटों में नेत्र विशेषज्ञ से मिलें।",
        "tamil":   "மிகவும் தீவிரமான நோய். 48-72 மணி நேரத்தில் நிபுணரை அணுகவும்.",
        "telugu":  "చాలా తీవ్రమైన వ్యాధి. 48-72 గంటల్లో నిపుణ వైద్యుడిని సందర్శించండి.",
    },
}


@dataclass
class TriageReport:
    """Structured triage output — serialisable to JSON for Flutter."""
    image_path:      str
    grade:           int
    grade_label:     str
    confidence:      float
    class_probs:     list[float]
    triage_action:   str
    urgency:         str
    refer:           bool
    localization:    dict
    inference_ms:    float
    hindi_summary:   str
    tamil_summary:   str
    telugu_summary:  str
    heatmap_path:    str | None = None   # path to saved overlay image

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


# ──────────────────────────────────────────────
# Inference engine class
# ──────────────────────────────────────────────

class EdgeDRInference:
    """
    End-to-end inference pipeline.

    For demo / development: uses PyTorch model + Grad-CAM.
    For edge deployment  : switch to TFLite runtime (tflite_path).

    Args:
        model_path  : Path to .pt checkpoint OR .tflite model
        config_path : Path to configs/config.yaml
        use_tflite  : Use TFLite runtime (True for production deployment)
    """

    def __init__(self, model_path: str, config_path: str, use_tflite: bool = False):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.use_tflite   = use_tflite
        self.image_size   = self.cfg["data"]["model_input_size"]
        self.transform    = get_transforms("val", self.image_size)
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if use_tflite:
            self._load_tflite(model_path)
        else:
            self._load_pytorch(model_path)

    def _load_pytorch(self, checkpoint_path: str):
        ckpt  = torch.load(checkpoint_path, map_location=self.device)
        model = EdgeDRModel(
            backbone=self.cfg["model"]["backbone"],
            num_classes=self.cfg["data"]["num_classes"],
            dropout=0.0,
            pretrained=False,
        ).to(self.device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        self.model    = model
        self.explainer = GradCAMExplainer(model, device=self.device)

    def _load_tflite(self, tflite_path: str):
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        self._in_details  = self.interpreter.get_input_details()
        self._out_details = self.interpreter.get_output_details()

    def predict(self,
                image_path: str,
                save_heatmap: bool = True,
                output_dir: str = "outputs/") -> TriageReport:
        """
        Full pipeline: preprocess → infer → Grad-CAM → triage report.

        Args:
            image_path  : path to raw fundus image (JPEG / PNG)
            save_heatmap: whether to save Grad-CAM overlay to disk
            output_dir  : directory for heatmap output

        Returns:
            TriageReport (JSON-serialisable)
        """
        # 1. Preprocess
        processed = preprocess_image(image_path,
                                      output_size=self.image_size,
                                      skip_quality_gate=True)
        if processed is None:
            raise ValueError(f"Could not load / process image: {image_path}")

        img_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        img_t   = self.transform(image=img_rgb)["image"].unsqueeze(0)

        t0 = time.perf_counter()

        if self.use_tflite:
            result = self._run_tflite(img_rgb, img_t)
        else:
            result = self.explainer.explain(img_t)

        inference_ms = (time.perf_counter() - t0) * 1000

        # 2. Build report
        grade_info  = DR_GRADE_INFO[result.pred_class]
        heatmap_path = None

        if save_heatmap and not self.use_tflite:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            stem = Path(image_path).stem
            heatmap_path = str(Path(output_dir) / f"{stem}_gradcam.jpg")
            save_explanation(result, heatmap_path)

        return TriageReport(
            image_path     = str(image_path),
            grade          = result.pred_class,
            grade_label    = grade_info["label"],
            confidence     = round(result.confidence * 100, 1),
            class_probs    = [round(float(p) * 100, 2) for p in result.class_probs],
            triage_action  = grade_info["action"],
            urgency        = grade_info["urgency"],
            refer          = result.pred_class in REFERRAL_GRADES,
            localization   = result.localization,
            inference_ms   = round(inference_ms, 1),
            hindi_summary  = grade_info["hindi"],
            tamil_summary  = grade_info["tamil"],
            telugu_summary = grade_info["telugu"],
            heatmap_path   = heatmap_path,
        )

    def _run_tflite(self, img_rgb: np.ndarray,
                     img_t: torch.Tensor) -> "GradCAMResult":
        """TFLite-only path (no Grad-CAM — heatmap skipped on device for speed)."""
        from src.xai.gradcam import GradCAMResult, _localize_lesions

        # Prepare INT8 input
        in_det = self._in_details[0]
        scale, zero_pt = in_det["quantization"]
        img_np = img_rgb.astype(np.float32) / 255.0
        img_np = np.expand_dims(img_np, 0)
        img_q  = (img_np / scale + zero_pt).astype(np.int8)
        self.interpreter.set_tensor(in_det["index"], img_q)
        self.interpreter.invoke()

        out_det = self._out_details[0]
        out_q   = self.interpreter.get_tensor(out_det["index"])
        out_s, out_zp = out_det["quantization"]
        logits = (out_q.astype(np.float32) - out_zp) * out_s
        probs  = np.exp(logits) / np.exp(logits).sum()
        probs  = probs.flatten()

        # Dummy heatmap (uniform) — real Grad-CAM runs in PyTorch on server
        h, w = img_rgb.shape[:2]
        heatmap = np.full((h, w), 0.5, dtype=np.float32)
        orig_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        return GradCAMResult(
            heatmap        = heatmap,
            overlay_bgr    = orig_bgr,
            original_bgr   = orig_bgr,
            pred_class     = int(np.argmax(probs)),
            class_probs    = probs,
            confidence     = float(probs.max()),
            explained_class= int(np.argmax(probs)),
            localization   = _localize_lesions(heatmap),
        )


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

@click.command()
@click.option("--image", required=True, help="Path to fundus image")
@click.option("--config", default="configs/config.yaml")
@click.option("--checkpoint", default="checkpoints/best_model.pt")
@click.option("--tflite", default=None, help="Path to TFLite model (optional)")
@click.option("--output-dir", default="outputs/")
@click.option("--json-output", is_flag=True, default=False,
              help="Print result as JSON (for integration with app)")
def main(image, config, checkpoint, tflite, output_dir, json_output):
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    use_tflite = tflite is not None
    model_path = tflite if use_tflite else checkpoint

    engine = EdgeDRInference(model_path, config, use_tflite=use_tflite)
    report = engine.predict(image, save_heatmap=True, output_dir=output_dir)

    if json_output:
        print(report.to_json())
    else:
        print(f"\n{'='*55}")
        print(f"  EDGE-DR Screening Report")
        print(f"{'='*55}")
        print(f"  Grade      : {report.grade} - {report.grade_label}")
        print(f"  Confidence : {report.confidence:.1f}%")
        print(f"  Action     : {report.triage_action}")
        print(f"  Urgency    : {report.urgency.upper()}")
        print(f"  Refer?     : {'YES [REFERRAL]' if report.refer else 'No'}")
        print(f"  Latency    : {report.inference_ms:.1f} ms")
        print(f"  Hindi      : {report.hindi_summary}")
        if report.heatmap_path:
            print(f"  Heatmap    : {report.heatmap_path}")
        print(f"{'='*55}\n")
        print(f"  Lesion localization:")
        for k, v in report.localization.items():
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()

