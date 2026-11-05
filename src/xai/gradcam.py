"""
EDGE-DR — Grad-CAM Explainability Engine
==========================================
Implements Gradient-weighted Class Activation Mapping (Grad-CAM)
for fundus image explainability.

Produces:
  - Saliency heatmaps overlaid on fundus images
  - Clinical lesion annotation (microaneurysms / hemorrhages / exudates)
  - Quadrant-level localization text report

Usage:
    from src.xai.gradcam import GradCAMExplainer
    explainer = GradCAMExplainer(model)
    result = explainer.explain(image_tensor, pred_class)
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


# ──────────────────────────────────────────────
# Grad-CAM core
# ──────────────────────────────────────────────

class GradCAMExplainer:
    """
    Grad-CAM implementation with clean hook-based gradient capture.

    Zero-compute overhead: saliency is computed in a single
    forward + backward pass through the existing inference graph —
    no second network pass required.

    Args:
        model       : EdgeDRModel (or any PyTorch model)
        target_layer: nn.Module — the last convolutional block
                      (defaults to model.gradcam_target_layer)
        device      : torch.device
    """

    def __init__(self, model, target_layer=None, device=None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()

        self.target_layer = target_layer or model.gradcam_target_layer

        self._gradients  = None
        self._activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def _compute_heatmap(self, logits: torch.Tensor,
                          target_class: int,
                          img_h: int, img_w: int) -> np.ndarray:
        """Core Grad-CAM computation — returns float32 heatmap in [0,1]."""
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # Global average pool of gradients over spatial dims
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # [1, C, 1, 1]
        cam     = (weights * self._activations).sum(dim=1, keepdim=True)  # [1,1,H,W]
        cam     = F.relu(cam)                                       # only positive
        cam     = cam.squeeze().cpu().numpy()

        # Resize to original image size
        if cam.ndim == 0:   # scalar edge case
            cam = np.ones((img_h, img_w), dtype=np.float32)
        else:
            cam = cv2.resize(cam, (img_w, img_h))

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)

    def explain(self,
                img_tensor: torch.Tensor,
                target_class: Optional[int] = None,
                alpha: float = 0.4,
                colormap: int = cv2.COLORMAP_JET) -> "GradCAMResult":
        """
        Run Grad-CAM on a single preprocessed image tensor.

        Args:
            img_tensor   : [1, 3, H, W] normalised tensor
            target_class : Class to explain. If None, uses argmax.
            alpha        : Heatmap blend weight (0=no overlay, 1=full overlay)
            colormap     : OpenCV colormap for rendering

        Returns:
            GradCAMResult with heatmap, overlay, predicted class, confidence
        """
        img_tensor = img_tensor.to(self.device).requires_grad_(True)
        _, _, ih, iw = img_tensor.shape

        with torch.enable_grad():
            logits = self.model(img_tensor)

        probs = torch.softmax(logits, dim=1).squeeze().detach().cpu().numpy()
        pred_class = int(probs.argmax())
        confidence = float(probs.max())

        explain_class = target_class if target_class is not None else pred_class
        heatmap = self._compute_heatmap(logits, explain_class, ih, iw)

        # Build colour overlay
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        coloured = cv2.applyColorMap(heatmap_uint8, colormap)       # BGR

        # Reconstruct original image (undo ImageNet normalisation)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        orig = img_tensor.squeeze().detach().cpu().numpy()           # [3,H,W]
        orig = np.transpose(orig, (1, 2, 0))                        # [H,W,3]
        orig = (orig * std + mean).clip(0, 1)
        orig_bgr = (orig[..., ::-1] * 255).astype(np.uint8)         # RGB→BGR

        overlay = cv2.addWeighted(orig_bgr, 1 - alpha, coloured, alpha, 0)

        # Lesion localization report
        localization = _localize_lesions(heatmap)

        return GradCAMResult(
            heatmap=heatmap,
            overlay_bgr=overlay,
            original_bgr=orig_bgr,
            pred_class=pred_class,
            class_probs=probs,
            confidence=confidence,
            explained_class=explain_class,
            localization=localization,
        )

    def remove_hooks(self):
        """Remove registered hooks to prevent memory leaks."""
        for handle in [self._fwd_handle, self._bwd_handle]:
            if handle:
                handle.remove()


# ──────────────────────────────────────────────
# Lesion localization helper
# ──────────────────────────────────────────────

QUADRANT_NAMES = {
    (0, 0): "Superior-Nasal",
    (0, 1): "Superior-Temporal",
    (1, 0): "Inferior-Nasal",
    (1, 1): "Inferior-Temporal",
}


def _localize_lesions(heatmap: np.ndarray,
                       activation_threshold: float = 0.6) -> dict:
    """
    Identify quadrants with high saliency and estimate lesion type
    based on activation intensity pattern.

    Returns dict:
        quadrants      : list of active quadrant names
        peak_activation: float, max activation value
        lesion_hints   : list of probable lesion type strings
    """
    h, w = heatmap.shape
    # Split into 2×2 quadrants
    quadrant_scores = {}
    for qi in range(2):
        for qj in range(2):
            block = heatmap[qi * h//2 : (qi+1) * h//2,
                             qj * w//2 : (qj+1) * w//2]
            quadrant_scores[(qi, qj)] = float(block.mean())

    active_quadrants = [
        QUADRANT_NAMES[k]
        for k, v in quadrant_scores.items()
        if v >= activation_threshold * max(quadrant_scores.values())
    ]

    peak = float(heatmap.max())

    # Heuristic lesion hint based on peak activation pattern
    lesion_hints = []
    if peak > 0.9:
        lesion_hints.append("Likely neovascularization (high dense focal activation)")
    elif peak > 0.7:
        lesion_hints.append("Likely blot hemorrhages or exudates")
    elif peak > 0.5:
        lesion_hints.append("Likely microaneurysms (diffuse low activation)")
    else:
        lesion_hints.append("No significant focal lesion detected")

    return {
        "quadrants":       active_quadrants,
        "peak_activation": peak,
        "lesion_hints":    lesion_hints,
        "quadrant_scores": {QUADRANT_NAMES[k]: round(v, 4)
                            for k, v in quadrant_scores.items()},
    }


# ──────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────

@dataclass
class GradCAMResult:
    heatmap:       np.ndarray     # float32 [H,W] in [0,1]
    overlay_bgr:   np.ndarray     # uint8  [H,W,3] — BGR image with heatmap
    original_bgr:  np.ndarray     # uint8  [H,W,3] — reconstructed original
    pred_class:    int            # argmax predicted class 0-4
    class_probs:   np.ndarray     # float32 [5] softmax probabilities
    confidence:    float          # max probability
    explained_class: int          # class the CAM was computed for
    localization:  dict           # quadrant analysis

    # Convenience properties
    @property
    def triage_action(self) -> str:
        from src.data.dataset import DR_CLASSES, REFERRAL_GRADES
        grade = self.pred_class
        if grade in REFERRAL_GRADES:
            urgency = "within 1 week" if grade == 4 else \
                      "within 2–4 weeks" if grade == 3 else \
                      "within 4–6 weeks"
            return f"REFER to District Eye Hospital {urgency}"
        return "MONITOR: Repeat annual screening"

    @property
    def grade_label(self) -> str:
        labels = ["No DR", "Mild NPDR", "Moderate NPDR",
                  "Severe NPDR", "Proliferative DR (PDR)"]
        return labels[self.pred_class]


# ──────────────────────────────────────────────
# Save utility
# ──────────────────────────────────────────────

def save_explanation(result: GradCAMResult, output_path: str):
    """Save side-by-side: original | heatmap overlay."""
    side_by_side = np.hstack([result.original_bgr, result.overlay_bgr])
    cv2.imwrite(output_path, side_by_side)
    return output_path

