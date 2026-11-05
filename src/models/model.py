"""
EDGE-DR — EfficientNet-B0 / MobileNetV3 Model Definition
==========================================================
Features:
  - EfficientNet-B0 and MobileNetV3-Large backbones (ImageNet pre-trained)
  - Custom classification head with dropout
  - Focal Loss for class-imbalanced training
  - Quadratic Weighted Kappa (QWK) metric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.metrics import cohen_kappa_score
import numpy as np


# ──────────────────────────────────────────────
# Focal Loss
# ──────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss (Lin et al. 2017).
    Reduces loss contribution from easy/well-classified samples,
    focusing training on hard, misclassified examples —
    critical for extreme class imbalance in DR grading (Grade 0 dominates).

    Args:
        gamma    : focusing parameter (default 2.0)
        alpha    : per-class weight tensor or None
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(self,
                 gamma: float = 2.0,
                 alpha: torch.Tensor = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha      # shape [num_classes]
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)                     # [B, C]
        p     = torch.exp(log_p)                                  # [B, C]

        # Gather log-probability of correct class
        log_pt = log_p.gather(1, targets.view(-1, 1)).squeeze(1)  # [B]
        pt     = p.gather(1, targets.view(-1, 1)).squeeze(1)      # [B]

        focal_term = (1.0 - pt) ** self.gamma
        loss = -focal_term * log_pt

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(0, targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ──────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────

class EdgeDRModel(nn.Module):
    """
    Lightweight DR classification model for edge deployment.

    Backbone options:
        'efficientnet_b0'   — 5.3M params, default choice
        'mobilenet_v3_large' — 5.5M params, slightly faster on NPU

    The final conv layer name exposed for Grad-CAM:
        EfficientNet-B0:      model.features[-1][0]
        MobileNetV3-Large:    model.features[-1][0]
    """

    BACKBONE_CONFIGS = {
        "efficientnet_b0": {
            "factory":    models.efficientnet_b0,
            "weights":    models.EfficientNet_B0_Weights.IMAGENET1K_V1,
            "in_features": 1280,
            "gradcam_layer": "features",   # last block of features
        },
        "mobilenet_v3_large": {
            "factory":    models.mobilenet_v3_large,
            "weights":    models.MobileNet_V3_Large_Weights.IMAGENET1K_V1,
            "in_features": 960,
            "gradcam_layer": "features",
        },
    }

    def __init__(self,
                 backbone: str = "efficientnet_b0",
                 num_classes: int = 5,
                 dropout: float = 0.3,
                 pretrained: bool = True):
        super().__init__()
        cfg = self.BACKBONE_CONFIGS[backbone]
        self.backbone_name = backbone

        # Load backbone
        weights = cfg["weights"] if pretrained else None
        base = cfg["factory"](weights=weights)

        # Strip original classifier
        self.features = base.features
        self.avgpool  = base.avgpool

        in_features = cfg["in_features"]

        # Custom head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.SiLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """Returns the last convolutional block for Grad-CAM hook."""
        return self.features[-1]


# ──────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────

def quadratic_weighted_kappa(preds: np.ndarray,
                              targets: np.ndarray) -> float:
    """Compute Quadratic Weighted Kappa (QWK) — primary metric for DR grading."""
    return cohen_kappa_score(targets, preds, weights="quadratic")


def compute_sensitivity_specificity(preds: np.ndarray,
                                     targets: np.ndarray,
                                     positive_classes: list = None) -> dict:
    """
    Compute binary sensitivity / specificity treating
    positive_classes (default: [1,2,3,4]) as 'DR present'.
    """
    if positive_classes is None:
        positive_classes = [1, 2, 3, 4]

    binary_true = np.isin(targets, positive_classes).astype(int)
    binary_pred = np.isin(preds,   positive_classes).astype(int)

    TP = ((binary_pred == 1) & (binary_true == 1)).sum()
    TN = ((binary_pred == 0) & (binary_true == 0)).sum()
    FP = ((binary_pred == 1) & (binary_true == 0)).sum()
    FN = ((binary_pred == 0) & (binary_true == 1)).sum()

    sensitivity = TP / (TP + FN + 1e-8)
    specificity = TN / (TN + FP + 1e-8)
    return {"sensitivity": sensitivity, "specificity": specificity,
            "TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN)}

