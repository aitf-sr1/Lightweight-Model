from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from config import (LABELS, N_LABELS, FOCAL_ALPHA, FOCAL_GAMMA,
                    ASL_GAMMA_POS, ASL_GAMMA_NEG,
                    LABEL_SMOOTH, THRESHOLD_SEARCH_FRAC, SEED)

MetricsResult = namedtuple('MetricsResult', [
    'f1_macro', 'f1_each', 'aucs',
    'subset_acc', 'mean_acc', 'per_label_acc',
])

EpochResult = namedtuple('EpochResult', [
    'loss', 'f1_macro', 'f1_each', 'aucs',
    'subset_acc', 'mean_acc', 'per_label_acc',
    'probs', 'targets',
])

class BCELoss(nn.Module):
    def __init__(self, label_smoothing: float = LABEL_SMOOTH, pos_weight=None):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.register_buffer(
            'pos_weight',
            pos_weight if pos_weight is not None else torch.ones(N_LABELS))

    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            ls      = self.label_smoothing
            targets = targets * (1.0 - ls / 2.0) + (1.0 - targets) * (ls / 2.0)
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight)

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = FOCAL_ALPHA, gamma: float = FOCAL_GAMMA,
                 label_smoothing: float = LABEL_SMOOTH, pos_weight=None):
        super().__init__()
        self.alpha           = alpha
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer(
            'pos_weight',
            pos_weight if pos_weight is not None else torch.ones(N_LABELS))

    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            ls = self.label_smoothing
            targets_s = targets * (1.0 - ls / 2.0) + (1.0 - targets) * (ls / 2.0)
        else:
            targets_s = targets
        probs   = torch.sigmoid(logits)
        bce     = F.binary_cross_entropy_with_logits(
            logits, targets_s, pos_weight=self.pos_weight, reduction='none')
        p_t     = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()

class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_pos: float = ASL_GAMMA_POS,
                 gamma_neg: float = ASL_GAMMA_NEG,
                 label_smoothing: float = LABEL_SMOOTH, pos_weight=None):
        super().__init__()
        self.gamma_pos       = gamma_pos
        self.gamma_neg       = gamma_neg
        self.label_smoothing = label_smoothing
        self.register_buffer(
            'pos_weight',
            pos_weight if pos_weight is not None else torch.ones(N_LABELS))

    def forward(self, logits, targets):
        targets_hard = targets.clone()
        if self.label_smoothing > 0:
            ls      = self.label_smoothing
            targets = targets * (1 - ls / 2) + (1 - targets) * (ls / 2)
        probs   = torch.sigmoid(logits)
        bce     = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pos_weight)
        gamma_t = self.gamma_pos * targets_hard + self.gamma_neg * (1 - targets_hard)
        p_t     = probs * targets_hard + (1 - probs) * (1 - targets_hard)
        return ((1 - p_t) ** gamma_t * bce).mean()

def make_criterion(loss_type: str, focal_gamma: float, asl_gamma_neg: float,
                   label_smooth: float, pos_weight: torch.Tensor) -> nn.Module:
    if loss_type == 'bce':
        return BCELoss(label_smoothing=label_smooth, pos_weight=pos_weight)
    if loss_type == 'focal':
        return FocalLoss(alpha=FOCAL_ALPHA, gamma=focal_gamma,
                         label_smoothing=label_smooth, pos_weight=pos_weight)
    if loss_type == 'asl':
        return AsymmetricFocalLoss(gamma_pos=ASL_GAMMA_POS, gamma_neg=asl_gamma_neg,
                                   label_smoothing=label_smooth, pos_weight=pos_weight)
    raise ValueError(f"Unknown loss_type: {loss_type}")

def compute_metrics(targets: np.ndarray, probs: np.ndarray,
                    thresholds=None) -> MetricsResult:
    if thresholds is None:
        thresholds = np.full(N_LABELS, 0.5)
    preds    = (probs >= thresholds[None, :]).astype(int)
    f1_macro = f1_score(targets, preds, average='macro', zero_division=0)
    f1_each  = f1_score(targets, preds, average=None,   zero_division=0)
    aucs = []
    for i in range(N_LABELS):
        try:
            aucs.append(roc_auc_score(targets[:, i], probs[:, i]))
        except Exception:
            aucs.append(0.5)
    subset_acc    = accuracy_score(targets, preds)
    per_label_acc = [accuracy_score(targets[:, i], preds[:, i])
                     for i in range(N_LABELS)]
    mean_acc = float(np.mean(per_label_acc))
    return MetricsResult(f1_macro=f1_macro, f1_each=f1_each, aucs=aucs,
                         subset_acc=subset_acc, mean_acc=mean_acc,
                         per_label_acc=per_label_acc)

def find_best_thresholds(targets: np.ndarray, probs: np.ndarray) -> np.ndarray:
    candidates = np.arange(0.1, 0.91, 0.01)
    best_t = np.full(N_LABELS, 0.5)
    for i in range(N_LABELS):
        best_f1 = -1.0
        for t in candidates:
            f1 = f1_score(targets[:, i].astype(int), (probs[:, i] >= t).astype(int),
                          zero_division=0)
            if f1 > best_f1:
                best_f1, best_t[i] = f1, t
    return best_t

def split_val_for_threshold(val_probs: np.ndarray, val_targets: np.ndarray,
                             frac: float = THRESHOLD_SEARCH_FRAC,
                             seed: int = SEED):
    n   = len(val_probs)
    rng = np.random.default_rng(seed)

    if abs(frac - 0.5) > 1e-6:
        idx      = rng.permutation(n)
        split    = int(n * frac)
        s_idx, e_idx = idx[:split], idx[split:]
    else:
        label_freq    = val_targets.sum(0).clip(min=1.0)
        sample_rarity = (val_targets * (1.0 / label_freq)).sum(1)
        tiebreak      = rng.random(n)
        order         = np.lexsort((tiebreak, -sample_rarity))
        s_idx, e_idx  = order[::2], order[1::2]

    return (val_probs[s_idx],  val_targets[s_idx],
            val_probs[e_idx],  val_targets[e_idx])
