
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

from config import (LABELS, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD,
                    DATA_ROOT, FACEONLY_BASE, SEED)

VIZ_SIZE = 384

_STAGE_LABEL = {
    0: "Stage 1 – edges",
    1: "Stage 2 – textures",
    3: "Stage 4 – parts",
    5: "Stage 6 – semantics",
}

class _GradCAMCapture:

    def __init__(self, model):
        self.model      = model
        self._act       = None
        self._hooks     = []

    def __enter__(self):
        def fwd_hook(m, inp, out):
            self._act = inp[0]
            self._act.retain_grad()

        self._hooks.append(self.model.pool.register_forward_hook(fwd_hook))
        return self

    def compute_grads(self, logits: torch.Tensor):
        torch.sigmoid(logits).sum().backward()

    @property
    def heatmap(self) -> np.ndarray | None:
        if self._act is None or self._act.grad is None:
            return None
        grad    = self._act.grad
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam     = torch.relu((weights * self._act).sum(1))
        cam     = cam[0].detach().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def __exit__(self, *_):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class _StageCapture:

    def __init__(self, model, stage_indices: list[int]):
        self.model         = model
        self.stage_indices = stage_indices
        self._acts: dict   = {}
        self._hooks        = []

    def __enter__(self):
        for idx in self.stage_indices:
            stages = self.model.backbone.blocks
            if idx >= len(stages):
                continue

            def hook(m, inp, out, _i=idx):
                out.retain_grad()
                self._acts[_i] = out

            self._hooks.append(stages[idx].register_forward_hook(hook))
        return self

    def compute_grads(self, logits: torch.Tensor):
        torch.sigmoid(logits).sum().backward()

    def heatmap_for(self, idx: int) -> np.ndarray | None:
        act = self._acts.get(idx)
        if act is None or act.grad is None:
            return None
        weights = act.grad.mean(dim=(2, 3), keepdim=True)
        cam     = torch.relu((weights * act).sum(1))
        cam     = cam[0].detach().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def __exit__(self, *_):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def _img_tensor(img_pil: Image.Image, device: str) -> torch.Tensor:
    resample = getattr(Image, 'Resampling', Image).BILINEAR
    img_r    = img_pil.resize((IMG_SIZE, IMG_SIZE), resample)
    return TF.normalize(TF.to_tensor(img_r),
                        IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(device)

def _img_arr(img_pil: Image.Image) -> np.ndarray:
    resample = getattr(Image, 'Resampling', Image).BILINEAR
    return np.array(img_pil.resize((VIZ_SIZE, VIZ_SIZE), resample))

def _blend(img_arr: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    cmap = cv2.applyColorMap(np.uint8(255 * cv2.resize(heat, (VIZ_SIZE, VIZ_SIZE))),
                              cv2.COLORMAP_JET)
    cmap = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_arr, 1 - alpha, cmap, alpha, 0)

def _resolve(p: str, img_root: Path = DATA_ROOT) -> Path:
    p = Path(p)
    return p if p.is_absolute() else img_root / p

def show_gradcam_grid(model, df_val, device: str, out_dir: Path,
                      epoch: int, img_root: Path = DATA_ROOT) -> Path | None:
    model.eval()
    epoch_dir = out_dir / f'epoch_{epoch:02d}'
    epoch_dir.mkdir(parents=True, exist_ok=True)
    panels = []

    for lbl in LABELS:
        subset = df_val[df_val[lbl] == 1]
        if subset.empty:
            continue
        subset = subset.sample(1, random_state=SEED)

        for _, row in subset.iterrows():
            fpath = _resolve(row['frame_path'], img_root)
            try:
                img_pil = Image.open(fpath).convert('RGB')
            except Exception:
                continue

            img_vis = _img_arr(img_pil)
            img_t   = _img_tensor(img_pil, device).requires_grad_(True)

            with torch.enable_grad():
                with _GradCAMCapture(model) as cap:
                    logits = model(img_t)
                    cap.compute_grads(logits)

            heat = cap.heatmap
            if heat is None:
                continue

            blend = _blend(img_vis, heat)
            cv2.putText(blend, lbl, (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                        cv2.LINE_AA)
            panels.append(blend)
            break

    if not panels:
        return None

    combined = np.concatenate(panels, axis=1)
    outp = epoch_dir / f'gradcam_ep{epoch:02d}.png'
    Image.fromarray(combined).save(outp)
    return outp

def show_test_gradcam_combined(model, df_test, device: str,
                                out_dir: Path,
                                sample_count: int = 30,
                                img_root: Path = DATA_ROOT) -> list[Path]:
    model.eval()
    test_dir = out_dir / 'test_gradcam'
    test_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    subset = df_test.sample(min(sample_count, len(df_test)), random_state=SEED)

    for _, row in subset.iterrows():
        fpath = _resolve(row['frame_path'], img_root)
        try:
            img_pil = Image.open(fpath).convert('RGB')
        except Exception:
            continue

        img_vis = _img_arr(img_pil)
        img_t   = _img_tensor(img_pil, device).requires_grad_(True)

        with torch.enable_grad():
            with _GradCAMCapture(model) as cap:
                logits = model(img_t)
                cap.compute_grads(logits)

        orig  = img_vis.copy()
        probs = torch.sigmoid(logits[0]).detach().cpu().numpy()
        pred_lbl = [LABELS[i] for i, p in enumerate(probs) if p > 0.5]
        text = ', '.join(pred_lbl) if pred_lbl else 'none'
        cv2.putText(orig, text[:28], (6, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        heat = cap.heatmap
        viz  = _blend(img_vis, heat) if heat is not None else img_vis.copy()
        cv2.putText(viz, "gradcam", (6, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        combined = np.concatenate([orig, viz], axis=1)
        outp     = test_dir / f'{fpath.stem}_gcam.png'
        Image.fromarray(combined).save(outp)
        saved.append(outp)

    return saved

def show_stage_viz(model, sample_row, device: str, out_dir: Path,
                   img_root: Path = DATA_ROOT) -> Path | None:
    model.eval()
    fpath = _resolve(sample_row['frame_path'], img_root)
    try:
        img_pil = Image.open(fpath).convert('RGB')
    except Exception:
        return None

    img_vis = _img_arr(img_pil)
    img_t   = _img_tensor(img_pil, device).requires_grad_(True)

    n_stages = len(model.backbone.blocks)
    selected = [i for i in [0, 1, 3, n_stages - 1] if i < n_stages]

    with torch.enable_grad():
        with _StageCapture(model, selected) as cap:
            logits = model(img_t)
            cap.compute_grads(logits)

    orig = img_vis.copy()
    cv2.putText(orig, "Original", (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    panels = [orig]

    for idx in selected:
        heat = cap.heatmap_for(idx)
        if heat is None:
            continue
        panel = _blend(img_vis.copy(), heat, alpha=0.5)
        label = _STAGE_LABEL.get(idx, f"Stage {idx + 1}")
        cv2.putText(panel, label, (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)

    if len(panels) < 2:
        return None

    combined = np.concatenate(panels, axis=1)
    outp     = out_dir / f'stage_viz_{fpath.stem}.png'
    Image.fromarray(combined).save(outp)
    return outp
