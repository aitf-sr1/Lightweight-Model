
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

from config import (LABELS, IMG_SIZE, VIZ_SIZE,
                    IMAGENET_MEAN, IMAGENET_STD, DATA_ROOT, FACEONLY_BASE, SEED)

_PATCH_SIZE   = 14
_N_PATCHES_1D = IMG_SIZE // _PATCH_SIZE
_N_PATCHES    = _N_PATCHES_1D ** 2

_BLOCK_LABEL = {
    1:  "Block 1 – low-level edges",
    3:  "Block 3 – textures / color",
    5:  "Block 5 – facial parts",
    7:  "Block 7 – expression structure",
    9:  "Block 9 – semantic regions",
    12: "Block 12 – high-level semantics",
}

class _GradAttnCapture:

    def __init__(self, model):
        self.model        = model
        self.maps: list   = []
        self._hooks       = []
        self._saved_fused = []

    def __enter__(self):
        maps = self.maps

        for block in self.model.backbone.blocks:
            attn  = block.attn
            saved = getattr(attn, 'fused_attn', False)
            self._saved_fused.append((attn, saved))
            attn.fused_attn = False

            def hook(m, inp, out, _maps=maps):
                a = inp[0]
                a.retain_grad()
                _maps.append(a)

            self._hooks.append(attn.attn_drop.register_forward_hook(hook))

        return self

    def compute_grads(self, logits: torch.Tensor):
        score = torch.sigmoid(logits).sum()
        score.backward()

    def __exit__(self, *_):
        for h in self._hooks:
            h.remove()
        for attn, saved in self._saved_fused:
            attn.fused_attn = saved
        self._hooks.clear()
        self._saved_fused.clear()

def _gradient_rollout(attn_maps: list) -> np.ndarray:
    N      = attn_maps[0].shape[-1]
    device = attn_maps[0].device
    eye    = torch.eye(N, device=device)
    rollout = eye.clone()

    for attn_t in attn_maps:
        avg_attn = attn_t[0].mean(0).detach()
        grad     = attn_t.grad
        if grad is not None:
            avg_grad = torch.relu(grad[0].mean(0).detach())
            weighted = avg_grad * avg_attn
        else:
            weighted = avg_attn

        aug     = weighted + eye
        aug     = aug / (aug.sum(-1, keepdim=True) + 1e-8)
        rollout = aug @ rollout

    cls_attn = rollout[0, 1:].cpu().numpy()
    grid     = cls_attn.reshape(_N_PATCHES_1D, _N_PATCHES_1D).astype(np.float32)
    grid     = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    return grid

def _gradient_single_layer(attn_t: torch.Tensor) -> np.ndarray | None:
    avg_attn = attn_t[0].mean(0).detach()
    grad     = attn_t.grad
    if grad is not None:
        avg_grad = torch.relu(grad[0].mean(0).detach())
        cls_vec  = (avg_grad[0, 1:] * avg_attn[0, 1:]).cpu().numpy()
    else:
        cls_vec  = avg_attn[0, 1:].cpu().numpy()

    if cls_vec.shape[0] != _N_PATCHES:
        return None

    grid = cls_vec.reshape(_N_PATCHES_1D, _N_PATCHES_1D).astype(np.float32)
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    return grid

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

def show_attention_grid(model, df_val, device: str, out_dir: Path,
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
                with _GradAttnCapture(model) as cap:
                    logits = model(img_t)
                    cap.compute_grads(logits)

            if not cap.maps:
                continue

            heat  = _gradient_rollout(cap.maps)
            blend = _blend(img_vis, heat)
            cv2.putText(blend, lbl, (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                        cv2.LINE_AA)
            panels.append(blend)
            break

    if not panels:
        return None

    combined = np.concatenate(panels, axis=1)
    outp = epoch_dir / f'attn_ep{epoch:02d}.png'
    Image.fromarray(combined).save(outp)
    return outp

def show_test_attention_combined(model, df_test, device: str,
                                 out_dir: Path,
                                 sample_count: int = 30,
                                 img_root: Path = DATA_ROOT) -> list[Path]:
    model.eval()
    test_dir = out_dir / 'test_attention'
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
            with _GradAttnCapture(model) as cap:
                logits = model(img_t)
                cap.compute_grads(logits)

        orig  = img_vis.copy()
        probs = torch.sigmoid(logits[0]).detach().cpu().numpy()
        pred_lbl = [LABELS[i] for i, p in enumerate(probs) if p > 0.5]
        text = ', '.join(pred_lbl) if pred_lbl else 'none'
        cv2.putText(orig, text[:28], (6, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        if cap.maps:
            heat = _gradient_rollout(cap.maps)
            attn = _blend(img_vis, heat)
        else:
            attn = img_vis.copy()
        cv2.putText(attn, "grad-rollout", (6, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        combined = np.concatenate([orig, attn], axis=1)
        outp     = test_dir / f'{fpath.stem}_attn.png'
        Image.fromarray(combined).save(outp)
        saved.append(outp)

    return saved

def show_layer_viz(model, sample_row, device: str, out_dir: Path,
                   img_root: Path = DATA_ROOT) -> Path | None:
    model.eval()
    fpath = _resolve(sample_row['frame_path'], img_root)
    try:
        img_pil = Image.open(fpath).convert('RGB')
    except Exception:
        return None

    img_vis = _img_arr(img_pil)
    img_t   = _img_tensor(img_pil, device).requires_grad_(True)

    with torch.enable_grad():
        with _GradAttnCapture(model) as cap:
            logits = model(img_t)
            cap.compute_grads(logits)

    if not cap.maps:
        return None

    n_blocks      = len(cap.maps)
    selected_0idx = [0, 2, 4, 6, 8, n_blocks - 1]

    orig = img_vis.copy()
    cv2.putText(orig, "Original", (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    panels = [orig]

    for idx in selected_0idx:
        if idx >= len(cap.maps):
            continue
        grid = _gradient_single_layer(cap.maps[idx])
        if grid is None:
            continue

        panel     = _blend(img_vis.copy(), grid, alpha=0.5)
        block_num = idx + 1
        label     = _BLOCK_LABEL.get(block_num, f"Block {block_num}")
        cv2.putText(panel, label[:20], (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)

    combined = np.concatenate(panels, axis=1)
    outp     = out_dir / f'layer_viz_{fpath.stem}.png'
    Image.fromarray(combined).save(outp)
    return outp
