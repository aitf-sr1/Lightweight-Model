# EfficientNetV2-B0 — 2D Emotion Classification

CNN modern berbasis EfficientNetV2 untuk image-level multi-label emotion classification. Backbone `tf_efficientnetv2_b0.in1k` dipretrain ImageNet-1k. Resolusi input 192×192 lebih kecil dari versi -S untuk menghemat VRAM.

## Arsitektur

```
Input [B, 3, 192, 192]
  └─ backbone: tf_efficientnetv2_b0.in1k (timm, global_pool='')
       → spatial feature map [B, C, H, W]
  └─ pool: AdaptiveAvgPool2d(1) → [B, C, 1, 1]   ← titik hook GradCAM
  └─ head: Flatten → Linear(C, 256) → ReLU → Dropout → Linear(256, 4)
Output [B, 4] logit (sigmoid diterapkan di loss / saat inferensi)
```

Backbone dikonfigurasi dengan `global_pool=''` sehingga mengembalikan spatial feature map `[B, C, H, W]` (bukan vektor). Pooling dilakukan terpisah oleh `self.pool` agar GradCAM dapat di-hook tepat di input pooling.

## Konfigurasi Default

| Parameter         | Nilai                                   |
|-------------------|-----------------------------------------|
| Backbone (timm)   | `tf_efficientnetv2_b0.in1k`             |
| Input resolusi    | 192 × 192                               |
| Normalisasi       | ImageNet mean/std standar               |
| Batch size        | 32 × accum 4 = effective batch 128      |
| Learning rate     | 3e-4 (backbone) / 3e-3 (head, 10×)      |
| Weight decay      | 1e-4                                    |
| Epochs            | 25                                      |
| Freeze epochs     | 3 (backbone dibekukan epoch 1–3)        |
| Warmup epochs     | 3 (linear warmup setelah unfreeze)      |
| Cosine annealing  | sisa epoch setelah warmup               |
| Grad clip         | 1.0                                     |
| Dropout           | 0.5                                     |
| Label smoothing   | 0.05                                    |
| Loss default      | `focal` (γ=2.0, α=0.5)                  |
| Seed              | 42                                      |

## Strategi Training

**Fase freeze (epoch 1–3):** Hanya pool dan head dilatih; backbone dibekukan. Scheduler: `ConstantLR(factor=1.0)` agar LR tidak berubah.

**Fase unfreeze (epoch 4–25):** LR backbone dan head di-reset ke nilai awal, lalu linear warmup 3 epoch dari 0.1× ke 1.0×, diikuti CosineAnnealingLR sisa epoch.

**Evaluasi terpisah dari training:** `eval_criterion` dibuat tanpa `pos_weight` untuk val/test sehingga val-loss tidak dipengaruhi distribusi train dan bisa dibandingkan secara fair antar run sweep.

**Mixed precision:** `torch.autocast` + `GradScaler` aktif di CUDA.

## Loss Functions

| `loss_type` | Kelas                 | Parameter kunci                          |
|-------------|------------------------|------------------------------------------|
| `bce`       | `BCELoss`              | `label_smoothing`, `pos_weight`          |
| `focal`     | `FocalLoss`            | `focal_gamma ∈ {1.5,2.0,2.5}`, `alpha`  |
| `asl`       | `AsymmetricFocalLoss`  | `asl_gamma_neg ∈ {2.0,3.0,4.0}`         |

`pos_weight` = `neg_count / pos_count` per label, dihitung dari train CSV saja. Dipakai hanya untuk training criterion; eval criterion tidak memakai pos_weight.

## Dataset & Augmentasi

| Mode        | Sumber gambar                              | CSV split                              |
|-------------|--------------------------------------------|----------------------------------------|
| `normal`    | `hasil_label9/cropped_faces/`              | `hasil_label9/Label2d/`                |
| `supercrop` | `hasil_label9/tighter_cropped_faces/`      | `hasil_label9/Label2d-super-crop/`     |
| `faceonly`  | `hasil_label9_faceonly/faceonly_faces/`    | `hasil_label9_faceonly/Label2d/`       |

**Augmentasi train:** RandomResizedCrop (scale 0.8–1.0, ratio 0.9–1.1), hflip (p=0.5), brightness ±0.2, saturation 0.8–1.2 (p=0.5), hue ±0.05 (p=0.3), GaussianBlur radius 0.5–1.5 (p=0.2). Val/test: resize deterministik.

**NumWorkers:** Di-set ke 0 ketika `debugpy` atau `pydevd` terdeteksi di `sys.modules` (VSCode debugger konflik dengan `os.fork()`).

## Threshold Per-Label

Val set dibagi dua dengan stratifikasi rarity label (sampel dengan label langka dibagi merata ke kedua separuh). Separuh pertama untuk grid-search threshold (0.1–0.9, step 0.01) yang memaksimalkan F1 per label. Separuh kedua untuk evaluasi final agar tidak ada leakage threshold→F1.

## Visualisasi (GradCAM)

`gradcam.py` mengimplementasikan dua mekanisme:

**`_GradCAMCapture`:** Hook pada `model.pool` (input AdaptiveAvgPool2d). Setelah backward pass, gradient di-average secara spasial menjadi bobot, dikalikan feature map, dan di-ReLU → heatmap `[H, W]`.

**`_StageCapture`:** Hook pada `model.backbone.blocks[idx]` untuk beberapa indeks stage. Menghasilkan GradCAM per stage dari low-level (edges) hingga high-level (semantics). Untuk EfficientNetV2, backbone stage diakses via `model.backbone.blocks`.

Output yang disimpan:
- `viz/epoch_XX/gradcam_epXX.png` — grid satu sample per label dari val set (setiap 3 epoch)
- `viz/test_gradcam/` — original | GradCAM overlay untuk setiap test sample
- `viz/stage_viz_*.png` — panel multi-stage dari satu sample test
- Test GradCAM di-log ke W&B (maks 10 gambar); `stage_viz` juga di-log ke W&B

## Sweep Hyperparameter (W&B)

```
loss_type     ∈ {bce, focal, asl}
lr            ∈ log-uniform[5e-6, 5e-4]
lr_head_mult  ∈ {5, 10, 20}
dropout       ∈ {0.3, 0.4, 0.5}
weight_decay  ∈ {1e-5, 1e-4, 1e-3}
label_smooth  ∈ {0.0, 0.05, 0.1}
focal_gamma   ∈ {1.5, 2.0, 2.5}
asl_gamma_neg ∈ {2.0, 3.0, 4.0}
```

Mode `compare` menambah `dataset_mode ∈ {normal, supercrop, faceonly}`. Metode Bayes, early termination Hyperband (min_iter=5, eta=2). Setiap run dijalankan di subprocess terpisah via `_sweep_worker.py`.

## Cara Pakai

```bash
cd training/2d/efficientnetv2

python train.py --mode normal    --wandb
python train.py --mode supercrop --loss asl --wandb
python train.py --mode faceonly  --wandb

python sweep.py --mode normal    --count 40
python sweep.py --mode compare   --count 60
python sweep.py --mode normal    --sweep-id <id> --count 20
```

## Output Run

```
runs/<mode>/<run_id>/
├── checkpoints/
│   ├── best.pth          # state_dict + thresholds + f1_macro + cfg
│   └── best_epXX.pth     # snapshot epoch 1, tiap 3 epoch, epoch terakhir
├── viz/
│   ├── epoch_XX/         # GradCAM grid per label (val set)
│   ├── test_gradcam/     # original + GradCAM overlay test samples
│   └── stage_viz_*.png   # panel multi-stage test sample
└── metrics.csv           # per-epoch: loss, f1, mean_acc, sub_acc, lr, elapsed
```
