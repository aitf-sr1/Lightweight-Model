# MobileNetV3-Small — 2D Emotion Classification

Backbone ringan untuk skenario inferensi cepat atau device dengan VRAM kecil. MobileNetV3-Small sangat kompak (~2.5M parameter) sehingga memungkinkan batch size besar.

## Arsitektur

```
Input [B, 3, 224, 224]
  └─ backbone: mobilenetv3_small_100.lamb_in1k (timm, global_pool='')
       → spatial feature map [B, C, H, W]
  └─ pool: AdaptiveAvgPool2d(1) → [B, C, 1, 1]   ← titik hook GradCAM
  └─ head: Flatten → Linear(C, 256) → ReLU → Dropout → Linear(256, 4)
Output [B, 4] logit (sigmoid diterapkan di loss / saat inferensi)
```

Backbone dikonfigurasi `global_pool=''` agar mengembalikan spatial feature map. Pooling dilakukan terpisah sehingga GradCAM bisa di-hook di input pooling.

## Konfigurasi Default

| Parameter         | Nilai                                   |
|-------------------|-----------------------------------------|
| Backbone (timm)   | `mobilenetv3_small_100.lamb_in1k`       |
| Input resolusi    | 224 × 224                               |
| Normalisasi       | ImageNet mean/std standar               |
| Batch size        | 64 × accum 2 = effective batch 128      |
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

Batch size 64 dipilih karena MobileNetV3-Small sangat ringan sehingga aman menggunakan batch besar bahkan di VRAM terbatas. Dengan accum_steps=2 efektif batch size tetap 128.

## Strategi Training

**Fase freeze (epoch 1–3):** Hanya pool dan head dilatih; backbone dibekukan. Scheduler: `ConstantLR(factor=1.0)`.

**Fase unfreeze (epoch 4–25):** Semua parameter aktif. Linear warmup 3 epoch (0.1× → 1.0×), dilanjut CosineAnnealingLR. LR backbone 10× lebih kecil dari head.

**Evaluasi terpisah:** `eval_criterion` tanpa `pos_weight` untuk val/test agar loss antar run comparable.

**Mixed precision:** `torch.autocast` + `GradScaler` aktif di CUDA.

## Loss Functions

| `loss_type` | Kelas                 | Parameter kunci                          |
|-------------|------------------------|------------------------------------------|
| `bce`       | `BCELoss`              | `label_smoothing`, `pos_weight`          |
| `focal`     | `FocalLoss`            | `focal_gamma ∈ {1.5,2.0,2.5}`, `alpha`  |
| `asl`       | `AsymmetricFocalLoss`  | `asl_gamma_neg ∈ {2.0,3.0,4.0}`         |

`pos_weight` = `neg_count / pos_count` per label dari train CSV. Hanya dipakai di training criterion.

## Dataset & Augmentasi

| Mode        | Sumber gambar                              | CSV split                              |
|-------------|--------------------------------------------|----------------------------------------|
| `normal`    | `hasil_label9/cropped_faces/`              | `hasil_label9/Label2d/`                |
| `supercrop` | `hasil_label9/tighter_cropped_faces/`      | `hasil_label9/Label2d-super-crop/`     |
| `faceonly`  | `hasil_label9_faceonly/faceonly_faces/`    | `hasil_label9_faceonly/Label2d/`       |

**Augmentasi train:** RandomResizedCrop (scale 0.8–1.0, ratio 0.9–1.1), hflip (p=0.5), brightness ±0.2, saturation 0.8–1.2 (p=0.5), hue ±0.05 (p=0.3), GaussianBlur (p=0.2). Val/test: resize deterministik.

**NumWorkers:** Di-set ke 0 saat `debugpy`/`pydevd` aktif (konflik dengan `os.fork()` di VSCode debugger).

## Threshold Per-Label

Val set dibagi dua dengan stratifikasi rarity label. Grid-search threshold (0.1–0.9, step 0.01) di separuh pertama; evaluasi F1 di separuh kedua untuk menghindari leakage.

## Visualisasi (GradCAM)

`gradcam.py` mengimplementasikan dua kaptur:

**`_GradCAMCapture`:** Hook pada `model.pool`. Setelah backward, gradient di-average spasial × feature map → ReLU → heatmap.

**`_StageCapture`:** Hook pada beberapa `model.backbone.stages[idx]` untuk GradCAM multi-stage. Menampilkan evolusi fitur dari low-level (edges/textures) ke high-level (semantics).

Output:
- `viz/epoch_XX/gradcam_epXX.png` — grid per label dari val set (setiap 3 epoch), hanya disimpan lokal (tidak di-upload ke W&B untuk hemat storage)
- `viz/test_gradcam/` — original | GradCAM overlay test samples
- `viz/stage_viz_*.png` — panel multi-stage test (di-upload ke W&B)

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

Mode `compare` menambah `dataset_mode ∈ {normal, supercrop, faceonly}`. Metode Bayes, early termination Hyperband (min_iter=5). Setiap run di subprocess terpisah via `_sweep_worker.py`.

## Cara Pakai

```bash
cd training/2d/mobilenetv3

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
│   └── best_epXX.pth
├── viz/
│   ├── epoch_XX/         # GradCAM grid val (lokal saja)
│   ├── test_gradcam/     # test overlay
│   └── stage_viz_*.png   # panel multi-stage
└── metrics.csv
```
