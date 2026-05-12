# MobileViTV2 — 2D Emotion Classification

Backbone ringan generasi keempat untuk multi-label emotion classification. Tersedia dua varian size yang dipilih via `--model-size`. Mendukung resume dari checkpoint yang terinterupsi via `resume_run.py`.

## Arsitektur

```
Input [B, 3, 224, 224]
  └─ backbone: mobilevitv2_100 atau mobilevitv2_150 (timm, global_pool='')
       → spatial feature map [B, C, H, W]
  └─ pool: AdaptiveAvgPool2d(1) → [B, C, 1, 1]   ← titik hook GradCAM
  └─ head: Flatten → Linear(C, 256) → ReLU → Dropout → Linear(256, 4)
Output [B, 4] logit (sigmoid diterapkan di loss / saat inferensi; tidak ada sigmoid di dalam model)
```

Backbone dikonfigurasi `global_pool=''` agar mengembalikan spatial feature map. Head tidak menyertakan sigmoid — logit mentah dikeluarkan dan sigmoid diterapkan oleh loss function atau saat inferensi (`torch.sigmoid(model(x))`).

## Varian Model

| `--model-size` | timm name                                   | Catatan              |
|----------------|---------------------------------------------|----------------------|
| `small`        | `mobilevitv2_100`    | Default, lebih ringan |
| `medium`       | `mobilevitv2_150`  | Lebih akurat          |

## Konfigurasi Default

| Parameter         | Nilai                                   |
|-------------------|-----------------------------------------|
| Input resolusi    | 224 × 224                               |
| Normalisasi       | ImageNet mean/std standar               |
| Batch size        | 16 × accum 8 = effective batch 128      |
| Learning rate     | 3e-4 (backbone) / 3e-3 (head, 10×)      |
| Weight decay      | 1e-4                                    |
| Epochs            | 25                                      |
| Freeze epochs     | 3                                       |
| Warmup epochs     | 3 (linear warmup setelah unfreeze)      |
| Cosine annealing  | sisa epoch setelah warmup               |
| Grad clip         | 1.0                                     |
| Dropout           | 0.4                                     |
| Label smoothing   | 0.05                                    |
| Loss default      | `focal` (γ=2.0, α=0.5)                  |
| Seed              | 42                                      |

Batch size 16 dengan accum_steps 8 digunakan karena model medium membutuhkan lebih banyak VRAM.

## Strategi Training

**Fase freeze (epoch 1–3):** Hanya pool dan head dilatih. Scheduler: `ConstantLR(factor=1.0)`.

**Fase unfreeze (epoch 4–25):** Semua parameter aktif. Linear warmup 3 epoch dari 0.1× ke 1.0×, lalu CosineAnnealingLR. LR backbone 10× lebih kecil dari head.

**Resume otomatis:** `last.pth` disimpan setiap akhir epoch dengan state lengkap (model, optimizer, scheduler, scaler, best_f1, thresholds). Jika `last.pth` ditemukan saat training dimulai, training otomatis dilanjutkan dari epoch berikutnya tanpa perlu flag tambahan.

**Mixed precision:** `torch.autocast` + `GradScaler` aktif di CUDA.

## Resume Manual dari Run yang Terinterupsi

```bash
# Cari run_id di wandb dashboard atau nama folder runs/<mode>/<run_id>/
python resume_run.py <run_id>

# Override jumlah epoch lebih lama dari default
python resume_run.py <run_id> --epochs 30
```

Script `resume_run.py` mencari `runs/*/<run_id>/checkpoints/last.pth`, membaca konfigurasi dari checkpoint, lalu men-spawn `_sweep_worker.py` dengan `WANDB_RESUME=must` untuk melanjutkan run W&B yang sama.

## Loss Functions

| `loss_type` | Kelas                 | Parameter kunci                          |
|-------------|------------------------|------------------------------------------|
| `bce`       | `BCELoss`              | `label_smoothing`, `pos_weight`          |
| `focal`     | `FocalLoss`            | `focal_gamma ∈ {1.5,2.0,2.5}`           |
| `asl`       | `AsymmetricFocalLoss`  | `asl_gamma_neg ∈ {2.0,3.0,4.0}`         |

`pos_weight` = `neg_count / pos_count` per label dari train CSV.

## Dataset & Augmentasi

| Mode        | Sumber gambar                              | CSV split                              |
|-------------|--------------------------------------------|----------------------------------------|
| `normal`    | `hasil_label9/cropped_faces/`              | `hasil_label9/Label2d/`                |
| `supercrop` | `hasil_label9/tighter_cropped_faces/`      | `hasil_label9/Label2d-super-crop/`     |
| `faceonly`  | `hasil_label9_faceonly/faceonly_faces/`    | `hasil_label9_faceonly/Label2d/`       |

**Augmentasi train:** RandomResizedCrop (scale 0.8–1.0), hflip (p=0.5), brightness ±0.2, saturation (p=0.5), hue (p=0.3), GaussianBlur (p=0.2).

**NumWorkers:** Di-set ke 0 saat `debugpy`/`pydevd` aktif.

## Threshold Per-Label

Val set dibagi dua dengan stratifikasi rarity label. Grid-search threshold di separuh pertama; evaluasi F1 di separuh kedua untuk menghindari leakage.

## Visualisasi (GradCAM)

`gradcam.py` mengimplementasikan:

**`_GradCAMCapture`:** Hook pada `model.pool`. Setelah backward, gradient × feature map → ReLU → heatmap.

**`_StageCapture`:** Hook pada beberapa `model.backbone.stages[idx]` untuk GradCAM multi-stage.

Output:
- `viz/epoch_XX/gradcam_epXX.png` — grid per label dari val set (lokal saja, tidak di-upload ke W&B untuk hemat storage)
- `viz/test_gradcam/` — original | GradCAM overlay test samples
- `viz/stage_viz_*.png` — panel multi-stage test (di-upload ke W&B)

## Sweep Hyperparameter (W&B)

```
loss_type     ∈ {bce, asl}          (focal tidak disertakan di sweep ini)
lr            ∈ log-uniform[5e-6, 5e-4]
lr_head_mult  ∈ {5, 10, 20}
dropout       ∈ {0.3, 0.4, 0.5}
weight_decay  ∈ {1e-5, 1e-4, 1e-3}
label_smooth  ∈ {0.0, 0.05, 0.1}
focal_gamma   ∈ {1.5, 2.0, 2.5}
asl_gamma_neg ∈ {2.0, 3.0, 4.0}
```

Mode `compare` menambah `dataset_mode ∈ {normal, supercrop, faceonly}`. Setiap run di subprocess terpisah via `_sweep_worker.py`.

## Cara Pakai

```bash
cd training/2d/mobilevitv2

# Training baseline
python train.py --mode normal    --loss focal --model-size small  --wandb
python train.py --mode normal    --loss asl   --model-size medium --wandb
python train.py --mode faceonly  --wandb

# Sweep
python sweep.py --mode normal    --count 40
python sweep.py --mode compare   --count 60
python sweep.py --mode normal    --sweep-id <id> --count 20

# Resume run terinterupsi
python resume_run.py <run_id>
python resume_run.py <run_id> --epochs 35
```

## Output Run

```
runs/<mode>/<run_id>/
├── checkpoints/
│   ├── best.pth          # state_dict + thresholds + f1_macro + cfg
│   ├── best_epXX.pth     # snapshot epoch 1, tiap 3 epoch, epoch terakhir
│   └── last.pth          # full state tiap akhir epoch (untuk resume)
├── viz/
│   ├── epoch_XX/         # GradCAM grid val (lokal saja)
│   ├── test_gradcam/     # test overlay
│   └── stage_viz_*.png
└── metrics.csv
```
