# ConvNeXt V2 Pico — 2D Emotion Classification

Modern ConvNet untuk image-level multi-label emotion classification. Backbone ConvNeXt V2 Pico di-pretrain dengan FCMAE (Fully Convolutional Masked Autoencoder) lalu fine-tuned ke ImageNet-1k sebelum di-fine-tune ke task emosi.

## Arsitektur

```
Input [B, 3, 224, 224]
  └─ backbone: convnextv2_atto.fcmae_ft_in1k (timm)
       global_pool='' → spatial feature map [B, C, H, W]
  └─ pool: AdaptiveAvgPool2d(1) → [B, C, 1, 1]   ← titik hook GradCAM
  └─ head: Flatten → Linear(C, 256) → ReLU → Dropout → Linear(256, 4)
Output [B, 4] logit (sigmoid diterapkan di loss / saat inferensi)
```

Backbone dikonfigurasi dengan `global_pool=''` agar mengembalikan spatial feature map `[B, C, H, W]` alih-alih vektor. Pooling dilakukan secara terpisah oleh `self.pool` sehingga GradCAM bisa di-hook tepat di lapisan sebelum pooling.

## Konfigurasi Default

| Parameter         | Nilai                                   |
|-------------------|-----------------------------------------|
| Backbone (timm)   | `convnextv2_atto.fcmae_ft_in1k`         |
| Input resolusi    | 224 × 224                               |
| Normalisasi       | ImageNet mean/std standar               |
| Batch size        | 32 × accum 4 = effective batch 128      |
| Learning rate     | 3e-4 (backbone) / 3e-3 (head, 10×)      |
| Weight decay      | 1e-4                                    |
| Epochs            | 25                                      |
| Freeze epochs     | 3 (backbone dibekukan di epoch 1–3)     |
| Warmup epochs     | 3 (linear warmup setelah unfreeze)      |
| Cosine annealing  | sisa epoch setelah warmup               |
| Grad clip         | 1.0                                     |
| Dropout           | 0.5                                     |
| Label smoothing   | 0.05                                    |
| Loss default      | `focal` (γ=2.0, α=0.5)                  |
| Seed              | 42                                      |

## Strategi Training

**Fase freeze (epoch 1–3):** Backbone dibekukan (`requires_grad=False`), hanya pool dan head yang dilatih. Scheduler: `ConstantLR(factor=1.0)` agar LR tidak berubah selama fase ini.

**Fase unfreeze (epoch 4–25):** Seluruh backbone dibuka, lalu LR di-warmup linear selama 3 epoch (dari 0.1× ke 1.0×), diikuti CosineAnnealingLR untuk sisa epoch. Backbone diberi LR 10× lebih kecil dari head.

**Gradient accumulation:** `loss / ACCUM_STEPS` sebelum backward; optimizer step dilakukan setiap `ACCUM_STEPS` iterasi.

**Mixed precision:** `torch.autocast` + `GradScaler` aktif di CUDA.

## Loss Functions

Semua loss bersifat multi-label (sigmoid, bukan softmax). Dipilih via `--loss`:

| `loss_type` | Kelas                 | Parameter kunci                          |
|-------------|------------------------|------------------------------------------|
| `bce`       | `BCELoss`              | `label_smoothing`, `pos_weight`          |
| `focal`     | `FocalLoss`            | `focal_gamma ∈ {1.5,2.0,2.5}`, `alpha`  |
| `asl`       | `AsymmetricFocalLoss`  | `asl_gamma_neg ∈ {2.0,3.0,4.0}`         |

`pos_weight` dihitung dari train CSV saja (`neg_count / pos_count` per label) untuk menangani class imbalance. Untuk val/test digunakan `eval_criterion` tanpa `pos_weight` agar loss antar run bisa dibandingkan secara fair.

## Dataset & Augmentasi

**Dataset modes:**

| Mode        | Sumber gambar                              | CSV split                              |
|-------------|--------------------------------------------|----------------------------------------|
| `normal`    | `hasil_label9/cropped_faces/`              | `hasil_label9/Label2d/`                |
| `supercrop` | `hasil_label9/tighter_cropped_faces/`      | `hasil_label9/Label2d-super-crop/`     |
| `faceonly`  | `hasil_label9_faceonly/faceonly_faces/`    | `hasil_label9_faceonly/Label2d/`       |

**Augmentasi train:**
- `RandomResizedCrop` (scale 0.8–1.0, ratio 0.9–1.1)
- Horizontal flip (p=0.5)
- Brightness jitter ±0.2
- Saturation jitter 0.8–1.2 (p=0.5)
- Hue jitter ±0.05 (p=0.3)
- Gaussian blur radius 0.5–1.5 (p=0.2)

Val/test: resize deterministik ke 224×224.

**NumWorkers:** Secara otomatis di-set ke 0 jika `debugpy` atau `pydevd` terdeteksi di sys.modules (VSCode debugger konflik dengan `os.fork()`).

## Threshold Per-Label

Setelah setiap epoch val, set val dibagi dua dengan stratifikasi rarity label (sampel langka dibagi merata ke kedua separuh). Separuh pertama dipakai untuk grid-search threshold (0.1–0.9, step 0.01) yang memaksimalkan F1 per label. Separuh kedua dipakai untuk evaluasi final agar tidak ada leakage threshold→F1.

## Visualisasi (GradCAM)

`gradcam.py` mengimplementasikan dua kaptur:

**`_GradCAMCapture`:** Hook pada `model.pool` (input ke AdaptiveAvgPool2d). Forward hook menangkap spatial feature map `[B, C, H, W]` dan `retain_grad()` dipanggil. Setelah `backward`, gradient di-average atas spatial (cam weights), lalu dikalikan feature map dan di-ReLU → heatmap `[H, W]`.

**`_StageCapture`:** Hook pada beberapa backbone stage sekaligus. Setelah backward, tiap stage menghasilkan heatmap per-stage untuk panel low→high level.

Output yang disimpan:
- `viz/epoch_XX/gradcam_epXX.png` — grid per label dari val set (setiap 3 epoch)
- `viz/test_gradcam/` — original | GradCAM overlay per sample test
- `viz/stage_viz_*.png` — panel stage-level dari satu sample test

GradCAM per-epoch tidak di-upload ke W&B untuk menghemat storage. Hanya `stage_viz` yang di-log ke W&B di akhir training.

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

Mode `compare` menambah `dataset_mode ∈ {normal, supercrop, faceonly}`.

Sweep menggunakan metode Bayes dengan early termination Hyperband (min_iter=5, eta=2), target metrik `val/f1_macro`.

Setiap run sweep dijalankan di subprocess terpisah via `_sweep_worker.py` untuk menghindari CUDA state corruption antar run.

## Cara Pakai

```bash
cd training/2d/convnextv2

python train.py --mode normal    --wandb
python train.py --mode supercrop --loss asl --wandb
python train.py --mode faceonly  --wandb

python sweep.py --mode normal    --count 40
python sweep.py --mode compare   --count 60
python sweep.py --mode normal    --sweep-id <existing-id> --count 20
```

## Output Run

```
runs/<mode>/<run_id>/
├── checkpoints/
│   ├── best.pth          # state_dict + thresholds + f1_macro + cfg
│   └── best_epXX.pth     # snapshot epoch 1, setiap 3 epoch, dan epoch terakhir
├── viz/
│   ├── epoch_XX/         # GradCAM grid per epoch val
│   └── test_gradcam/     # original + GradCAM overlay test set
└── metrics.csv           # per-epoch: loss, f1_macro, mean_acc, sub_acc, lr, elapsed
```
