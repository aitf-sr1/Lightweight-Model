# EVA-02 Tiny — 2D Emotion Classification

Vision Transformer berbasis EVA-02 untuk multi-label emotion classification per frame. EVA-02 menggunakan CLIP-style normalization dan patch size 14, berbeda dari CNN pipeline lainnya.

## Arsitektur

```
Input [B, 3, 336, 336]
  └─ backbone: eva02_tiny_patch14_336.mim_in22k_ft_in1k (timm)
       → vektor fitur [B, feat_dim]  (backbone dengan built-in global pool)
  └─ head: Linear(feat_dim, 256) → ReLU → Dropout → Linear(256, 4)
Output [B, 4] logit (sigmoid diterapkan di loss / saat inferensi)
```

EVA-02 tidak memisahkan pool dari backbone (backbone langsung mengembalikan vektor setelah CLS pooling internal). Karena itu visualisasi memakai **attention rollout** bukan GradCAM berbasis spatial feature map.

`feat_dim` dideteksi via dummy forward pass (lebih reliable daripada `num_features` karena beberapa model menerapkan post-conv yang mengubah dimensi).

## Konfigurasi Default

| Parameter         | Nilai                                            |
|-------------------|-------------------------------------------------|
| Backbone (timm)   | `eva02_tiny_patch14_336.mim_in22k_ft_in1k`      |
| Input resolusi    | **336 × 336** (fixed — hanya menerima 336×336)  |
| Normalisasi       | OpenCLIP-style (BUKAN ImageNet standar)         |
| Mean              | `[0.48145466, 0.4578275, 0.40821073]`           |
| Std               | `[0.26862954, 0.26130258, 0.27577711]`          |
| Batch size        | 64 × accum 2 = effective batch 128              |
| Learning rate     | 3e-4 (backbone) / 3e-3 (head, 10×)              |
| Weight decay      | 1e-4                                            |
| Epochs            | 25                                              |
| Freeze epochs     | 3                                               |
| Warmup epochs     | 3 (linear warmup setelah unfreeze)              |
| Cosine annealing  | sisa epoch setelah warmup                       |
| Grad clip         | 1.0                                             |
| Dropout           | 0.5                                             |
| Label smoothing   | 0.05                                            |
| Loss default      | `focal` (γ=2.0, α=0.5)                          |
| Seed              | 42                                              |

**Penting:** EVA-02 menggunakan normalisasi OpenCLIP-style (bukan ImageNet standar). Nilai ini diperoleh dari `timm.data.resolve_model_data_config(model)`. Menggunakan normalisasi yang salah akan menurunkan performa signifikan.

## Strategi Training

**Fase freeze (epoch 1–3):** Backbone dibekukan, BN di-eval mode untuk menjaga stabilitas. Scheduler: `ConstantLR(factor=1.0)`.

**Fase unfreeze (epoch 4–25):** LR backbone dan head di-reset ke nilai awal (ini penting — tanpa reset, cosine decay dari fase freeze akan mengikis LR head secara tidak terduga). Linear warmup 3 epoch (0.1× → 1.0×), lalu CosineAnnealingLR dengan `eta_min=1e-6`.

**Dua criterion terpisah:** `criterion` dengan `pos_weight` untuk training; `eval_criterion` tanpa `pos_weight` untuk val/test agar val-loss tidak dipengaruhi distribusi train dan bisa dibandingkan antar run secara fair.

**Mixed precision:** `torch.autocast` + `GradScaler` aktif di CUDA.

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

**Augmentasi train:** RandomResizedCrop (scale 0.8–1.0, ratio 0.9–1.1), hflip (p=0.5), brightness ±0.2, saturation (p=0.5), hue (p=0.3), GaussianBlur (p=0.2). Val/test: resize deterministik ke 336×336.

**NumWorkers:** Di-set ke 0 saat `debugpy`/`pydevd` aktif.

## Threshold Per-Label

Val set dibagi dua dengan stratifikasi rarity label. Grid-search threshold (0.1–0.9, step 0.01) di separuh pertama; evaluasi di separuh kedua untuk menghindari leakage.

## Persiapan Dataset Faceonly

Sebelum pertama kali memakai mode `faceonly`, buat dataset face-crop:

```bash
cd training/2d/eva
python crop_faces.py
```

Script ini:
1. Mengunduh model MediaPipe BlazeFace (`blaze_face_short_range.tflite`)
2. Mendeteksi wajah tiap frame dengan confidence threshold 0.3
3. Melakukan tight crop wajah + padding (side: 0.8%, top: 1.5%, bot: 0.5% dari bbox)
4. Resize ke 336×336 (sesuai input EVA-02)
5. Menyimpan gambar ke `hasil_label9_faceonly/faceonly_faces/`
6. Membuat CSV baru di `hasil_label9_faceonly/Label2d/` dengan path yang diperbarui

Jika wajah tidak terdeteksi, fallback ke center-square crop. Untuk frame dari `tighter_cropped_faces/`, sumber gambarnya dialihkan ke `cropped_faces/` yang berukuran 512px (lebih detail).

Opsi:
```bash
python crop_faces.py --padding 0.02          # uniform padding semua sisi
python crop_faces.py --confidence 0.5        # threshold deteksi lebih ketat
python crop_faces.py --dry-run               # preview tanpa nulis file
```

## Cek Split & Leakage

```bash
python check_splits.py --mode normal
```

Memeriksa:
- Jumlah sampel dan positif per split
- Exact path overlap antar split (frame duplikat)
- Parent-folder overlap sebagai proxy video/session ID (mencegah data leakage per subjek)
- Top-10 parent folder per split untuk sense check distribusi

## Visualisasi (Gradient-Weighted Attention Rollout)

`attention_viz.py` mengimplementasikan teknik **gradient-weighted attention rollout** untuk ViT:

**`_GradAttnCapture`:** Hook pada `attn.attn_drop` di setiap block. Menonaktifkan `fused_attn` sementara agar attention matrix `[B, H, N, N]` bisa di-capture. Setelah backward, tiap layer punya tensor attention + gradient-nya.

**`_gradient_rollout`:** Untuk setiap layer, `weighted = relu(grad) * attn` di-average atas heads. Lalu rollout standar: `R = W_L @ W_{L-1} @ ... @ W_1`. Hasilnya adalah CLS→patch attention yang dirata-ratakan atas semua layer, menunjukkan patch mana yang paling relevan terhadap prediksi. Output: grid `[24×24]` (336/14=24 patch per dimensi).

**`_gradient_single_layer`:** GradCAM satu layer — `relu(grad[0, 1:]) * attn[0, 1:]` dari CLS token ke patch tokens.

Output yang disimpan:
- `viz/epoch_XX/attn_epXX.png` — rollout per label dari val set (setiap epoch)
- `viz/test_attention/` — original | rollout overlay test samples
- `viz/layer_viz_*.png` — panel 7 kolom (original + 6 block pilihan: 1, 3, 5, 7, 9, last) — di-upload ke W&B

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
cd training/2d/eva

# Buat dataset faceonly (sekali saja)
python crop_faces.py

# Cek leakage split
python check_splits.py --mode normal

# Training
python train.py --mode normal    --wandb
python train.py --mode supercrop --loss asl --wandb
python train.py --mode faceonly  --wandb

# Sweep
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
│   ├── epoch_XX/         # attention rollout grid per label (val)
│   ├── test_attention/   # original + rollout overlay test samples
│   └── layer_viz_*.png   # panel 7 kolom per-block
└── metrics.csv           # lebih detail dari 2d lain: f1/auc/acc per label, threshold
```
