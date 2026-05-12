# 2D Models

Klasifikasi level gambar: tiap frame diprediksi 4 emosi secara independen; agregasi per-klip dilakukan saat inferensi terpisah.

## Pipeline

| Folder           | Backbone (timm)                                    | Input  | Param  | Loss default |
|------------------|----------------------------------------------------|--------|--------|--------------|
| `eva`            | `eva02_tiny_patch14_336.mim_in22k_ft_in1k`         | 336×336 | ~5.5M   | focal        |
| `efficientnetv2` | `tf_efficientnetv2_b0.in1k`                        | 192×192 | ~7.1M   | focal        |
| `mobilenetv3`    | `mobilenetv3_small_100.lamb_in1k`                  | 224×224 | ~2.5M | focal        |
| `convnextv2`     | `convnextv2_atto.fcmae_ft_in1k`                    | 224×224 | ~3.4M   | focal        |
| `mobilenetv4`    | `mobilenetv4_conv_small` / `mobilenetv4_hybrid_medium` | 224×224 | ~3.77M atau ~9.72M | focal |

EVA-02 menggunakan OpenCLIP normalization (bukan ImageNet standar) dan attention rollout sebagai visualisasi, bukan GradCAM. Semua CNN lainnya menggunakan GradCAM via hook di `model.pool`.

## Konfigurasi Umum

| Parameter        | Nilai                                            |
|------------------|--------------------------------------------------|
| Effective batch  | 128 (batch × accum_steps)                        |
| LR backbone      | 3e-4                                             |
| LR head          | 3e-3 (backbone × 10)                             |
| Epochs           | 25                                               |
| Freeze epochs    | 3                                                |
| Warmup epochs    | 3 (linear, setelah unfreeze)                     |
| Cosine annealing | sisa epoch setelah warmup                        |
| Grad clip        | 1.0                                              |
| Label smoothing  | 0.05                                             |
| Seed             | 42                                               |

Batch size fisik berbeda per model (eva: 64, efficientnetv2: 32, mobilenetv3: 64, convnextv2: 32, mobilenetv4: 16) karena perbedaan ukuran input dan VRAM.

## Struktur per pipeline

```
<model>/
├── README.md
├── config.py        # konstanta hyperparam + path data
├── dataset.py       # Dataset PyTorch + augmentasi
├── model.py         # model wrapper + classification head
├── loss.py          # BCE / Focal / ASL + metrics & threshold utils
├── train.py         # training loop + test eval
├── sweep.py         # W&B sweep entry-point
├── gradcam.py       # atau attention_viz.py (eva)
└── crop_faces.py    # khusus eva (MediaPipe face crop)
```

## Konvensi Umum

- Output sigmoid (multi-label, 4 kelas)
- Threshold per-label dicari di separuh val; dilaporkan dari separuh evaluasi (hindari leakage)
- `compute_pos_weights` dari train CSV saja; `eval_criterion` tanpa `pos_weight` untuk val/test
- Augmentasi hanya di train; val/test deterministik (resize ke target resolusi)
- W&B: `wandb.watch(log=None)` dan `unwatch()` di akhir run sweep
- Loss function via factory: `make_criterion(loss_type, focal_gamma, asl_gamma_neg, label_smooth, pos_weight)`

Untuk parameter default lengkap dan contoh CLI, lihat README di masing-masing folder model.
