import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as T

from config import (LABELS, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD,
                    DATA_ROOT, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, SEED)

class FrameDataset(Dataset):
    def __init__(self, csv_path: Path, augment: bool = False,
                 img_root: Path = DATA_ROOT):
        self.df       = pd.read_csv(csv_path)
        self.augment  = augment
        self.img_root = img_root
        self.resample = getattr(Image, 'Resampling', Image).BILINEAR

        if 'frame_path' not in self.df.columns:
            raise ValueError(f"Kolom 'frame_path' tidak ada di {csv_path}")
        missing = [l for l in LABELS if l not in self.df.columns]
        if missing:
            raise ValueError(f"Kolom label hilang di {csv_path}: {missing}")

    def __len__(self):
        return len(self.df)

    def _resolve(self, p: str) -> Path:
        p = Path(p)
        return p if p.is_absolute() else self.img_root / p

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        fpath = self._resolve(row['frame_path'])

        try:
            img = Image.open(fpath).convert('RGB')
        except Exception:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE))

        if self.augment:
            ci, cj, ch, cw = T.RandomResizedCrop.get_params(
                img, scale=(0.8, 1.0), ratio=(0.9, 1.1))
            img = TF.resized_crop(img, ci, cj, ch, cw,
                                  (IMG_SIZE, IMG_SIZE), self.resample)

            if random.random() < 0.5:
                img = TF.hflip(img)

            img = TF.adjust_brightness(img, 1.0 + random.uniform(-0.2, 0.2))
            if random.random() < 0.5:
                img = TF.adjust_saturation(img, random.uniform(0.8, 1.2))
            if random.random() < 0.3:
                img = TF.adjust_hue(img, random.uniform(-0.05, 0.05))
            if random.random() < 0.2:
                img = img.filter(
                    ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        else:
            img = img.resize((IMG_SIZE, IMG_SIZE), self.resample)

        tensor = TF.normalize(TF.to_tensor(img), IMAGENET_MEAN, IMAGENET_STD)
        labels = torch.tensor([row[l] for l in LABELS], dtype=torch.float32)
        return tensor, labels

def worker_init_fn(worker_id: int):
    seed = SEED + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def compute_pos_weights(train_csv: Path) -> torch.Tensor:
    df      = pd.read_csv(train_csv)
    pos     = df[LABELS].sum(0).values.astype(float)
    neg     = len(df) - pos
    weights = neg / np.maximum(pos, 1.0)
    print(f"[PosWeight] {dict(zip(LABELS, weights.round(2)))}")
    return torch.tensor(weights, dtype=torch.float32)

def make_loaders(train_csv: Path, val_csv: Path, test_csv: Path,
                 batch_size: int = BATCH_SIZE, img_root: Path = DATA_ROOT):
    g = torch.Generator()
    g.manual_seed(SEED)

    train_ds = FrameDataset(train_csv, augment=True,  img_root=img_root)
    val_ds   = FrameDataset(val_csv,   augment=False, img_root=img_root)
    test_ds  = FrameDataset(test_csv,  augment=False, img_root=img_root)

    import sys
    _n_workers = 0 if ('debugpy' in sys.modules or 'pydevd' in sys.modules) else NUM_WORKERS
    kw_shared = dict(num_workers=_n_workers, pin_memory=PIN_MEMORY and _n_workers > 0,
                     worker_init_fn=worker_init_fn if _n_workers > 0 else None,
                     persistent_workers=(_n_workers > 0))

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, generator=g, **kw_shared)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, **kw_shared)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, **kw_shared)
    return train_loader, val_loader, test_loader
