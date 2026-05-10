
import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

DEFAULT_PAD_SIDE = 0.008
DEFAULT_PAD_TOP  = 0.015
DEFAULT_PAD_BOT  = 0.005

TARGET_SIZE = 336

DATA_ROOT  = Path('/home/khai/AITF_PIPELINE/Anotasi Data/hasil_label9')
NORMAL_DIR = DATA_ROOT / 'Label2d'
OUT_BASE   = Path('/home/khai/AITF_PIPELINE/Anotasi Data/hasil_label9_faceonly')
OUT_LABEL  = OUT_BASE / 'Label2d'

MODEL_DIR  = Path('/home/khai/AITF_PIPELINE/Anotasi Data/models')

DETECTOR_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/"
    "blaze_face_short_range.tflite"
)
DETECTOR_PATH = MODEL_DIR / "blaze_face_short_range.tflite"

def download_model(dest: Path = DETECTOR_PATH) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Download] {dest.name} ...")
    urllib.request.urlretrieve(DETECTOR_URL, dest)
    print(f"[Download] Selesai → {dest}")
    return dest

def make_detector(model_path: Path, min_confidence: float = 0.3):
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    options = mp_vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.IMAGE,
        min_detection_confidence=min_confidence,
    )
    return mp_vision.FaceDetector.create_from_options(options)

def _make_square_crop(x1, y1, x2, y2, W, H):
    bw, bh = x2 - x1, y2 - y1
    side   = max(bw, bh)

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2


    cx = max(side / 2, min(W - side / 2, cx))
    cy = max(side / 2, min(H - side / 2, cy))

    side = min(side, W, H)

    cx1 = int(cx - side / 2)
    cy1 = int(cy - side / 2)
    return cx1, cy1, cx1 + side, cy1 + side

def crop_face_tight(img_pil: Image.Image, detector,
                    pad_side=DEFAULT_PAD_SIDE,
                    pad_top=DEFAULT_PAD_TOP,
                    pad_bot=DEFAULT_PAD_BOT) -> Image.Image:
    import mediapipe as mp

    W, H   = img_pil.size
    img_np = np.array(img_pil)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
    result   = detector.detect(mp_image)

    if result.detections:
        det  = result.detections[0]
        bbox = det.bounding_box


        pw = int(pad_side * bbox.width)
        ph_top = int(pad_top * bbox.height)
        ph_bot = int(pad_bot * bbox.height)

        x1 = max(0, bbox.origin_x - pw)
        y1 = max(0, bbox.origin_y - ph_top)
        x2 = min(W, bbox.origin_x + bbox.width  + pw)
        y2 = min(H, bbox.origin_y + bbox.height + ph_bot)

        x1, y1, x2, y2 = _make_square_crop(x1, y1, x2, y2, W, H)
    else:

        side = min(W, H)
        x1   = (W - side) // 2
        y1   = (H - side) // 2
        x2, y2 = x1 + side, y1 + side

    resample = getattr(Image, 'Resampling', Image).LANCZOS
    return img_pil.crop((x1, y1, x2, y2)).resize(
        (TARGET_SIZE, TARGET_SIZE), resample)

def process_csv(csv_path: Path, out_csv: Path, detector,
                pad_side, pad_top, pad_bot,
                dry_run=False, verbose=False):
    df       = pd.read_csv(csv_path)
    new_rows = []
    skipped  = 0

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=csv_path.name, leave=False):
        rel_path = Path(row['frame_path'])
        src_abs  = DATA_ROOT / rel_path if not rel_path.is_absolute() else rel_path


        rel_str = str(rel_path)
        if rel_str.startswith('cropped_faces/'):
            new_rel = Path('faceonly_faces') / rel_path.relative_to('cropped_faces')
        elif rel_str.startswith('tighter_cropped_faces/'):
            new_rel = Path('faceonly_faces') / rel_path.relative_to('tighter_cropped_faces')
            norm = DATA_ROOT / 'cropped_faces' / rel_path.relative_to('tighter_cropped_faces')
            if norm.exists():
                src_abs = norm
        else:
            new_rel = Path('faceonly_faces') / rel_path

        dst_abs = OUT_BASE / new_rel

        if not src_abs.exists():
            skipped += 1
            continue

        if not dry_run and not dst_abs.exists():
            dst_abs.parent.mkdir(parents=True, exist_ok=True)
            try:
                img = Image.open(src_abs).convert('RGB')
                out = crop_face_tight(img, detector, pad_side, pad_top, pad_bot)
                out.save(dst_abs, quality=92)
            except Exception as e:
                if verbose:
                    print(f'  skip {src_abs.name}: {e}')
                skipped += 1
                continue

        new_row = dict(row)
        new_row['frame_path'] = str(new_rel)
        new_rows.append(new_row)

    if not dry_run:
        OUT_LABEL.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(new_rows).to_csv(out_csv, index=False)

    print(f'  {csv_path.name}: {len(new_rows)} ok, {skipped} skip')
    return len(new_rows)

def main():
    parser = argparse.ArgumentParser(
        description='Buat dataset face-only crop untuk sweep EVA02.')
    parser.add_argument('--padding', type=float, default=None,
                        help='Padding seragam semua sisi (override pad_side/top/bot).')
    parser.add_argument('--pad-side', type=float, default=DEFAULT_PAD_SIDE)
    parser.add_argument('--pad-top',  type=float, default=DEFAULT_PAD_TOP)
    parser.add_argument('--pad-bot',  type=float, default=DEFAULT_PAD_BOT)
    parser.add_argument('--confidence', type=float, default=0.3)
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview tanpa nulis file.')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    pad_side = args.padding if args.padding is not None else args.pad_side
    pad_top  = args.padding if args.padding is not None else args.pad_top
    pad_bot  = args.padding if args.padding is not None else args.pad_bot

    print(f'Padding  — side:{pad_side:.0%}  top:{pad_top:.0%}  bot:{pad_bot:.0%}')
    print(f'Output   — {OUT_BASE}')
    print(f'Dry run  — {args.dry_run}\n')

    model_path = download_model()
    detector   = make_detector(model_path, min_confidence=args.confidence)

    splits = [
        (NORMAL_DIR / 'train.csv', OUT_LABEL / 'train.csv'),
        (NORMAL_DIR / 'val.csv',   OUT_LABEL / 'val.csv'),
        (NORMAL_DIR / 'test.csv',  OUT_LABEL / 'test.csv'),
    ]

    total = 0
    for src_csv, dst_csv in splits:
        total += process_csv(src_csv, dst_csv, detector,
                             pad_side, pad_top, pad_bot,
                             dry_run=args.dry_run, verbose=args.verbose)

    detector.close()
    print(f'\nSelesai: {total} gambar.')
    if not args.dry_run:
        print(f'CSV  → {OUT_LABEL}')
        print(f'Img  → {OUT_BASE}/faceonly_faces/')

if __name__ == '__main__':
    main()
