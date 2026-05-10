
import argparse
from pathlib import Path
import pandas as pd
from collections import Counter, defaultdict

from config import (
    DATA_ROOT, FACEONLY_BASE,
    NORMAL_LABEL_DIR, SUPERCROP_LABEL_DIR, FACEONLY_LABEL_DIR,
    LABELS
)


def _resolve(p: str, img_root: Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (img_root / pp)


def _read_csvs(mode: str):
    label_dirs = {
        'normal':    NORMAL_LABEL_DIR,
        'supercrop': SUPERCROP_LABEL_DIR,
        'faceonly':  FACEONLY_LABEL_DIR,
    }
    img_roots = {
        'normal':    DATA_ROOT,
        'supercrop': DATA_ROOT,
        'faceonly':  FACEONLY_BASE,
    }
    if mode not in label_dirs:
        raise SystemExit(f"Unknown mode: {mode}")
    ld, ir = label_dirs[mode], img_roots[mode]
    tr = pd.read_csv(ld / 'train.csv')
    va = pd.read_csv(ld / 'val.csv')
    te = pd.read_csv(ld / 'test.csv')
    return tr, va, te, ir


def _norm_paths(df: pd.DataFrame, img_root: Path):
    paths = df['frame_path'].astype(str).tolist()
    return [str(_resolve(p, img_root).resolve()) for p in paths]


def _video_id(p: str, levels: int = 1) -> str:

    pp = Path(p)
    for _ in range(levels):
        if pp.parent == pp:
            break
        pp = pp.parent
    return pp.name


def describe_splits(tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame):
    def _desc(df):
        n = len(df)
        pos = df[LABELS].sum(0).to_dict()
        return n, {k: int(v) for k, v in pos.items()}
    nt, pt = _desc(tr)
    nv, pv = _desc(va)
    ne, pe = _desc(te)
    print(f"Samples — train: {nt}, val: {nv}, test: {ne}")
    print(f"Positives train: {pt}")
    print(f"Positives val:   {pv}")
    print(f"Positives test:  {pe}")


def check_leakage(mode: str = 'normal', parent_levels: int = 1):
    tr, va, te, img_root = _read_csvs(mode)
    describe_splits(tr, va, te)

    tr_p = _norm_paths(tr, img_root)
    va_p = _norm_paths(va, img_root)
    te_p = _norm_paths(te, img_root)

    tr_set, va_set, te_set = set(tr_p), set(va_p), set(te_p)


    dup_tr_va = tr_set & va_set
    dup_tr_te = tr_set & te_set
    dup_va_te = va_set & te_set

    print("\nExact path overlaps:")
    print(f"- train ∩ val:  {len(dup_tr_va)}")
    print(f"- train ∩ test: {len(dup_tr_te)}")
    print(f"- val   ∩ test: {len(dup_va_te)}")


    tr_vid = [_video_id(p, levels=1) for p in tr_p]
    va_vid = [_video_id(p, levels=1) for p in va_p]
    te_vid = [_video_id(p, levels=1) for p in te_p]

    tr_vid_set, va_vid_set, te_vid_set = set(tr_vid), set(va_vid), set(te_vid)
    print("\nParent-folder overlaps (proxy for video/session):")
    print(f"- train ∩ val:  {len(tr_vid_set & va_vid_set)}")
    print(f"- train ∩ test: {len(tr_vid_set & te_vid_set)}")
    print(f"- val   ∩ test: {len(va_vid_set & te_vid_set)}")


    def top_counter(xs):
        return Counter(xs).most_common(10)
    print("\nTop parent folders:")
    print(f"- train: {top_counter(tr_vid)}")
    print(f"- val:   {top_counter(va_vid)}")
    print(f"- test:  {top_counter(te_vid)}")


    multi_map = defaultdict(set)
    for vid in tr_vid_set:
        multi_map[vid].add('train')
    for vid in va_vid_set:
        multi_map[vid].add('val')
    for vid in te_vid_set:
        multi_map[vid].add('test')
    cross = {k: v for k, v in multi_map.items() if len(v) > 1}
    if cross:
        print("\nWARNING: Detected parent-folder entities across splits:")
        for k, v in list(cross.items())[:20]:
            print(f"  - {k}: {sorted(v)}")
        if len(cross) > 20:
            print(f"  ... and {len(cross)-20} more")
    else:
        print("\nNo parent-folder entity appears in multiple splits.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Quick leakage check for EVA splits')
    ap.add_argument('--mode', choices=['normal', 'supercrop', 'faceonly'], default='normal')
    args = ap.parse_args()
    check_leakage(args.mode)
