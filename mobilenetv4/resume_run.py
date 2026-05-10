"""
resume_run.py
=============

Lanjutin run sweep yang ke-interrupt dari `last.pth` checkpoint.

Workflow:
  1. Sweep crash di tengah jalan (Ctrl+C, OOM, IDE close, dll)
  2. `last.pth` udah ke-save di runs/<mode>/<run_id>/checkpoints/last.pth
  3. Jalankan: `python resume_run.py <run_id>`
  4. Wandb run yang sama di-attach lagi (resume="must"), training lanjut dari
     epoch terakhir + 1 sampai NUM_EPOCHS

Find run_id di wandb dashboard atau di nama folder runs/<mode>/<run_id>/.

Cara pakai
----------
  python resume_run.py cb4gms55
  python resume_run.py cb4gms55 --epochs 30   # override NUM_EPOCHS lebih lama
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))


def find_checkpoint(run_id: str) -> Path | None:
    matches = list((_HERE / 'runs').glob(f'*/{run_id}/checkpoints/last.pth'))
    return matches[0] if matches else None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_id', help='Wandb run id yang mau di-resume.')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override NUM_EPOCHS (kalau mau train lebih lama '
                             'dari config default).')
    args = parser.parse_args()

    last_ckpt = find_checkpoint(args.run_id)
    if last_ckpt is None:
        print(f"❌ last.pth gak ditemukan untuk run {args.run_id}")
        print(f"   Search: {_HERE / 'runs'}/*/{args.run_id}/checkpoints/last.pth")
        sys.exit(1)

    mode = last_ckpt.parts[-4]
    print(f"✓ Found {last_ckpt}")
    print(f"  mode    = {mode}")
    print(f"  run_id  = {args.run_id}")

    ckpt = torch.load(last_ckpt, weights_only=False, map_location='cpu')
    cfg  = dict(ckpt['cfg'])
    print(f"  last ep = {ckpt.get('epoch', '?')}")
    print(f"  best F1 = {ckpt.get('best_f1', 0):.4f}")
    print(f"  cfg     = {cfg}")

    if args.epochs is not None:
        cfg['num_epochs'] = args.epochs
        print(f"  override num_epochs → {args.epochs}")

    from config import WANDB_PROJECT, WANDB_ENTITY

    env = os.environ.copy()
    env['WANDB_RUN_ID']     = args.run_id
    env['WANDB_RESUME']     = 'must'
    env['SWEEP_CFG_JSON']   = json.dumps(cfg)
    env['WANDB_PROJECT']    = WANDB_PROJECT
    if WANDB_ENTITY:
        env['WANDB_ENTITY'] = WANDB_ENTITY
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print(f"\n[Resume] spawn worker untuk run {args.run_id} ...")
    result = subprocess.run(
        [sys.executable, str(_HERE / '_sweep_worker.py')],
        env=env, cwd=str(_HERE),
    )
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
