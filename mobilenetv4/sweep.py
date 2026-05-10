import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (WANDB_PROJECT, WANDB_ENTITY,
                    SWEEP_PARAMETERS, SWEEP_PARAMETERS_COMPARE,
                    NUM_EPOCHS, SEED)

_HERE   = Path(__file__).parent
_WORKER = _HERE / "_sweep_worker.py"

def _make_sweep_cfg(name: str, parameters: dict) -> dict:
    return {
        "name":   name,
        "method": "bayes",
        "metric": {"name": "val/f1_macro", "goal": "maximize"},
        "parameters": parameters,
        "early_terminate": {"type": "hyperband", "min_iter": 5, "eta": 2},
    }

_CONFIGS = {
    "normal":    _make_sweep_cfg("mv4-normal-crop",  SWEEP_PARAMETERS),
    "supercrop": _make_sweep_cfg("mv4-super-crop",   SWEEP_PARAMETERS),
    "faceonly":  _make_sweep_cfg("mv4-face-only",    SWEEP_PARAMETERS),
    "compare":   _make_sweep_cfg("mv4-compare-all",  SWEEP_PARAMETERS_COMPARE),
}

def _make_train_fn(fixed_mode: str | None):
    def train_fn():
        import wandb

        with wandb.init(reinit=True) as run:
            cfg    = dict(wandb.config)
            run_id = run.id

        if fixed_mode is not None:
            cfg["dataset_mode"] = fixed_mode
        elif "dataset_mode" not in cfg:
            cfg["dataset_mode"] = "normal"
        cfg["num_epochs"]  = NUM_EPOCHS
        cfg["seed"]        = SEED
        cfg["model_size"]  = "small"

        env = os.environ.copy()
        env["WANDB_RUN_ID"]      = run_id
        env["WANDB_RESUME"]      = "must"
        env["SWEEP_CFG_JSON"]    = json.dumps(cfg)
        env["WANDB_PROJECT"]     = WANDB_PROJECT
        if WANDB_ENTITY:
            env["WANDB_ENTITY"]  = WANDB_ENTITY
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        result = subprocess.run(
            [sys.executable, str(_WORKER)],
            env=env, cwd=str(_HERE),
        )
        if result.returncode != 0:
            print(f"[Sweep] worker exited with code {result.returncode}")

    return train_fn

def main():
    parser = argparse.ArgumentParser(description="WandB sweep MobileNetV4.")
    parser.add_argument("--mode", choices=["normal", "supercrop", "faceonly", "compare"],
                        default="normal")
    parser.add_argument("--count",    type=int, default=40)
    parser.add_argument("--sweep-id", default=None)
    args = parser.parse_args()

    import wandb

    fixed_mode = None if args.mode == "compare" else args.mode

    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"[Sweep] Bergabung ke sweep: {sweep_id}")
    else:
        sweep_cfg = _CONFIGS[args.mode]
        sweep_id  = wandb.sweep(sweep_cfg, project=WANDB_PROJECT, entity=WANDB_ENTITY)
        print(f"[Sweep] '{sweep_cfg['name']}' dibuat → id: {sweep_id}")
        print(f"        Tambah agent: python sweep.py --mode {args.mode} "
              f"--sweep-id {sweep_id}")

    wandb.agent(sweep_id, function=_make_train_fn(fixed_mode),
                project=WANDB_PROJECT, entity=WANDB_ENTITY, count=args.count)

if __name__ == "__main__":
    main()
