import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import wandb
from train import train_and_eval

cfg    = json.loads(os.environ["SWEEP_CFG_JSON"])
run_id = os.environ["WANDB_RUN_ID"]
proj   = os.environ.get("WANDB_PROJECT")
ent    = os.environ.get("WANDB_ENTITY") or None

run = wandb.init(id=run_id, resume="must", project=proj, entity=ent)
try:
    train_and_eval(cfg, run)
finally:
    wandb.finish()
