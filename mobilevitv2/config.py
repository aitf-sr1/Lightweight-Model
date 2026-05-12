import os, warnings
os.environ["GLOG_minloglevel"]      = "3"
os.environ["GLOG_logtostderr"]      = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["GRPC_VERBOSITY"]        = "ERROR"
os.environ["ABSL_MIN_LOG_LEVEL"]    = "3"
warnings.filterwarnings("ignore")

from pathlib import Path
import torch

LABELS   = ['Boredom', 'Engagement', 'Confusion', 'Frustration']
N_LABELS = len(LABELS)

TIMM_MODELS = {
    'medium': 'mobilevitv2_150',
    'small':  'mobilevitv2_100',
}
IMG_SIZE   = 224
MODEL_SIZE = 'small'

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

BATCH_SIZE    = 16
ACCUM_STEPS   = 8
NUM_WORKERS   = 6
PIN_MEMORY    = True
LR            = 3e-4
LR_HEAD_MULT  = 10
WEIGHT_DECAY  = 1e-4
NUM_EPOCHS    = 25
FREEZE_UNTIL  = 3
WARMUP_EPOCHS = 3
GRAD_CLIP     = 1.0
LABEL_SMOOTH  = 0.05
FOCAL_ALPHA   = 0.5
FOCAL_GAMMA   = 2.0
ASL_GAMMA_POS = 1.0
ASL_GAMMA_NEG = 3.0
LOSS_TYPE     = 'focal'
DROPOUT       = 0.4
THRESHOLD_SEARCH_FRAC = 0.5
SEED          = 42

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DATA_ROOT = Path('/home/khai/AITF_PIPELINE/Anotasi Data/hasil_label9')

NORMAL_LABEL_DIR    = DATA_ROOT / 'Label2d'
SUPERCROP_LABEL_DIR = DATA_ROOT / 'Label2d-super-crop'

FACEONLY_BASE      = Path('/home/khai/AITF_PIPELINE/Anotasi Data/hasil_label9_faceonly')
FACEONLY_LABEL_DIR = FACEONLY_BASE / 'Label2d'

WANDB_PROJECT = "mobilevitv2"
WANDB_ENTITY  = None

SWEEP_PARAMETERS = {
    "loss_type":     {"values": ["bce", "asl"]},
    "lr":            {"distribution": "log_uniform_values", "min": 5e-6, "max": 5e-4},
    "lr_head_mult":  {"values": [5, 10, 20]},
    "dropout":       {"values": [0.3, 0.4, 0.5]},
    "weight_decay":  {"values": [1e-5, 1e-4, 1e-3]},
    "label_smooth":  {"values": [0.0, 0.05, 0.1]},
    "focal_gamma":   {"values": [1.5, 2.0, 2.5]},
    "asl_gamma_neg": {"values": [2.0, 3.0, 4.0]},
}

SWEEP_PARAMETERS_COMPARE = {
    **SWEEP_PARAMETERS,
    "dataset_mode": {"values": ["normal", "supercrop", "faceonly"]},
}
