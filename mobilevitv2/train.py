
import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
_HERE = Path(__file__).parent

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler

from config import (LABELS, N_LABELS, SEED, DEVICE, BATCH_SIZE, ACCUM_STEPS,
                    NUM_EPOCHS, FREEZE_UNTIL, WARMUP_EPOCHS, GRAD_CLIP,
                    FOCAL_ALPHA, LR, LR_HEAD_MULT, WEIGHT_DECAY,
                    DATA_ROOT, FACEONLY_BASE,
                    NORMAL_LABEL_DIR, SUPERCROP_LABEL_DIR, FACEONLY_LABEL_DIR,
                    WANDB_PROJECT, WANDB_ENTITY, LOSS_TYPE, MODEL_SIZE,
                    ASL_GAMMA_NEG)
from dataset import make_loaders, compute_pos_weights
from model import ImageModel, count_params
from loss import (make_criterion, EpochResult, MetricsResult,
                  compute_metrics, find_best_thresholds, split_val_for_threshold)
from gradcam import (show_gradcam_grid, show_test_gradcam_combined, show_stage_viz)

def set_seed(s: int = SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

class CsvMetricsLogger:
    _FIELDS = [
        'epoch', 'mode', 'lr', 'elapsed',
        'train_loss', 'train_f1', 'train_mean_acc', 'train_sub_acc',
        'val_loss',   'val_f1',   'val_mean_acc',   'val_sub_acc']

    def __init__(self, path: Path, resume: bool = False):
        self.path = path
        if resume and path.exists():
            return
        with open(path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=self._FIELDS).writeheader()

    def write(self, **kw):
        row = {k: kw.get(k, '') for k in self._FIELDS}
        with open(self.path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=self._FIELDS).writerow(row)

def run_epoch(model, loader, criterion, optimizer, scaler,
              device, is_train, thresholds=None):
    if is_train:
        model.train()
    else:
        model.eval()

    ac_device  = 'cuda' if device.startswith('cuda') else 'cpu'
    ac_enabled = device.startswith('cuda')
    total_loss, all_probs, all_tgts = 0.0, [], []

    with torch.set_grad_enabled(is_train):
        for step, (images, labels) in enumerate(
                tqdm(loader, desc='Train' if is_train else 'Val  ', leave=False)):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            ctx = (torch.autocast(device_type=ac_device, enabled=ac_enabled)
                   if hasattr(torch, 'autocast')
                   else torch.cuda.amp.autocast(enabled=ac_enabled))

            with ctx:
                logits = model(images)
                loss   = criterion(logits, labels)

            if is_train:
                scaled = loss / ACCUM_STEPS
                (scaler.scale(scaled).backward()
                 if scaler is not None else scaled.backward())

                if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(loader):
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * images.size(0)
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_tgts.append(labels.detach().cpu().numpy())

    all_probs = np.concatenate(all_probs) if all_probs else np.zeros((0, N_LABELS))
    all_tgts  = np.concatenate(all_tgts)  if all_tgts  else np.zeros((0, N_LABELS))
    avg_loss  = total_loss / max(1, len(loader.dataset))

    m = (compute_metrics(all_tgts, all_probs, thresholds)
         if len(all_probs)
         else MetricsResult(0.0, [0.0]*N_LABELS, [0.5]*N_LABELS,
                            0.0, 0.0, [0.0]*N_LABELS))

    return EpochResult(loss=avg_loss, f1_macro=m.f1_macro, f1_each=m.f1_each,
                       aucs=m.aucs, subset_acc=m.subset_acc,
                       mean_acc=m.mean_acc, per_label_acc=m.per_label_acc,
                       probs=all_probs, targets=all_tgts)

def train_and_eval(cfg: dict, wandb_run=None):
    set_seed(SEED)

    lr           = float(cfg.get('lr',           LR))
    lr_head_mult = float(cfg.get('lr_head_mult', LR_HEAD_MULT))
    dropout      = float(cfg.get('dropout',      0.4))
    weight_decay = float(cfg.get('weight_decay', WEIGHT_DECAY))
    label_smooth = float(cfg.get('label_smooth', 0.05))
    focal_gamma  = float(cfg.get('focal_gamma',  2.0))
    asl_gamma_neg= float(cfg.get('asl_gamma_neg', ASL_GAMMA_NEG))
    batch_size   = int(cfg.get('batch_size',     BATCH_SIZE))
    dataset_mode = str(cfg.get('dataset_mode',   'normal'))
    loss_type    = str(cfg.get('loss_type',      LOSS_TYPE))
    model_size   = str(cfg.get('model_size',     MODEL_SIZE))
    num_epochs   = int(cfg.get('num_epochs',     NUM_EPOCHS))

    _LABEL_DIRS = {
        'normal':    NORMAL_LABEL_DIR,
        'supercrop': SUPERCROP_LABEL_DIR,
        'faceonly':  FACEONLY_LABEL_DIR,
    }
    _IMG_ROOTS = {
        'normal':    DATA_ROOT,
        'supercrop': DATA_ROOT,
        'faceonly':  FACEONLY_BASE,
    }
    label_dir = _LABEL_DIRS.get(dataset_mode, NORMAL_LABEL_DIR)
    img_root  = _IMG_ROOTS.get(dataset_mode, DATA_ROOT)
    train_csv = label_dir / 'train.csv'
    val_csv   = label_dir / 'val.csv'
    test_csv  = label_dir / 'test.csv'

    run_id   = (wandb_run.id if wandb_run else 'local')
    out_dir  = _HERE / 'runs' / dataset_mode / run_id
    ckpt_dir = out_dir / 'checkpoints'
    viz_dir  = out_dir / 'viz'
    for d in [ckpt_dir, viz_dir]:
        d.mkdir(parents=True, exist_ok=True)

    ckpt_best = ckpt_dir / 'best.pth'
    ckpt_last = ckpt_dir / 'last.pth'


    resume_ckpt = None
    if ckpt_last.exists():
        try:
            resume_ckpt = torch.load(ckpt_last, weights_only=False, map_location='cpu')
            print(f"[Resume] {ckpt_last} ditemukan — last completed epoch "
                  f"= {resume_ckpt.get('epoch', '?')}")
        except Exception as e:
            print(f"[Resume] gagal load {ckpt_last}: {e} — start fresh")
            resume_ckpt = None

    csv_log = CsvMetricsLogger(out_dir / 'metrics.csv',
                               resume=(resume_ckpt is not None))

    train_loader, val_loader, test_loader = make_loaders(
        train_csv, val_csv, test_csv, batch_size, img_root=img_root)

    model = ImageModel(N_LABELS, dropout=dropout, model_size=model_size).to(DEVICE)
    print(f'Params: {count_params(model):.1f}M  mode={dataset_mode}  '
          f'loss={loss_type}  size={model_size}')

    start_epoch    = (resume_ckpt['epoch'] + 1) if resume_ckpt else 1
    past_freeze    = start_epoch > FREEZE_UNTIL
    transitioned   = past_freeze

    for p in model.backbone.parameters():
        p.requires_grad_(past_freeze)

    bb_params   = list(model.backbone.parameters())
    head_params = list(model.pool.parameters()) + list(model.head.parameters())
    optimizer   = torch.optim.AdamW([
        {'params': bb_params,   'lr': lr},
        {'params': head_params, 'lr': lr * lr_head_mult},
    ], weight_decay=weight_decay)

    from torch.optim.lr_scheduler import ConstantLR, LinearLR, CosineAnnealingLR, SequentialLR
    if past_freeze:
        warmup  = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                            total_iters=WARMUP_EPOCHS)
        cosine  = CosineAnnealingLR(optimizer,
                                     T_max=num_epochs - FREEZE_UNTIL - WARMUP_EPOCHS)
        scheduler = SequentialLR(optimizer, [warmup, cosine],
                                  milestones=[WARMUP_EPOCHS])
    else:
        scheduler = ConstantLR(optimizer, factor=1.0, total_iters=FREEZE_UNTIL)
    scaler    = GradScaler(enabled=(DEVICE == 'cuda'))

    pos_weight = compute_pos_weights(train_csv).to(DEVICE)
    criterion  = make_criterion(loss_type, focal_gamma, asl_gamma_neg,
                                label_smooth, pos_weight).to(DEVICE)

    best_f1         = -1.0
    thresholds      = np.full(N_LABELS, 0.5)
    best_thresholds = np.full(N_LABELS, 0.5)
    mode_str        = 'train' if past_freeze else 'freeze'

    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt['state_dict'])
        try:
            optimizer.load_state_dict(resume_ckpt['optimizer'])
            scheduler.load_state_dict(resume_ckpt['scheduler'])
            if scaler is not None and resume_ckpt.get('scaler') is not None:
                scaler.load_state_dict(resume_ckpt['scaler'])
        except Exception as e:
            print(f"[Resume] gagal load opt/sched/scaler ({e}) — pakai state baru")
        best_f1         = float(resume_ckpt.get('best_f1', -1.0))
        thresholds      = np.array(resume_ckpt.get('thresholds',      thresholds))
        best_thresholds = np.array(resume_ckpt.get('best_thresholds', best_thresholds))
        print(f"[Resume] start_epoch={start_epoch}  best_f1={best_f1:.4f}  "
              f"mode={mode_str}")

    print(f"\nEp | Mode     | Loss    | F1     | MeanAcc | LR")
    print('-' * 60)

    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()

        if epoch == FREEZE_UNTIL + 1 and not transitioned:
            for p in model.backbone.parameters():
                p.requires_grad_(True)
            mode_str = 'train'

            warmup  = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                total_iters=WARMUP_EPOCHS)
            cosine  = CosineAnnealingLR(optimizer,
                                         T_max=num_epochs - FREEZE_UNTIL - WARMUP_EPOCHS)
            scheduler = SequentialLR(optimizer, [warmup, cosine],
                                      milestones=[WARMUP_EPOCHS])
            transitioned = True

        tr = run_epoch(model, train_loader, criterion, optimizer, scaler,
                       DEVICE, True, thresholds)
        va = run_epoch(model, val_loader,   criterion, None,      None,
                       DEVICE, False, thresholds)

        s_probs, s_tgts, e_probs, e_tgts = split_val_for_threshold(
            va.probs, va.targets)
        new_thr  = find_best_thresholds(s_tgts, s_probs)
        va_optim = compute_metrics(e_tgts, e_probs, new_thr)

        scheduler.step()
        cur_lr  = optimizer.param_groups[1]['lr']
        elapsed = time.time() - t0

        is_best = va_optim.f1_macro > best_f1
        if is_best:
            best_f1         = va_optim.f1_macro
            best_thresholds = new_thr.copy()
            ckpt_data = dict(epoch=epoch, state_dict=model.state_dict(),
                             thresholds=best_thresholds,
                             f1_macro=va_optim.f1_macro, cfg=cfg)
            torch.save(ckpt_data, ckpt_best)
            if epoch == 1 or epoch % 3 == 0 or epoch == num_epochs:
                torch.save(ckpt_data, ckpt_dir / f'best_ep{epoch:02d}.pth')
            print(f"      [BEST] F1={va_optim.f1_macro:.4f}")

        thresholds = new_thr


        last_data = dict(
            epoch=epoch,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            scaler=(scaler.state_dict() if scaler is not None else None),
            best_f1=best_f1,
            thresholds=thresholds,
            best_thresholds=best_thresholds,
            cfg=cfg,
        )
        torch.save(last_data, ckpt_last)

        print(f"{epoch:02d} | {mode_str:>8} | {tr.loss:.4f} | "
              f"{va_optim.f1_macro:.4f} | {va_optim.mean_acc:.4f} | "
              f"sub={va_optim.subset_acc:.4f} | {cur_lr:.2e}  ({int(elapsed)}s)")
        per_lbl = "  ".join(
            f"{lbl[:3]}:f1={va_optim.f1_each[i]:.3f}/acc={va_optim.per_label_acc[i]:.3f}"
            for i, lbl in enumerate(LABELS))
        print(f"   {per_lbl}")

        csv_log.write(epoch=epoch, mode=mode_str, lr=cur_lr, elapsed=elapsed,
                      train_loss=tr.loss, train_f1=tr.f1_macro,
                      train_mean_acc=tr.mean_acc, train_sub_acc=tr.subset_acc,
                      val_loss=va.loss, val_f1=va_optim.f1_macro,
                      val_mean_acc=va_optim.mean_acc, val_sub_acc=va_optim.subset_acc)

        if wandb_run:
            log = {
                "epoch": epoch, "lr": cur_lr, "mode": mode_str,
                "train/loss":     tr.loss,
                "train/f1_macro": tr.f1_macro,
                "val/loss":       va.loss,
                "val/f1_macro":   va_optim.f1_macro,
                "val/mean_acc":   va_optim.mean_acc,
                "val/sub_acc":    va_optim.subset_acc,
                "val/auc_mean":   float(np.mean(va_optim.aucs)),
                "best":           int(is_best),
            }
            for i, lbl in enumerate(LABELS):
                log[f"val/f1_{lbl}"]  = float(va_optim.f1_each[i])
                log[f"val/acc_{lbl}"] = float(va_optim.per_label_acc[i])
                log[f"val/auc_{lbl}"] = float(va_optim.aucs[i])
                log[f"val/thr_{lbl}"] = float(thresholds[i])
            wandb_run.log(log)

        if epoch == 1 or epoch % 3 == 0 or epoch == num_epochs:
            try:
                df_val = pd.read_csv(val_csv)
                gcam_p = show_gradcam_grid(
                    model, df_val, DEVICE, viz_dir, epoch, img_root=img_root)
                if gcam_p:
                    print(f'      [Viz] tersimpan → {gcam_p}')


            except Exception as e:
                import traceback
                print(f'      [Viz] ERROR: {e}')
                traceback.print_exc()
            optimizer.zero_grad(set_to_none=True)

    if ckpt_best.exists():
        ckpt = torch.load(ckpt_best, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        best_thresholds = ckpt.get('thresholds', best_thresholds)
        print(f"\n[Test] loaded best ckpt (ep {ckpt.get('epoch','?')}, "
              f"F1={ckpt.get('f1_macro',0):.4f})")

    te = run_epoch(model, test_loader, criterion, None, None,
                   DEVICE, False, best_thresholds)

    print(f"\nTest: loss={te.loss:.4f}  F1={te.f1_macro:.4f}  "
          f"AUC={np.mean(te.aucs):.4f}  MeanAcc={te.mean_acc:.4f}")
    for i, lbl in enumerate(LABELS):
        print(f"  {lbl}: F1={te.f1_each[i]:.4f}  "
              f"AUC={te.aucs[i]:.4f}  Acc={te.per_label_acc[i]:.4f}")

    if wandb_run:
        test_log = {
            "test/loss":     te.loss,
            "test/f1_macro": te.f1_macro,
            "test/auc_mean": float(np.mean(te.aucs)),
            "test/mean_acc": te.mean_acc,
            "test/sub_acc":  te.subset_acc,
        }
        for i, lbl in enumerate(LABELS):
            test_log[f"test/f1_{lbl}"]  = float(te.f1_each[i])
            test_log[f"test/auc_{lbl}"] = float(te.aucs[i])
            test_log[f"test/acc_{lbl}"] = float(te.per_label_acc[i])
        wandb_run.log(test_log)
        wandb_run.summary['best_val_f1'] = best_f1
        wandb_run.summary['test_f1']     = te.f1_macro

    try:
        df_test = pd.read_csv(test_csv)
        test_ps = show_test_gradcam_combined(
            model, df_test, DEVICE, viz_dir, sample_count=30, img_root=img_root)
        stage_p = show_stage_viz(
            model, df_test.sample(1, random_state=SEED).iloc[0],
            DEVICE, viz_dir, img_root=img_root)

        if wandb_run:
            import wandb
            if test_ps:
                wandb_run.log({"test_gradcam": [
                    wandb.Image(str(p), caption=p.name) for p in test_ps[:10]]})
            if stage_p:
                wandb_run.log({"stage_viz": wandb.Image(str(stage_p))})
    except Exception as e:
        print(f'Test viz error: {e}')
    model.zero_grad(set_to_none=True)

    import gc
    if wandb_run:
        wandb_run.unwatch(model)
    del train_loader, val_loader, test_loader
    del optimizer, criterion, scaler, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return model, best_thresholds

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train MobileViTV2.')
    parser.add_argument('--mode',       choices=['normal', 'supercrop', 'faceonly'],
                        default='normal')
    parser.add_argument('--loss',       choices=['bce', 'focal', 'asl'], default=LOSS_TYPE)
    parser.add_argument('--model-size', choices=['medium', 'small'], default='medium')
    parser.add_argument('--wandb',      action='store_true')
    args = parser.parse_args()

    default_cfg = {
        'dataset_mode':  args.mode,
        'loss_type':     args.loss,
        'model_size':    args.model_size,
        'lr':            LR,
        'lr_head_mult':  LR_HEAD_MULT,
        'dropout':       0.4,
        'weight_decay':  WEIGHT_DECAY,
        'label_smooth':  0.05,
        'focal_gamma':   2.0,
        'asl_gamma_neg': ASL_GAMMA_NEG,
        'batch_size':    BATCH_SIZE,
    }

    wb_run = None
    if args.wandb:
        import wandb
        wb_run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY,
                            name=f'baseline-{args.mode}-{args.loss}-{args.model_size}',
                            config=default_cfg)

    print(f"[Train] mode={args.mode}  loss={args.loss}  "
          f"size={args.model_size}  wandb={'on' if wb_run else 'off'}")
    model, thresholds = train_and_eval(default_cfg, wb_run)
    print('Selesai. Thresholds terbaik:', thresholds)

    if wb_run:
        wb_run.finish()
