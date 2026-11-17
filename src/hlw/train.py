"""Train a water-segmentation U-Net on weak labels (S2) or S2-teacher targets (S1).

Example:
    python -m hlw.train --sensor s2 --fold holdout_travis --label scl_jrc --seed 0
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import config as C
from . import selection as S
from .dataset import S1TeacherDataset, S2WeakDataset
from .model import WaterNet, save_weights


def masked_bce(logits, target):
    valid = target != 255
    if valid.sum() == 0:
        return logits.sum() * 0
    t = torch.where(valid, target, torch.zeros_like(target)).float()
    loss = F.binary_cross_entropy_with_logits(logits.float(), t, reduction="none")
    return loss[valid].mean()


def run_name(sensor, fold, label, init, seed):
    return f"{sensor}_{fold}_{label}_{init}_s{seed}"


def train(sensor="s2", fold="holdout_travis", label="scl_jrc", init="moco", seed=0,
          iters=2000, batch=16, lr=3e-4, workers=4, chip=256, quiet=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    name = run_name(sensor, fold, label if sensor == "s2" else "teacher", init, seed)
    out = C.RUNS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    if sensor == "s2":
        scenes = S.training_scenes(fold)
        ds = S2WeakDataset(scenes, label, chip=chip, length=iters * batch, seed=seed)
        info = {"n_scenes": len(scenes)}
    else:
        pairs = S.s1_training_pairs(fold)
        ds = S1TeacherDataset(pairs, fold, chip=chip, length=iters * batch, seed=seed)
        info = {"n_pairs": len(pairs)}
    dl = DataLoader(ds, batch_size=batch, num_workers=workers, pin_memory=True,
                    persistent_workers=workers > 0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = WaterNet(sensor, init=init).to(dev)
    enc = list(model.net.encoder.parameters())
    rest = [p for n, p in model.named_parameters() if not n.startswith("net.encoder.")]
    enc_lr = lr / 3 if init == "moco" else lr
    opt = torch.optim.AdamW([{"params": enc, "lr": enc_lr}, {"params": rest, "lr": lr}],
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[enc_lr, lr], total_steps=iters,
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    t0, log = time.time(), []
    model.train()
    for it, (x, y) in enumerate(dl):
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=dev == "cuda"):
            loss = masked_bce(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if it % 100 == 0 or it == iters - 1:
            log.append({"iter": it, "loss": float(loss), "sec": round(time.time() - t0, 1)})
            if not quiet:
                print(name, log[-1], flush=True)
    meta = {"name": name, "sensor": sensor, "fold": fold, "label": label, "init": init,
            "seed": seed, "iters": iters, "batch": batch, "lr": lr, "chip": chip,
            "train_minutes": round((time.time() - t0) / 60, 2),
            "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", **info}
    save_weights(model, out / "model.pt", meta)
    (out / "meta.json").write_text(json.dumps({**meta, "log": log}, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", default="s2", choices=["s2", "s1"])
    ap.add_argument("--fold", default="holdout_travis")
    ap.add_argument("--label", default="scl_jrc")
    ap.add_argument("--init", default="moco", choices=["moco", "none"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    train(a.sensor, a.fold, a.label, a.init, a.seed, a.iters, a.batch, workers=a.workers)


if __name__ == "__main__":
    main()
