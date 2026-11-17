"""U-Net segmentation models with TorchGeo pretrained encoders.

S2: smp U-Net, ResNet-18 encoder initialised from `ResNet18_Weights.SENTINEL2_ALL_MOCO`
    (SSL4EO-S12 MoCo, 13 L1C bands). L2A has no B10, so a zero channel is inserted.
S1: smp U-Net, ResNet-50 encoder initialised from `ResNet50_Weights.SENTINEL1_ALL_MOCO`
    (SSL4EO-S12 MoCo, VV/VH GRD in dB).
A Dropout2d layer before the head allows MC-dropout uncertainty.
"""

from __future__ import annotations

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch import nn

S2_MEAN, S2_STD = 0.0, 10000.0
S1_MEAN = np.array([-12.59, -20.26], np.float32)
S1_STD = np.array([5.26, 5.91], np.float32)
B10_INDEX = 10  # position of B10 in the 13-band SSL4EO order


def s2_to_input(refl: np.ndarray) -> np.ndarray:
    """(12,H,W) uint16 L2A (S2_BANDS order) -> (13,H,W) float32 normalised, B10 = 0."""
    x = (refl.astype(np.float32) - S2_MEAN) / S2_STD
    return np.insert(x, B10_INDEX, 0.0, axis=0)


def s1_to_input(db: np.ndarray) -> np.ndarray:
    """(2,H,W) dB -> normalised float32, NaN -> 0 (i.e. the mean)."""
    x = (db - S1_MEAN[:, None, None]) / S1_STD[:, None, None]
    return np.nan_to_num(x, nan=0.0).astype(np.float32)


class WaterNet(nn.Module):
    def __init__(self, sensor: str = "s2", init: str = "moco", dropout: float = 0.2):
        super().__init__()
        self.sensor = sensor
        enc, ch = ("resnet18", 13) if sensor == "s2" else ("resnet50", 2)
        self.net = smp.Unet(enc, encoder_weights=None, in_channels=ch, classes=1)
        head = self.net.segmentation_head
        self.net.segmentation_head = nn.Sequential(nn.Dropout2d(dropout), head)
        if init == "moco":
            self.load_pretrained()

    def load_pretrained(self):
        from torchgeo.models import ResNet18_Weights, ResNet50_Weights

        w = ResNet18_Weights.SENTINEL2_ALL_MOCO if self.sensor == "s2" \
            else ResNet50_Weights.SENTINEL1_ALL_MOCO
        sd = w.get_state_dict(progress=False)
        sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
        # smp encoders override load_state_dict (returns None); use the base implementation
        missing, unexpected = nn.Module.load_state_dict(self.net.encoder, sd, strict=False)
        if unexpected or [m for m in missing if not m.startswith("fc.")]:
            raise RuntimeError(f"weight mismatch: missing={missing} unexpected={unexpected}")

    def forward(self, x):
        return self.net(x)[:, 0]

    def enable_mc_dropout(self):
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout2d):
                m.train()


@torch.no_grad()
def predict_tiled(models, x: np.ndarray, tile: int = 512, overlap: int = 64, mc: int = 0,
                  device: str | None = None, batch: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window prediction over a full scene.

    models: list of WaterNet (ensemble) or a single one. If `mc` > 0, each model is sampled
    `mc` times with dropout active. Returns (mean probability, std of probability) (H,W).
    """
    models = models if isinstance(models, (list, tuple)) else [models]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _, H, W = x.shape
    ph, pw = max(H, tile), max(W, tile)
    xp = np.zeros((x.shape[0], ph, pw), np.float32)
    xp[:, :H, :W] = x
    step = tile - overlap
    ys = list(range(0, ph - tile + 1, step)) or [0]
    xs = list(range(0, pw - tile + 1, step)) or [0]
    if ys[-1] != ph - tile:
        ys.append(ph - tile)
    if xs[-1] != pw - tile:
        xs.append(pw - tile)
    win = np.outer(np.hanning(tile) + 1e-3, np.hanning(tile) + 1e-3).astype(np.float32)
    wsum = np.zeros((ph, pw), np.float32)
    coords = [(y, x_) for y in ys for x_ in xs]
    for y, x_ in coords:
        wsum[y:y + tile, x_:x_ + tile] += win
    # Each sample (ensemble member or MC pass) is first blended into one full map; mean/std
    # are then taken across samples (Welford), so tile-overlap disagreement is not counted
    # as model uncertainty.
    mean = np.zeros((ph, pw), np.float64)
    m2 = np.zeros((ph, pw), np.float64)
    n = 0
    for m in models:
        m.to(device)
        if mc:
            m.enable_mc_dropout()
        else:
            m.eval()
        for _ in range(max(mc, 1)):
            acc = np.zeros((ph, pw), np.float32)
            for i in range(0, len(coords), batch):
                cs = coords[i: i + batch]
                xb = torch.from_numpy(np.stack([xp[:, y:y + tile, x_:x_ + tile] for y, x_ in cs]))
                with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                    p = torch.sigmoid(m(xb.to(device)).float()).cpu().numpy()
                for (y, x_), pi in zip(cs, p):
                    acc[y:y + tile, x_:x_ + tile] += pi * win
            sample = acc / wsum
            n += 1
            delta = sample - mean
            mean += delta / n
            m2 += delta * (sample - mean)
    std = np.sqrt(m2 / n) if n > 1 else np.zeros_like(mean)
    return mean[:H, :W].astype(np.float32), std[:H, :W].astype(np.float32)


def save_weights(model: WaterNet, path, meta: dict | None = None, half: bool = True):
    sd = {k: (v.half() if half and v.is_floating_point() else v)
          for k, v in model.state_dict().items()}
    torch.save({"state_dict": sd, "sensor": model.sensor, "meta": meta or {}}, path)


def load_weights(path, device: str = "cpu") -> WaterNet:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = WaterNet(ck["sensor"], init="none")
    m.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
    m.meta = ck.get("meta", {})
    return m.eval()
