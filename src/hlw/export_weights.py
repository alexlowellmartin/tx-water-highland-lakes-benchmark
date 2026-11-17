"""Export the shipped models to weights/.

Shipped S2 model: seed-0 S2 U-Net of the Travis-held-out fold, i.e. trained on Buchanan, Inks,
LBJ and Marble Falls 2018-2023. (Ensemble members and the Buchanan-held-out fold stay in
data/runs/.) Shipped S1 model: the Travis-held-out S1 U-Net (65 MB, also in weights/).
"""

from __future__ import annotations

import hashlib
import json
import shutil

from . import config as C
from .evaluate import model_path


def main():
    C.WEIGHTS_DIR.mkdir(exist_ok=True)
    manifest = {}
    for sensor, name in (("s2", "hlw_s2_unet_r18.pt"), ("s1", "hlw_s1_unet_r50.pt")):
        src = model_path(sensor, "holdout_travis")
        if not src.exists():
            print("missing", src)
            continue
        dst = C.WEIGHTS_DIR / name
        shutil.copy(src, dst)
        meta = json.loads((src.parent / "meta.json").read_text())
        meta.pop("log", None)
        manifest[name] = {"sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
                          "bytes": dst.stat().st_size, **meta}
        print(dst, f"{dst.stat().st_size / 1e6:.1f} MB")
    (C.WEIGHTS_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
