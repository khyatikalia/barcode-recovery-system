"""Train the U-Net.   python train.py --data data --epochs 30 --out models/unet.pt

A slice of the TRAIN barcodes (default 10%) is held out for validation. It is used to
pick the best epoch and to calibrate the anti-hallucination threshold. The test split
is never touched here.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import UNet
from pipeline import (DEFAULT_MIN_SCORE, check_digit_ok, consistency_score,
                      decode_recovered, pick_device)


class PairDataset(Dataset):
    def __init__(self, files, clean_dir):
        self.files, self.clean_dir = files, Path(clean_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        stem = f.stem.rsplit("_v", 1)[0]
        x = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        y = cv2.imread(str(self.clean_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        to_t = lambda a: torch.from_numpy(a).float().div(255).unsqueeze(0)
        return to_t(x), to_t(y), stem


def loss_fn(pred, target, w_focus):
    """L1 over the whole image + extra L1 focused on the barcode (dark bars, dilated)."""
    l1 = F.l1_loss(pred, target)
    mask = F.max_pool2d((target < 0.5).float(), 15, 1, 7)
    focused = (mask * (pred - target).abs()).sum() / mask.sum().clamp(min=1)
    return l1 + w_focus * focused


@torch.no_grad()
def validate(model, loader, device, w_focus):
    model.eval()
    total, n, ok, wrong = 0.0, 0, 0, 0
    for x, y, stems in loader:
        x, y = x.to(device), y.to(device)
        p = model(x)
        total += loss_fn(p, y, w_focus).item() * x.size(0)
        imgs = (p.squeeze(1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        for im, stem in zip(imgs, stems):
            r = decode_recovered(im)
            if r is not None:
                ok += int(r[0] == stem)
                wrong += int(r[0] != stem)
        n += x.size(0)
    return total / n, ok / n, wrong / n


@torch.no_grad()
def calibrate(model, loader, device):
    """Pick the minimum consistency score so (nearly) no wrong value gets auto-accepted."""
    model.eval()
    good, bad = [], []
    for x, y, stems in loader:
        p = model(x.to(device))
        recs = (p.squeeze(1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        ins = (x.squeeze(1).numpy() * 255).round().astype(np.uint8)
        for rec, inp, stem in zip(recs, ins, stems):
            r = decode_recovered(rec)
            if r is None or check_digit_ok(r[0]) is False:
                continue
            s = consistency_score(inp, rec, r[0])
            (good if r[0] == stem else bad).append(s)
    thr = max(bad) + 0.01 if bad else DEFAULT_MIN_SCORE
    thr = float(min(max(thr, 0.5), 0.9))
    kept = np.mean([s >= thr for s in good]) if good else 0.0
    print(f"\nCalibration on validation set: {len(good)} correct and {len(bad)} wrong recovered values")
    print(f"  threshold = {thr:.2f}  -> keeps {100 * kept:.0f}% of the correct ones, "
          f"blocks {sum(s < thr for s in bad)}/{len(bad)} wrong ones")
    return thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=32, help="U-Net width (16 = faster, 32 = default)")
    ap.add_argument("--focus", type=float, default=2.0, help="weight of the focused loss")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="models/unet.pt")
    args = ap.parse_args()

    device = pick_device()
    print("device:", device)
    root = Path(args.data) / "train"
    files = sorted((root / "damaged").glob("*.png"))
    stems = sorted({f.stem.rsplit("_v", 1)[0] for f in files})
    rng = np.random.default_rng(0)
    val_stems = set(rng.choice(stems, size=max(1, int(args.val_frac * len(stems))), replace=False))
    tr_files = [f for f in files if f.stem.rsplit("_v", 1)[0] not in val_stems]
    va_files = [f for f in files if f.stem.rsplit("_v", 1)[0] in val_stems]
    print(f"train images: {len(tr_files)}   validation images: {len(va_files)} (held out from train)")

    tr = DataLoader(PairDataset(tr_files, root / "clean"), batch_size=args.batch,
                    shuffle=True, num_workers=args.workers)
    va = DataLoader(PairDataset(va_files, root / "clean"), batch_size=args.batch,
                    num_workers=args.workers)

    model = UNet(base=args.base).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best_key = (-1.0, -1e9)
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run, n = time.time(), 0.0, 0
        for x, y, _ in tr:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y, args.focus)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()
        vloss, vdec, vwrong = validate(model, va, device, args.focus)
        print(f"epoch {ep:3d}/{args.epochs}  train {run / n:.4f}  val {vloss:.4f}  "
              f"val: correct reads {100 * vdec:.1f}%, wrong reads {100 * vwrong:.1f}%  "
              f"({time.time() - t0:.0f}s)")
        key = (vdec - vwrong, -vloss)  # reward correct reads, punish wrong ones
        if key > best_key:
            best_key = key
            torch.save({"model": model.state_dict(), "base": args.base,
                        "val_decode": vdec, "epoch": ep}, args.out)
            print(f"   saved best -> {args.out}")

    # calibrate the anti-hallucination threshold with the best weights
    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt["model"])
    ckpt["min_consistency"] = calibrate(model, va, device)
    torch.save(ckpt, args.out)
    print(f"Final model saved to {args.out}")


if __name__ == "__main__":
    main()
