"""Split clean barcodes into train/test and create damaged versions.

Usage:
    python prepare_dataset.py --src data/clean_all --out data

Output:
    data/train/clean/<value>.png      data/train/damaged/<value>_v<k>.png
    data/test/clean/<value>.png       data/test/damaged/<value>_v0.png
The split is by barcode, so a test barcode is never seen during training.
(train.py later holds out a slice of train for validation; test is only used by evaluate.py.)
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import zxingcpp

from damage import damage

W, H = 384, 128  # model input size (width x height)


def decodes(img, truth):
    r = zxingcpp.read_barcodes(img, formats=zxingcpp.BarcodeFormat.EAN13)
    return bool(r) and r[0].text == truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/clean_all")
    ap.add_argument("--out", default="data")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--train-variants", type=int, default=2,
                    help="damaged versions per clean training image")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    files = sorted(src.glob("*.png"))
    if not files:
        raise SystemExit(f"No PNG files found in {src}")

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(files))
    n_test = int(args.test_frac * len(files))
    splits = {"test": idx[:n_test], "train": idx[n_test:]}

    for name, ids in splits.items():
        if (out / name).exists():
            shutil.rmtree(out / name)
        (out / name / "clean").mkdir(parents=True)
        (out / name / "damaged").mkdir(parents=True)
        variants = args.train_variants if name == "train" else 1
        total = base_ok = skipped = kept = 0
        for i in ids:
            f = files[i]
            gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            clean = cv2.resize(gray, (W, H), interpolation=cv2.INTER_AREA)
            if not decodes(clean, f.stem):
                skipped += 1
                continue
            kept += 1
            cv2.imwrite(str(out / name / "clean" / f"{f.stem}.png"), clean)
            for v in range(variants):
                d = damage(clean, rng)
                cv2.imwrite(str(out / name / "damaged" / f"{f.stem}_v{v}.png"), d)
                total += 1
                base_ok += decodes(d, f.stem)
        rate = 100 * base_ok / max(total, 1)
        print(f"{name:5s}: {kept} barcodes, {total} damaged images | plain ZXing reads "
              f"{rate:.1f}% of them before any recovery (skipped {skipped})")


if __name__ == "__main__":
    main()
