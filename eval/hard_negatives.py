"""Generate hard-negative EAN-13 barcode images (1-digit substitution & adjacent transposition).

Splits into calibration ('cal') and evaluation ('eval') sets by truth barcode.
Enforces seed != 42 and != 0, checks against train/val/test leakage, and writes
a run-config JSON next to outputs.
"""
import argparse
import csv
import random
from pathlib import Path

import cv2
from stdnum import ean

from eval.leakage import assert_no_leakage, load_excluded_barcodes, write_run_config
from eval.manifests import generate_random_awbs, validate_awb
from generate_barcodes import render_ean13

DEFAULT_SEED = 2027


def make_one_digit_substitution(truth: str, rng: random.Random) -> str:
    """Return a valid EAN-13 differing from `truth` by 1 payload digit (+ check digit)."""
    validate_awb(truth)
    pos = rng.randrange(12)
    orig = truth[pos]
    choices = [d for d in "0123456789" if d != orig]
    d = rng.choice(choices)
    p12 = truth[:pos] + d + truth[pos + 1 : 12]
    return p12 + ean.calc_check_digit(p12)


def make_adjacent_transposition(truth: str, rng: random.Random) -> str:
    """Return a valid EAN-13 formed by swapping two adjacent digits in payload."""
    validate_awb(truth)
    # Prefer pairs with |a - b| == 5 because swapping them preserves the exact check digit
    exact_pairs = [i for i in range(11) if abs(int(truth[i]) - int(truth[i + 1])) == 5]
    diff_pairs = [i for i in range(11) if truth[i] != truth[i + 1]]
    if not diff_pairs:
        raise ValueError(f"No distinct adjacent digits to transpose in {truth}")
    i = rng.choice(exact_pairs if exact_pairs else diff_pairs)
    p12 = truth[:i] + truth[i + 1] + truth[i] + truth[i + 2 : 12]
    return p12 + ean.calc_check_digit(p12)


def generate_hard_negatives(
    num_barcodes: int,
    seed: int = DEFAULT_SEED,
    cal_frac: float = 0.5,
    data_dir: str = "data",
    out_dir: str = "results/hard_negatives",
):
    if seed in (0, 42):
        raise ValueError(f"Seed must differ from 42 and 0, got {seed}")

    excluded = load_excluded_barcodes(data_dir)
    truths = generate_random_awbs(num_barcodes, seed, excluded)

    rng = random.Random(seed + 1)
    shuffled = list(truths)
    rng.shuffle(shuffled)
    n_cal = max(1, int(round(cal_frac * len(shuffled))))
    cal_set = set(shuffled[:n_cal])

    out_root = Path(out_dir)
    for split in ("cal", "eval"):
        (out_root / split).mkdir(parents=True, exist_ok=True)

    rows = []
    generated_negatives = []
    for truth in truths:
        split = "cal" if truth in cal_set else "eval"
        sub = make_one_digit_substitution(truth, rng)
        trans = make_adjacent_transposition(truth, rng)
        for kind, neg in (("substitution", sub), ("transposition", trans)):
            if neg in excluded:
                raise AssertionError(f"Hard negative {neg} collides with train/val/test")
            generated_negatives.append(neg)
            img = render_ean13(neg[:12], size=(384, 128))
            fname = f"{truth}_{kind}.png"
            fpath = out_root / split / fname
            cv2.imwrite(str(fpath), img)
            rows.append({
                "file": str(fpath),
                "truth": truth,
                "hard_negative": neg,
                "kind": kind,
                "split": split,
            })

    assert_no_leakage(truths + generated_negatives, data_dir=data_dir, excluded=excluded)

    manifest_csv = out_root / "hard_negatives_manifest.csv"
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file", "truth", "hard_negative", "kind", "split"]
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows, manifest_csv


def main():
    ap = argparse.ArgumentParser(description="Render hard-negative EAN-13 images split by barcode.")
    ap.add_argument("--count", type=int, default=40, help="Number of truth barcodes")
    ap.add_argument("--cal-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out-dir", default="results/hard_negatives")
    args = ap.parse_args()

    rows, manifest_csv = generate_hard_negatives(
        num_barcodes=args.count,
        seed=args.seed,
        cal_frac=args.cal_frac,
        data_dir=args.data,
        out_dir=args.out_dir,
    )
    cfg_path = write_run_config(
        Path(args.out_dir) / "run_config.json",
        seed=args.seed,
        args_dict=vars(args),
        extra={"num_images": len(rows)},
    )
    print(f"Wrote {len(rows)} hard-negative images -> {manifest_csv} (config: {cfg_path})")


if __name__ == "__main__":
    main()
