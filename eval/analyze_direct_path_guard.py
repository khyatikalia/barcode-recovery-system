"""Measure what the consistency guard would do on direct and enhanced reads.

Generates a fresh set of barcodes (disjoint from train/val/test), damages them
with damage.damage(), runs eval/guards_trace.py with model=None, and writes
results/direct_path_guard_analysis.csv + summary + run_config.json.
"""
import argparse
import csv
import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from damage import damage
from eval.guards_trace import trace_pipeline
from eval.leakage import assert_no_leakage, load_excluded_barcodes, write_run_config
from eval.manifests import generate_random_awbs
from generate_barcodes import render_ean13

DEFAULT_SEED = 31415


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    spread = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * (n ** 2))))
    return p, max(0.0, center - spread), min(1.0, center + spread)


def fmt_ci(k: int, n: int) -> str:
    p, lo, hi = wilson_ci(k, n)
    flag = " [FLAG: n<30]" if 0 < n < 30 else (" [FLAG: n=0]" if n == 0 else "")
    return f"{k}/{n} ({100*p:.1f}%, 95% CI [{100*lo:.1f}%, {100*hi:.1f}%]){flag}"


def score_stats(scores: list) -> str:
    if not scores:
        return "n=0 [FLAG: n<30]"
    arr = np.asarray(scores, dtype=np.float64)
    flag = " [FLAG: n<30]" if len(arr) < 30 else ""
    return (
        f"n={len(arr)}{flag} | min={arr.min():.3f}, p10={np.percentile(arr, 10):.3f}, "
        f"median={np.median(arr):.3f}, mean={arr.mean():.3f}, "
        f"p90={np.percentile(arr, 90):.3f}, max={arr.max():.3f}"
    )


def main():
    ap = argparse.ArgumentParser(description="Analyze side consistency score on direct & enhanced reads.")
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--data", default="data")
    ap.add_argument("--checkpoint", required=True, help="Explicit checkpoint path to read min_consistency (no fallback)")
    ap.add_argument("--out", default="results/direct_path_guard_analysis.csv")
    args = ap.parse_args()

    if args.seed in (0, 42):
        raise SystemExit(f"Error: --seed must not be 0 or 42 (got {args.seed})")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"Error: checkpoint not found: {ckpt_path}")
    ckpt_sha = sha256_file(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    ckpt_thr = float(ckpt["min_consistency"])
    print(f"Checkpoint: {ckpt_path} | sha256={ckpt_sha} | min_consistency={ckpt_thr:.4f}")

    excluded = load_excluded_barcodes(args.data)
    barcodes = generate_random_awbs(args.count, args.seed, excluded)
    assert_no_leakage(barcodes, data_dir=args.data, excluded=excluded)

    np_rng = np.random.default_rng(args.seed)
    rows = []
    for truth in barcodes:
        clean = render_ean13(truth[:12], size=(384, 128))
        dmg = damage(clean, np_rng)
        trace = trace_pipeline(dmg, model=None, device="cpu", db_manifest=None)
        if trace["decode_path"] in ("direct", "enhanced"):
            decoded = trace["candidate"]
            rows.append({
                "truth": truth,
                "decoded_value": decoded,
                "correct": decoded == truth,
                "decode_path": trace["decode_path"],
                "side_consistency_score": trace["side_consistency_score"],
            })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["truth", "decoded_value", "correct", "decode_path", "side_consistency_score"],
        )
        writer.writeheader()
        writer.writerows(rows)

    cfg_path = write_run_config(
        out_path.with_suffix(".run_config.json"),
        seed=args.seed,
        args_dict=vars(args),
        extra={
            "checkpoint_sha256": ckpt_sha,
            "checkpoint_min_consistency": ckpt_thr,
            "total_barcodes": len(barcodes),
            "direct_or_enhanced_reads": len(rows),
        },
    )

    correct_rows = [r for r in rows if r["correct"]]
    wrong_rows = [r for r in rows if not r["correct"]]
    direct_rows = [r for r in rows if r["decode_path"] == "direct"]
    enhanced_rows = [r for r in rows if r["decode_path"] == "enhanced"]

    print(f"\nWrote {len(rows)} rows -> {out_path} (config: {cfg_path})")
    print(f"Total fresh barcodes: {len(barcodes)} | Direct+Enhanced reads: {len(rows)} "
          f"(direct={len(direct_rows)}, enhanced={len(enhanced_rows)})")
    print(f"  Correct reads: {fmt_ci(len(correct_rows), len(rows))}")
    print(f"  Wrong reads  : {fmt_ci(len(wrong_rows), len(rows))}")
    print("\nScore distributions:")
    print(f"  Correct : {score_stats([r['side_consistency_score'] for r in correct_rows])}")
    print(f"  Wrong   : {score_stats([r['side_consistency_score'] for r in wrong_rows])}")
    print(f"  Direct  : {score_stats([r['side_consistency_score'] for r in direct_rows])}")
    print(f"  Enhanced: {score_stats([r['side_consistency_score'] for r in enhanced_rows])}")

    thresholds = [("0.50", 0.5), ("0.60", 0.6), ("0.70", 0.7), ("0.80", 0.8), (f"ckpt({ckpt_thr:.2f})", ckpt_thr)]
    print("\nThreshold impact (grouped by barcode):")
    for label, thr in thresholds:
        w_blk = sum(1 for r in wrong_rows if r["side_consistency_score"] < thr)
        c_blk = sum(1 for r in correct_rows if r["side_consistency_score"] < thr)
        print(
            f"  thr={label:<12s} | wrong blocked: {fmt_ci(w_blk, len(wrong_rows))} "
            f"| correct wrongly blocked: {fmt_ci(c_blk, len(correct_rows))}"
        )


if __name__ == "__main__":
    main()
