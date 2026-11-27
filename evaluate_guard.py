#!/usr/bin/env python3
"""Evaluate Database Guard & Routing on EAN-13 Barcodes.

Compares pipeline performance with Database Manifest Guard OFF vs ON.
Outputs:
  per_image.csv
  Terminal Summary Table with 95% Wilson Score Confidence Intervals
  Manifest Provenance & Overlap Analysis
"""
import argparse
import glob
import math
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pipeline import check_digit_ok, load_model, pick_device, run_pipeline


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Compute 95% Wilson score confidence interval."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    spread = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * (n ** 2))))
    low = max(0.0, center - spread)
    high = min(1.0, center + spread)
    return p, low, high


def format_stat(k: int, n: int) -> str:
    p, lo, hi = wilson_ci(k, n)
    return f"{k:4d} ({p * 100:5.1f}%)  [95% CI: {lo * 100:5.1f}% - {hi * 100:5.1f}%]"


def load_manifest(manifest_path: str):
    p = Path(manifest_path)
    if not p.exists():
        sys.exit(f"Error: Manifest file not found at {manifest_path}")

    if p.suffix.lower() in (".db", ".sqlite", ".sqlite3"):
        conn = sqlite3.connect(str(p))
        df = pd.read_sql_query("SELECT * FROM shipments", conn)
        conn.close()
    elif p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        sys.exit(f"Unsupported manifest format: {p.suffix}")

    # Standardize awb column
    for col in df.columns:
        if str(col).strip().lower() in ("awb", "barcode", "ean13"):
            df = df.rename(columns={col: "awb"})
            break
    df["awb"] = df["awb"].astype(str).str.strip()
    manifest_dict = {row["awb"]: row.to_dict() for _, row in df.iterrows()}
    return manifest_dict, set(manifest_dict.keys())


def find_images(img_arg: str):
    p = Path(img_arg)
    if p.is_dir():
        files = sorted(list(p.glob("*.png")) + list(p.glob("*.jpg")) + list(p.glob("*.jpeg")))
    else:
        files = sorted([Path(f) for f in glob.glob(img_arg)])
    return files


def main():
    parser = argparse.ArgumentParser(description="Evaluate Database Guard impact on EAN-13 pipeline.")
    parser.add_argument("--images", required=True, help="Directory or glob pattern of test images")
    parser.add_argument("--manifest", default="data/ean13_courier/ean13_manifest.db", help="Path to manifest (.db, .xlsx, .csv)")
    parser.add_argument("--model", default="models/unet.pt", help="Path to UNet weights checkpoint")
    parser.add_argument("--out", default="per_image.csv", help="Output path for per-image CSV")
    parser.add_argument("--allow-no-model", action="store_true", help="Run without AI recovery model (OpenCV/ZXing only)")
    args = parser.parse_args()

    # Model validation
    model = None
    device = "cpu"
    model_path = Path(args.model)
    if not model_path.exists() and (Path("models/unet_v2.pt")).exists():
        model_path = Path("models/unet_v2.pt")

    if model_path.exists():
        device = pick_device()
        print(f"Loading recovery model from {model_path} on {device}...")
        model = load_model(str(model_path), device)
    elif not args.allow_no_model:
        sys.exit(
            f"ERROR: Model checkpoint not found at '{args.model}'.\n"
            f"Please provide valid checkpoint with --model or pass --allow-no-model to evaluate non-AI stages."
        )
    else:
        print("Notice: Running in --allow-no-model mode (AI recovery stage bypassed).")

    # Load Manifest
    manifest_dict, manifest_awbs = load_manifest(args.manifest)

    # Find Images
    img_files = find_images(args.images)
    if not img_files:
        sys.exit(f"No image files matched: {args.images}")

    print(f"Loaded manifest with {len(manifest_awbs)} shipments.")
    print(f"Evaluating {len(img_files)} test images...")

    per_image_rows = []

    # Counters
    off_correct = 0
    off_wrong = 0
    off_review = 0

    on_correct = 0
    on_wrong = 0
    on_review = 0

    blocked_candidates_valid_cd = 0
    test_truths_in_manifest = 0
    total = len(img_files)

    for f in img_files:
        stem = f.stem
        # evaluate truth parsing: stem.rsplit("_v", 1)[0]
        truth = stem.rsplit("_v", 1)[0]
        if truth in manifest_awbs:
            test_truths_in_manifest += 1

        img = cv2.imread(str(f))
        if img is None:
            continue

        # Pipeline Guard OFF
        res_off = run_pipeline(img, model=model, device=device, db_manifest=None)

        # Pipeline Guard ON
        res_on = run_pipeline(img, model=model, device=device, db_manifest=manifest_dict)

        # Off stats
        if res_off["status"] == "accepted":
            if res_off["value"] == truth:
                off_correct += 1
            else:
                off_wrong += 1
        else:
            off_review += 1

        # On stats
        if res_on["status"] == "accepted":
            if res_on["value"] == truth:
                on_correct += 1
            else:
                on_wrong += 1
        else:
            on_review += 1

        # Check for guard intervention on valid check-digit candidate
        cand = res_off.get("candidate") or res_off.get("value")
        if cand and check_digit_ok(cand):
            if res_off["status"] == "accepted" and res_on["status"] == "manual_review":
                blocked_candidates_valid_cd += 1

        per_image_rows.append({
            "file": str(f),
            "truth": truth,
            "damages": "torn" if "torn" in str(f) else ("damaged" if "damaged" in str(f) else "clean"),
            "status_off": res_off["status"],
            "value_off": res_off["value"],
            "candidate_off": res_off["candidate"],
            "source_off": res_off["source"],
            "score_off": res_off["score"],
            "status_on": res_on["status"],
            "value_on": res_on["value"],
            "candidate_on": res_on["candidate"],
            "source_on": res_on["source"],
            "score_on": res_on["score"],
            "db_verified_on": res_on["db_verified"],
        })

    # Save per_image.csv
    df_out = pd.DataFrame(per_image_rows)
    df_out.to_csv(args.out, index=False)
    print(f"Wrote per-image results to {args.out}")

    # Summary Report
    print("\n" + "=" * 80)
    print("PIPELINE SAFETY EVALUATION: DATABASE MANIFEST GUARD (OFF vs ON)")
    print("=" * 80)
    print(f"Total Images Evaluated: {total}")
    print("-" * 80)
    print(f"{'Metric':<30} | {'Guard OFF (Raw Pipeline)':<23} | {'Guard ON (Database Guarded)':<23}")
    print("-" * 80)
    print(f"{'Correct Auto-Accepted':<30} | {format_stat(off_correct, total):<23} | {format_stat(on_correct, total):<23}")
    print(f"{'Wrong Auto-Accepted (Errors)':<30} | {format_stat(off_wrong, total):<23} | {format_stat(on_wrong, total):<23}")
    print(f"{'Sent to Manual Review':<30} | {format_stat(off_review, total):<23} | {format_stat(on_review, total):<23}")
    print("-" * 80)
    print(f"Candidates with valid check digit blocked by Database Guard: {blocked_candidates_valid_cd}")
    print("=" * 80)
    print("\nMANIFEST PROVENANCE & DATASET OVERLAP STATEMENT:")
    print(f"  • Total registered shipments in Database Manifest: {len(manifest_awbs)}")
    print(f"  • Test barcode ground truths present in Database Manifest: {test_truths_in_manifest} / {total} ({test_truths_in_manifest / max(total, 1) * 100:.1f}%)")
    if test_truths_in_manifest == total:
        print("  • Overlap Status: 100% of ground-truth test barcodes are registered in the manifest database.")
    else:
        print(f"  • Overlap Status: Partial overlap ({total - test_truths_in_manifest} test barcodes deliberately outside database to verify guard rejection).")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
