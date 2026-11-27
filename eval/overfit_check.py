"""Overfitting check using frozen checkpoint models/frozen/unet_v2_final.pt.

For images where plain ZXing fails:
  - Reports U-Net decode rate, wrong rate, and acceptance rate under consistency guard
  - Uses barcode-grouped Wilson intervals (1 row per barcode, not per image)
  - Flags any cell under 30 barcodes

Sets:
  (a) 1,000 images sampled from training split (seed 77777)
  (b) The original validation split (all va barcodes)
  (c) val2: 2,000+ new barcodes, seed 65432, disjoint from train/val/data/test/split manifest
"""
import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

from damage import damage_with_info
from eval.leakage import assert_no_leakage, load_excluded_barcodes, write_run_config
from eval.manifests import generate_random_awbs
from generate_barcodes import render_ean13
from pipeline import decode, enhance, load_model, pick_device, recover, decode_recovered, consistency_score, to_gray

CHECKPOINT = "models/frozen/unet_v2_final.pt"
SPLIT_MANIFEST = "models/unet_v2_split_manifest.csv"
SEED_SAMPLE_TRAIN = 77777
SEED_VAL2 = 65432


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return {"k": 0, "n": 0, "rate": 0.0, "ci_low": 0.0, "ci_high": 0.0, "flag": True}
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    spread = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * (n ** 2))))
    return {
        "k": int(k),
        "n": int(n),
        "rate": round(p, 6),
        "ci_low": round(max(0.0, center - spread), 6),
        "ci_high": round(min(1.0, center + spread), 6),
        "flag": n < 30,
    }


def fmt_ci(s):
    flag = " [FLAG: n<30]" if s["flag"] else ""
    return (
        f"{s['k']}/{s['n']} ({100*s['rate']:.1f}%, "
        f"95% CI [{100*s['ci_low']:.1f}%, {100*s['ci_high']:.1f}%]){flag}"
    )


def eval_set_on_zxing_failures(label, images, model, device, threshold):
    """
    For each (truth, damaged_img), skip those where plain ZXing+enhance already decodes.
    For the remaining (ZXing failures), run U-Net and measure:
      - decode rate: candidate returned (regardless of correctness)
      - wrong rate: candidate != truth
      - acceptance rate: candidate passes consistency guard at given threshold
    Grouped by barcode (1 truth = 1 barcode unit).
    """
    # Collect per-barcode results
    barcode_results = {}  # truth -> {decoded, wrong, accepted}
    for truth, img in images:
        gray = to_gray(img)

        # Try direct + enhanced; skip if ZXing already reads it
        r = decode(gray)
        if r is None:
            enhanced = enhance(gray)
            r = decode(enhanced)
        if r is not None:
            # ZXing success — excluded from this analysis
            continue

        # ZXing failed — run U-Net
        rec = recover(model, gray, device)
        r2 = decode_recovered(rec)

        decoded = r2 is not None
        wrong = decoded and (r2[0] != truth)
        if decoded:
            score = consistency_score(gray, rec, r2[0])
            accepted = score >= threshold
        else:
            score = None
            accepted = False

        # Barcode-level: if multiple images per barcode, any decode counts
        if truth not in barcode_results:
            barcode_results[truth] = {"any_decoded": False, "any_wrong": False, "any_accepted": False, "scores": []}
        if decoded:
            barcode_results[truth]["any_decoded"] = True
        if wrong:
            barcode_results[truth]["any_wrong"] = True
        if accepted:
            barcode_results[truth]["any_accepted"] = True
        if score is not None:
            barcode_results[truth]["scores"].append(score)

    n_barcodes = len(barcode_results)
    n_decoded = sum(1 for v in barcode_results.values() if v["any_decoded"])
    n_wrong = sum(1 for v in barcode_results.values() if v["any_wrong"])
    n_accepted = sum(1 for v in barcode_results.values() if v["any_accepted"])

    decode_stat = wilson_ci(n_decoded, n_barcodes)
    wrong_stat = wilson_ci(n_wrong, n_barcodes)
    accept_stat = wilson_ci(n_accepted, n_barcodes)

    print(f"\n[{label}] ZXing-failure barcodes: {n_barcodes}")
    print(f"  U-Net decode rate : {fmt_ci(decode_stat)}")
    print(f"  Wrong rate        : {fmt_ci(wrong_stat)}")
    print(f"  Acceptance rate   : {fmt_ci(accept_stat)}")

    return {
        "label": label,
        "n_barcodes": n_barcodes,
        "unet_decode_rate": decode_stat,
        "wrong_rate": wrong_stat,
        "acceptance_rate": accept_stat,
    }, barcode_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out-csv", default="results/overfit_check.csv")
    args = ap.parse_args()

    ckpt_path = Path(CHECKPOINT)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    ckpt_sha = sha256_file(ckpt_path)
    device = "cpu"
    model = load_model(str(ckpt_path), device)
    threshold = model.min_consistency
    print(f"Checkpoint: {ckpt_path} sha256={ckpt_sha[:16]}... min_consistency={threshold:.6f}")

    # Read split manifest
    manifest_rows = list(csv.DictReader(open(SPLIT_MANIFEST)))
    train_stems = {r["stem"] for r in manifest_rows if r["split"] == "train"}
    val_stems = {r["stem"] for r in manifest_rows if r["split"] == "val"}
    test_stems = {r["stem"] for r in manifest_rows if r["split"] == "test"}
    all_manifest_stems = {r["stem"] for r in manifest_rows}
    print(f"Split manifest: train={len(train_stems)}, val={len(val_stems)}, test={len(test_stems)}")

    # --- SET (a): 1,000 training-split images ---
    rng_a = random.Random(SEED_SAMPLE_TRAIN)
    sample_stems_a = rng_a.sample(sorted(train_stems), 1000)
    np_rng_a = np.random.default_rng(SEED_SAMPLE_TRAIN)
    images_a = []
    data_root = Path(args.data) / "train" / "damaged"
    for stem in sample_stems_a:
        # Use the first variant on disk
        candidates = sorted(data_root.glob(f"{stem}_v*.png"))
        if candidates:
            img = cv2.imread(str(candidates[0]))
            if img is not None:
                images_a.append((stem, img))
    print(f"Set (a): loaded {len(images_a)} training-split images")

    # --- SET (b): full validation split ---
    val_data_root = Path(args.data) / "train" / "damaged"
    images_b = []
    for stem in sorted(val_stems):
        candidates = sorted(val_data_root.glob(f"{stem}_v*.png"))
        if candidates:
            img = cv2.imread(str(candidates[0]))
            if img is not None:
                images_b.append((stem, img))
    print(f"Set (b): loaded {len(images_b)} validation-split images")

    # --- SET (c): val2 — 2,000 new barcodes ---
    if SEED_VAL2 in (0, 42, 31415, 98765):
        raise SystemExit("SEED_VAL2 is disallowed")
    base_excluded = load_excluded_barcodes(args.data)
    all_excluded = base_excluded | all_manifest_stems
    val2_barcodes = generate_random_awbs(2000, SEED_VAL2, all_excluded)
    assert_no_leakage(val2_barcodes, data_dir=args.data, excluded=all_excluded)
    print(f"Set (c) val2: {len(val2_barcodes)} barcodes, no leakage with train/val/test/manifest")

    val2_dir = Path("data/val2")
    val2_dir.mkdir(parents=True, exist_ok=True)
    np_rng_c = np.random.default_rng(SEED_VAL2)
    images_c = []
    for barcode in val2_barcodes:
        clean = render_ean13(barcode[:12], size=(384, 128))
        dmg, _, _ = damage_with_info(clean, np_rng_c, torn=False)
        fpath = val2_dir / f"{barcode}_v0.png"
        cv2.imwrite(str(fpath), dmg)
        images_c.append((barcode, dmg))
    # Make val2 read-only
    import subprocess
    subprocess.run(["chmod", "-R", "a-w", str(val2_dir)], check=True)
    print(f"Saved and locked {len(images_c)} val2 images -> {val2_dir}")

    # --- Run all three sets ---
    result_a, br_a = eval_set_on_zxing_failures("(a) train-sample-1000", images_a, model, device, threshold)
    result_b, br_b = eval_set_on_zxing_failures("(b) val-split", images_b, model, device, threshold)
    result_c, br_c = eval_set_on_zxing_failures("(c) val2-fresh-2000", images_c, model, device, threshold)

    # Write CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "set_label", "n_barcodes",
            "unet_decode_k", "unet_decode_n", "unet_decode_rate", "unet_decode_ci_low", "unet_decode_ci_high", "unet_decode_flag",
            "wrong_k", "wrong_n", "wrong_rate", "wrong_ci_low", "wrong_ci_high", "wrong_flag",
            "accept_k", "accept_n", "accept_rate", "accept_ci_low", "accept_ci_high", "accept_flag",
        ])
        writer.writeheader()
        for res in [result_a, result_b, result_c]:
            d, w, ac = res["unet_decode_rate"], res["wrong_rate"], res["acceptance_rate"]
            writer.writerow({
                "set_label": res["label"],
                "n_barcodes": res["n_barcodes"],
                "unet_decode_k": d["k"], "unet_decode_n": d["n"], "unet_decode_rate": d["rate"],
                "unet_decode_ci_low": d["ci_low"], "unet_decode_ci_high": d["ci_high"], "unet_decode_flag": d["flag"],
                "wrong_k": w["k"], "wrong_n": w["n"], "wrong_rate": w["rate"],
                "wrong_ci_low": w["ci_low"], "wrong_ci_high": w["ci_high"], "wrong_flag": w["flag"],
                "accept_k": ac["k"], "accept_n": ac["n"], "accept_rate": ac["rate"],
                "accept_ci_low": ac["ci_low"], "accept_ci_high": ac["ci_high"], "accept_flag": ac["flag"],
            })
    print(f"\nWrote {out_csv}")

    summary = {
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "min_consistency_threshold": threshold,
        "results": [result_a, result_b, result_c],
    }
    cfg = write_run_config(
        out_csv.with_suffix(".run_config.json"),
        seed=SEED_VAL2,
        args_dict=vars(args),
        extra={"checkpoint_sha256": ckpt_sha, "threshold": threshold},
    )
    summary_path = out_csv.with_name("overfit_check_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}, {cfg}")


if __name__ == "__main__":
    main()
