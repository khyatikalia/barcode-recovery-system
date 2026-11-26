"""Scaled direct-path consistency + DB guard analysis (Prompt 3c).

Generates fresh barcodes (disjoint from train/val/test and seed-31415 set) until
>=100 wrong direct/enhanced reads or 20,000 barcodes, records damage metadata via
damage_with_info(), splits 50/50 by barcode into cal/eval, selects thresholds on
cal only, and evaluates consistency + DB guard (random & neighbour manifests) on eval.
"""
import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from damage import damage_with_info
from eval.guards_trace import trace_pipeline
from eval.leakage import assert_no_leakage, load_excluded_barcodes, write_run_config
from eval.manifests import generate_neighbour_awbs, generate_random_awbs
from generate_barcodes import render_ean13

DEFAULT_SEED = 98765


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def wilson_ci(k: int, n: int, z: float = 1.96) -> dict:
    if n == 0:
        return {"k": 0, "n": 0, "rate": 0.0, "ci_low": 0.0, "ci_high": 0.0, "low_n_flag": True}
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
        "low_n_flag": n < 30,
    }


def fmt_ci(stat: dict) -> str:
    flag = " [FLAG: n<30]" if stat["low_n_flag"] else ""
    return (
        f"{stat['k']}/{stat['n']} ({100*stat['rate']:.1f}%, "
        f"95% CI [{100*stat['ci_low']:.1f}%, {100*stat['ci_high']:.1f}%]){flag}"
    )


def main():
    ap = argparse.ArgumentParser(description="Scaled direct-path & DB guard analysis.")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--target-wrong", type=int, default=100)
    ap.add_argument("--max-barcodes", type=int, default=20000)
    ap.add_argument("--data", default="data")
    ap.add_argument("--checkpoint", required=True, help="Explicit checkpoint path (no fallback)")
    ap.add_argument("--out", default="results/direct_path_guard_v2.csv")
    args = ap.parse_args()

    if args.seed in (0, 42, 31415):
        raise SystemExit(f"Error: --seed must not be 0, 42, or 31415 (got {args.seed})")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"Error: checkpoint not found: {ckpt_path}")
    ckpt_sha = sha256_file(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    ckpt_thr = float(ckpt["min_consistency"])
    print(f"Checkpoint: {ckpt_path} | sha256={ckpt_sha} | min_consistency={ckpt_thr:.4f}")

    # Exclude train, val, data/test, AND the earlier seed-31415 set
    base_excluded = load_excluded_barcodes(args.data)
    prior_31415 = set(generate_random_awbs(1000, 31415, base_excluded))
    excluded = base_excluded | prior_31415

    # Pre-generate candidate barcodes (up to max_barcodes) and verify no leakage
    candidates = generate_random_awbs(args.max_barcodes, args.seed, excluded)
    assert_no_leakage(candidates, data_dir=args.data, excluded=excluded)

    np_rng = np.random.default_rng(args.seed)
    rows = []
    wrong_count = 0
    total_processed = 0

    for truth in candidates:
        total_processed += 1
        clean = render_ean13(truth[:12], size=(384, 128))
        dmg, op_names, sev = damage_with_info(clean, np_rng, torn=False)
        trace = trace_pipeline(dmg, model=None, device="cpu", db_manifest=None)
        if trace["decode_path"] in ("direct", "enhanced"):
            decoded = trace["candidate"]
            is_correct = decoded == truth
            if not is_correct:
                wrong_count += 1
            rows.append({
                "truth": truth,
                "decoded_value": decoded,
                "correct": is_correct,
                "decode_path": trace["decode_path"],
                "side_consistency_score": round(float(trace["side_consistency_score"]), 6),
                "num_damages": len(op_names),
                "damage_types": "|".join(op_names),
                "severity": sev,
            })
        if wrong_count >= args.target_wrong:
            break

    reached_target = wrong_count >= args.target_wrong
    print(
        f"Processed {total_processed} barcodes -> {len(rows)} direct/enhanced reads, "
        f"{wrong_count} wrong reads (target={args.target_wrong}, reached={reached_target})"
    )

    # Split 50/50 by barcode (truth) into cal and eval
    unique_truths = sorted({r["truth"] for r in rows})
    split_rng = random.Random(args.seed + 1)
    split_rng.shuffle(unique_truths)
    mid = len(unique_truths) // 2
    cal_truths = set(unique_truths[:mid])
    eval_truths = set(unique_truths[mid:])
    assert len(cal_truths & eval_truths) == 0

    for r in rows:
        r["split"] = "cal" if r["truth"] in cal_truths else "eval"

    cal_rows = [r for r in rows if r["split"] == "cal"]
    eval_rows = [r for r in rows if r["split"] == "eval"]

    # Choose candidate thresholds on calibration ONLY
    cal_correct = [r for r in cal_rows if r["correct"]]
    cal_wrong = [r for r in cal_rows if not r["correct"]]
    grid = sorted({0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, round(ckpt_thr, 4)})
    cal_curve = []
    for thr in grid:
        wb = sum(1 for r in cal_wrong if r["side_consistency_score"] < thr)
        cb = sum(1 for r in cal_correct if r["side_consistency_score"] < thr)
        cal_curve.append({
            "threshold": thr,
            "wrong_blocked": wilson_ci(wb, len(cal_wrong)),
            "correct_wrongly_blocked": wilson_ci(cb, len(cal_correct)),
        })

    # Select: (a) lowest grid thr blocking >=90% cal wrong, (b) lowest thr blocking 100% cal wrong
    thr_90 = next((c["threshold"] for c in cal_curve if c["wrong_blocked"]["rate"] >= 0.90), grid[-1])
    thr_100 = next((c["threshold"] for c in cal_curve if c["wrong_blocked"]["rate"] >= 1.0), grid[-1])
    chosen_thresholds = sorted({ckpt_thr, thr_90, thr_100})

    # Evaluate on eval ONLY (grouped by barcode: 1 row per barcode here)
    eval_correct = [r for r in eval_rows if r["correct"]]
    eval_wrong = [r for r in eval_rows if not r["correct"]]

    eval_threshold_results = []
    for thr in chosen_thresholds:
        wb = sum(1 for r in eval_wrong if r["side_consistency_score"] < thr)
        cb = sum(1 for r in eval_correct if r["side_consistency_score"] < thr)
        eval_threshold_results.append({
            "threshold": thr,
            "wrong_blocked": wilson_ci(wb, len(eval_wrong)),
            "correct_wrongly_blocked": wilson_ci(cb, len(eval_correct)),
        })

    # DB guard step on eval reads with both manifest modes (random & neighbour)
    eval_truth_list = sorted(eval_truths)
    used_all = excluded | set(candidates[:total_processed])
    random_decoys = set(generate_random_awbs(1000, args.seed + 2, used_all))
    random_manifest = eval_truths | random_decoys

    neighbour_pairs = generate_neighbour_awbs(eval_truth_list, excluded=used_all)
    neighbour_manifest = eval_truths | {awb for awb, _ in neighbour_pairs}

    db_results = {}
    for mode_name, manifest_set in (("random", random_manifest), ("neighbour", neighbour_manifest)):
        # Without consistency guard
        wa_off = sum(1 for r in eval_wrong if r["decoded_value"] in manifest_set)
        ca_off = sum(1 for r in eval_correct if r["decoded_value"] in manifest_set)
        mode_res = {
            "manifest_size": len(manifest_set),
            "consistency_off": {
                "wrong_accepted": wilson_ci(wa_off, len(eval_rows)),
                "wrong_accepted_of_wrong": wilson_ci(wa_off, len(eval_wrong)),
                "correct_accepted": wilson_ci(ca_off, len(eval_correct)),
            },
            "consistency_on": {},
        }
        for thr in chosen_thresholds:
            wa_on = sum(
                1 for r in eval_wrong
                if r["decoded_value"] in manifest_set and r["side_consistency_score"] >= thr
            )
            ca_on = sum(
                1 for r in eval_correct
                if r["decoded_value"] in manifest_set and r["side_consistency_score"] >= thr
            )
            mode_res["consistency_on"][str(thr)] = {
                "wrong_accepted": wilson_ci(wa_on, len(eval_rows)),
                "wrong_accepted_of_wrong": wilson_ci(wa_on, len(eval_wrong)),
                "correct_accepted": wilson_ci(ca_on, len(eval_correct)),
            }
        db_results[mode_name] = mode_res

    # Breakdown tables on eval by num_damages and by damage_type (grouped by barcode)
    by_num_damages = {}
    dmg_groups = defaultdict(list)
    for r in eval_rows:
        dmg_groups[r["num_damages"]].append(r)
    for k in sorted(dmg_groups):
        grp = dmg_groups[k]
        n_barcodes = len({r["truth"] for r in grp})
        n_wrong = len({r["truth"] for r in grp if not r["correct"]})
        by_num_damages[str(k)] = {
            "wrong_rate": wilson_ci(n_wrong, n_barcodes),
            "correct_rate": wilson_ci(n_barcodes - n_wrong, n_barcodes),
        }

    by_damage_type = {}
    type_groups = defaultdict(list)
    for r in eval_rows:
        for op in r["damage_types"].split("|"):
            if op:
                type_groups[op].append(r)
    for op in sorted(type_groups):
        grp = type_groups[op]
        n_barcodes = len({r["truth"] for r in grp})
        n_wrong = len({r["truth"] for r in grp if not r["correct"]})
        by_damage_type[op] = {
            "wrong_rate": wilson_ci(n_wrong, n_barcodes),
            "correct_rate": wilson_ci(n_barcodes - n_wrong, n_barcodes),
        }

    # Write CSV
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "truth", "decoded_value", "correct", "decode_path", "side_consistency_score",
        "num_damages", "damage_types", "severity", "split",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_min_consistency": ckpt_thr,
        "total_barcodes_processed": total_processed,
        "reached_target_wrong": reached_target,
        "total_reads": len(rows),
        "total_wrong_reads": wrong_count,
        "cal_counts": {"total": len(cal_rows), "correct": len(cal_correct), "wrong": len(cal_wrong)},
        "eval_counts": {"total": len(eval_rows), "correct": len(eval_correct), "wrong": len(eval_wrong)},
        "cal_threshold_curve": cal_curve,
        "selected_thresholds_from_cal": {"ckpt": ckpt_thr, "cal_90pct_wrong": thr_90, "cal_100pct_wrong": thr_100},
        "eval_threshold_results": eval_threshold_results,
        "eval_db_guard_results": db_results,
        "eval_by_num_damages": by_num_damages,
        "eval_by_damage_type": by_damage_type,
    }
    summary_path = out_csv.with_name("direct_path_guard_v2_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    cfg_path = write_run_config(
        out_csv.with_suffix(".run_config.json"),
        seed=args.seed,
        args_dict=vars(args),
        extra={"checkpoint_sha256": ckpt_sha, "total_barcodes_processed": total_processed},
    )

    print(f"\nSaved CSV -> {out_csv}\nSaved Summary -> {summary_path}\nSaved Config -> {cfg_path}")
    print(f"\nCalibration-selected thresholds: ckpt={ckpt_thr:.2f}, cal_90%={thr_90:.2f}, cal_100%={thr_100:.2f}")
    print("\n[EVAL] Thresholds on evaluation split:")
    for item in eval_threshold_results:
        print(
            f"  thr={item['threshold']:.2f} | wrong blocked: {fmt_ci(item['wrong_blocked'])} "
            f"| correct wrongly blocked: {fmt_ci(item['correct_wrongly_blocked'])}"
        )
    print("\n[EVAL] DB Guard Wrong Accepts (of all eval reads):")
    for m in ("random", "neighbour"):
        res = db_results[m]
        print(f"  Mode={m} (manifest={res['manifest_size']}):")
        print(f"    Consistency OFF: wrong_accepted={fmt_ci(res['consistency_off']['wrong_accepted'])}")
        for thr in chosen_thresholds:
            print(
                f"    Consistency ON (thr={thr:.2f}): "
                f"wrong_accepted={fmt_ci(res['consistency_on'][str(thr)]['wrong_accepted'])}"
            )
    print("\n[EVAL] By num_damages (wrong rate):")
    for k, v in by_num_damages.items():
        print(f"  num_damages={k}: {fmt_ci(v['wrong_rate'])}")
    print("\n[EVAL] By damage_type (wrong rate):")
    for k, v in by_damage_type.items():
        print(f"  {k:14s}: {fmt_ci(v['wrong_rate'])}")


if __name__ == "__main__":
    main()
