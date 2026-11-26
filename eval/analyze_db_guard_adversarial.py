"""Adversarial DB guard & wrong-read profile analysis (Prompt 3d).

1. Characterises the 28 wrong direct/enhanced reads in results/direct_path_guard_v2.csv
   -> results/wrong_read_profile.csv
2. Evaluates DB guard on the eval half (n=2608) across manifest modes:
   random, neighbour, sequential, dense_valid_neighbours, missing_10pct
   with consistency OFF, ON@0.60, and ON@0.80.
3. Runs a SYNTHETIC error-injection test across all 5,216 truths using the
   empirical Hamming-distance & position distribution of the 28 wrong reads.
"""
import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

from stdnum import ean

from eval.leakage import load_excluded_barcodes, write_run_config
from eval.manifests import (generate_dense_valid_neighbours,
                            generate_neighbour_awbs, generate_random_awbs,
                            generate_sequential_awbs, validate_awb)

DEFAULT_SEED = 54321


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
        f"{stat['k']}/{stat['n']} ({100*stat['rate']:.2f}%, "
        f"95% CI [{100*stat['ci_low']:.2f}%, {100*stat['ci_high']:.2f}%]){flag}"
    )


def is_transposition(truth: str, dec: str, diff_pos: list) -> bool:
    if len(diff_pos) == 2:
        i, j = diff_pos
        if truth[i] == dec[j] and truth[j] == dec[i]:
            return True
    # Also check if 2 payload positions swapped + check digit changed
    p_diff = [p for p in diff_pos if p < 12]
    if len(p_diff) == 2:
        i, j = p_diff
        if truth[i] == dec[j] and truth[j] == dec[i]:
            return True
    return False


def profile_wrong_reads(rows: list, out_csv: Path) -> list:
    profile = []
    for r in rows:
        if r["correct"] != "False":
            continue
        t = r["truth"]
        d = r["decoded_value"]
        diff_pos = [i for i in range(13) if t[i] != d[i]]
        trans = is_transposition(t, d, diff_pos)
        profile.append({
            "truth": t,
            "decoded_value": d,
            "hamming_distance": len(diff_pos),
            "differing_positions": "|".join(str(p) for p in diff_pos),
            "is_transposition": trans,
            "decode_path": r["decode_path"],
            "side_consistency_score": r["side_consistency_score"],
            "num_damages": r["num_damages"],
            "damage_types": r["damage_types"],
            "split": r["split"],
        })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(profile[0].keys()))
        writer.writeheader()
        writer.writerows(profile)
    return profile


def inject_synthetic_wrong(truth: str, pattern_positions: list, rng: random.Random) -> str:
    """Create a valid-EAN13 synthetic wrong value matching observed differing positions."""
    data_pos = [p for p in pattern_positions if p < 12]
    want_cd_changed = 12 in pattern_positions
    if not data_pos:
        data_pos = [rng.randrange(12)]
        want_cd_changed = True

    for _ in range(50):
        digits = list(truth[:12])
        for p in data_pos:
            choices = [d for d in "0123456789" if d != truth[p]]
            digits[p] = rng.choice(choices)
        p12 = "".join(digits)
        cd = ean.calc_check_digit(p12)
        cand = p12 + cd
        if cand == truth:
            continue
        if want_cd_changed and cd != truth[12]:
            return cand
        if not want_cd_changed and cd == truth[12]:
            return cand

    # Fallback: preserve check digit by solving last data_pos if >=2 data positions
    digits = list(truth[:12])
    for p in data_pos[:-1]:
        choices = [d for d in "0123456789" if d != truth[p]]
        digits[p] = rng.choice(choices)
    if not want_cd_changed and len(data_pos) >= 2:
        last_p = data_pos[-1]
        for d in rng.sample(list("0123456789"), 10):
            if d == truth[last_p]:
                continue
            digits[last_p] = d
            p12 = "".join(digits)
            if ean.calc_check_digit(p12) == truth[12]:
                return p12 + truth[12]

    # Final guaranteed valid-check-digit fallback
    p12 = "".join(digits)
    return p12 + ean.calc_check_digit(p12)


def build_manifests_for_truths(truths: list, excluded: set, seed: int):
    truth_set = set(truths)
    rng = random.Random(seed)

    random_decoys = set(generate_random_awbs(1000, seed + 1, excluded | truth_set))
    m_random = truth_set | random_decoys

    m_neighbour = truth_set | {a for a, _ in generate_neighbour_awbs(truths, excluded)}
    m_seq = truth_set | {a for a, _ in generate_sequential_awbs(truths, excluded, window=50)}
    m_dense = truth_set | {a for a, _ in generate_dense_valid_neighbours(truths, excluded)}

    keep_n = int(round(0.90 * len(truths)))
    kept_truths = set(rng.sample(truths, keep_n))
    m_missing_10 = kept_truths | random_decoys

    manifests = {
        "random": m_random,
        "neighbour": m_neighbour,
        "sequential": m_seq,
        "dense_valid_neighbours": m_dense,
        "missing_10pct": m_missing_10,
    }
    for name, m in manifests.items():
        for awb in m:
            validate_awb(awb)
    return manifests


def main():
    ap = argparse.ArgumentParser(description="Adversarial DB guard & wrong-read profile.")
    ap.add_argument("--in-csv", default="results/direct_path_guard_v2.csv")
    ap.add_argument("--data", default="data")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out-csv", default="results/db_guard_adversarial.csv")
    args = ap.parse_args()

    if args.seed in (0, 42, 31415, 98765):
        raise SystemExit(f"Error: choose a fresh seed (got {args.seed})")

    with open(args.in_csv, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    # 1. Profile the 28 wrong reads
    profile = profile_wrong_reads(all_rows, Path("results/wrong_read_profile.csv"))
    ham_counts = Counter(p["hamming_distance"] for p in profile)
    trans_count = sum(1 for p in profile if p["is_transposition"])
    dmg_counts = Counter(op for p in profile for op in p["damage_types"].split("|") if op)

    print(f"Profiled {len(profile)} wrong reads -> results/wrong_read_profile.csv")
    print(f"  Hamming distances: {dict(sorted(ham_counts.items()))} | transpositions: {trans_count}/{len(profile)}")
    print(f"  Top damage types : {dmg_counts.most_common(5)}")

    # 2. Build manifests on eval half (n=2608) & validate check digits
    eval_rows = [r for r in all_rows if r["split"] == "eval"]
    eval_truths = sorted({r["truth"] for r in eval_rows})
    excluded = load_excluded_barcodes(args.data)

    eval_manifests = build_manifests_for_truths(eval_truths, excluded, args.seed)
    entries_per_truth = {
        name: round(len(m) / len(eval_truths), 2) for name, m in eval_manifests.items()
    }
    print("\nManifest sizes & entries per true AWB (eval half):")
    for name, m in eval_manifests.items():
        print(f"  {name:24s}: {len(m):7d} entries ({entries_per_truth[name]:.2f} per truth)")

    # 3. Evaluate DB guard on eval half with consistency OFF, ON@0.60, ON@0.80
    eval_wrong = [r for r in eval_rows if r["correct"] == "False"]
    eval_correct = [r for r in eval_rows if r["correct"] == "True"]
    thresholds = [("OFF", None), ("ON@0.60", 0.60), ("ON@0.80", 0.80)]

    adv_rows = []
    db_summary = {}
    any_wrong_accept = False
    for mode_name, m_set in eval_manifests.items():
        db_summary[mode_name] = {"manifest_size": len(m_set), "entries_per_truth": entries_per_truth[mode_name], "runs": {}}
        for label, thr in thresholds:
            wa = 0
            cb = 0
            for r in eval_rows:
                score = float(r["side_consistency_score"])
                cons_ok = True if thr is None else (score >= thr)
                in_db = r["decoded_value"] in m_set
                accepted = cons_ok and in_db
                if r["correct"] == "False" and accepted:
                    wa += 1
                    any_wrong_accept = True
                if r["correct"] == "True" and cons_ok and not in_db:
                    cb += 1
                adv_rows.append({
                    "truth": r["truth"],
                    "decoded_value": r["decoded_value"],
                    "correct": r["correct"],
                    "side_consistency_score": r["side_consistency_score"],
                    "manifest_mode": mode_name,
                    "consistency_setting": label,
                    "in_manifest": in_db,
                    "accepted": accepted,
                })
            db_summary[mode_name]["runs"][label] = {
                "wrong_accepts": wilson_ci(wa, len(eval_rows)),
                "correct_blocked_by_db": wilson_ci(cb, len(eval_correct)),
            }

    out_csv = Path(args.out_csv)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(adv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(adv_rows)

    print("\n[EVAL] DB Guard Results (n=2608):")
    for mode_name, info in db_summary.items():
        print(f"  Mode={mode_name}:")
        for label, stats in info["runs"].items():
            print(
                f"    {label:8s} | wrong_accepts={fmt_ci(stats['wrong_accepts'])} "
                f"| correct_blocked_by_db={fmt_ci(stats['correct_blocked_by_db'])}"
            )

    # 4. SYNTHETIC error-injection across all 5,216 truths
    all_truths = sorted({r["truth"] for r in all_rows})
    all_manifests = build_manifests_for_truths(all_truths, excluded, args.seed + 100)
    patterns = [[int(x) for x in p["differing_positions"].split("|")] for p in profile]
    inj_rng = random.Random(args.seed + 200)

    synthetic_injected = []
    for t in all_truths:
        pat = inj_rng.choice(patterns)
        inj = inject_synthetic_wrong(t, pat, inj_rng)
        validate_awb(inj)
        assert inj != t
        synthetic_injected.append((t, inj))

    synthetic_summary = {}
    print("\n[SYNTHETIC ERROR-INJECTION] Across all 5,216 truths (valid check digit only):")
    for mode_name, m_set in all_manifests.items():
        hits = sum(1 for _, inj in synthetic_injected if inj in m_set)
        stat = wilson_ci(hits, len(synthetic_injected))
        synthetic_summary[mode_name] = stat
        print(f"  [SYNTHETIC] {mode_name:24s}: wrong_accept={fmt_ci(stat)}")

    summary_payload = {
        "any_real_wrong_accept": any_wrong_accept,
        "wrong_read_profile_count": len(profile),
        "hamming_distance_counts": dict(sorted(ham_counts.items())),
        "transposition_count": trans_count,
        "eval_db_guard": db_summary,
        "synthetic_error_injection": synthetic_summary,
    }
    summary_path = out_csv.with_name("db_guard_adversarial_summary.json")
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")

    cfg_path = write_run_config(
        out_csv.with_suffix(".run_config.json"),
        seed=args.seed,
        args_dict=vars(args),
        extra={"eval_truths": len(eval_truths), "all_truths_synthetic": len(all_truths)},
    )
    print(f"\nSaved {out_csv}, {summary_path}, {cfg_path}")


if __name__ == "__main__":
    main()
