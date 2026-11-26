"""Generate validated EAN-13 manifests in 'random' or 'neighbour' mode.

Uses parameterised SQL only, validates 13 digits + check digit + uniqueness,
enforces seed != 42 and != 0, and checks against train/val/test leakage.
"""
import argparse
import random
import sqlite3
from pathlib import Path

from stdnum import ean

from eval.leakage import assert_no_leakage, load_excluded_barcodes, write_run_config
from pipeline import PIN_ROUTING_TABLE, check_digit_ok

DEFAULT_SEED = 1337


def validate_awb(awb: str) -> None:
    if not isinstance(awb, str) or len(awb) != 13 or not awb.isdigit():
        raise ValueError(f"Invalid AWB format (must be 13 digits): {awb!r}")
    if not check_digit_ok(awb):
        raise ValueError(f"Invalid EAN-13 check digit: {awb!r}")


def validate_manifest_rows(rows: list) -> None:
    seen = set()
    for awb, *_ in rows:
        validate_awb(awb)
        if awb in seen:
            raise ValueError(f"Duplicate AWB in manifest: {awb}")
        seen.add(awb)


def generate_random_awbs(count: int, seed: int, excluded: set) -> list:
    if seed in (0, 42):
        raise ValueError(f"Seed must differ from 42 and 0, got {seed}")
    rng = random.Random(seed)
    awbs = []
    seen = set()
    while len(awbs) < count:
        p12 = "".join(str(rng.randint(0, 9)) for _ in range(12))
        awb = p12 + ean.calc_check_digit(p12)
        if awb not in seen and awb not in excluded:
            seen.add(awb)
            awbs.append(awb)
    return awbs


def generate_neighbour_awbs(true_awbs: list, excluded: set) -> list:
    """For each true AWB, generate every 1-digit payload neighbour with a valid check digit."""
    rows = []
    seen = set()
    for truth in true_awbs:
        validate_awb(truth)
        for pos in range(12):
            orig_d = truth[pos]
            for d in "0123456789":
                if d == orig_d:
                    continue
                p12 = truth[:pos] + d + truth[pos + 1 : 12]
                cand = p12 + ean.calc_check_digit(p12)
                if cand not in seen and cand not in excluded and cand != truth:
                    seen.add(cand)
                    rows.append((cand, truth))
    return rows


def write_manifest_db(db_path: Path, awb_pairs: list, mode: str, seed: int) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    rng = random.Random(seed)
    pins = [f"{prefix}001" for prefix in sorted(PIN_ROUTING_TABLE.keys())]

    rows = []
    for awb, source_truth in awb_pairs:
        pin = rng.choice(pins)
        rows.append((awb, pin, mode, source_truth))

    validate_manifest_rows(rows)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "CREATE TABLE shipments ("
            "awb TEXT PRIMARY KEY, "
            "consignee_pin TEXT NOT NULL, "
            "mode TEXT NOT NULL, "
            "source_truth TEXT"
            ")"
        )
        conn.executemany(
            "INSERT INTO shipments (awb, consignee_pin, mode, source_truth) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser(description="Build validated EAN-13 evaluation manifests.")
    ap.add_argument("--mode", choices=("random", "neighbour"), required=True)
    ap.add_argument("--count", type=int, default=200, help="Count for random mode or truth count for neighbour mode")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", required=True, help="Output SQLite .db path")
    args = ap.parse_args()

    if args.seed in (0, 42):
        raise SystemExit(f"Error: --seed must not be 0 or 42 (got {args.seed})")

    excluded = load_excluded_barcodes(args.data)
    if args.mode == "random":
        awbs = generate_random_awbs(args.count, args.seed, excluded)
        pairs = [(a, None) for a in awbs]
    else:
        truths = generate_random_awbs(args.count, args.seed, excluded)
        pairs = generate_neighbour_awbs(truths, excluded)

    assert_no_leakage([a for a, _ in pairs], data_dir=args.data, excluded=excluded)
    out_path = Path(args.out)
    write_manifest_db(out_path, pairs, args.mode, args.seed)
    cfg_path = write_run_config(
        out_path.with_suffix(".run_config.json"),
        seed=args.seed,
        args_dict=vars(args),
        extra={"mode": args.mode, "num_shipments": len(pairs)},
    )
    print(f"Wrote {len(pairs)} validated AWBs ({args.mode}) -> {out_path} (config: {cfg_path})")


if __name__ == "__main__":
    main()
