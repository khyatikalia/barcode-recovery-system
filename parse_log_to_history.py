#!/usr/bin/env python3
"""Convert a pasted notebook training log into history.json and history.csv.

Usage:
    python parse_log_to_history.py [log_file] [--out history.json] [--csv history.csv]
    cat log.txt | python parse_log_to_history.py
    python parse_log_to_history.py  (and paste the log interactively, press Ctrl+D when done)
"""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path


def parse_log_text(text: str, base_lr: float = 1e-3, default_total_epochs: int = 30):
    pattern = re.compile(
        r"epoch\s+(\d+)(?:/(\d+))?\s+train\s+([\d.]+)\s+val\s+([\d.]+)\s+val:\s+correct reads\s+([\d.]+)%,\s+wrong reads\s+([\d.]+)%"
    )
    lr_pattern = re.compile(r"lr\s+([\d.eE+-]+)")

    records = []
    # First pass to infer total epochs if present in line format (e.g. 1/30)
    inferred_total = default_total_epochs
    for line in text.splitlines():
        m = pattern.search(line)
        if m and m.group(2):
            inferred_total = int(m.group(2))
            break

    for line in text.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        ep = int(m.group(1))
        tot = int(m.group(2)) if m.group(2) else inferred_total
        tr_loss = float(m.group(3))
        va_loss = float(m.group(4))
        va_read = round(float(m.group(5)) / 100.0, 4)
        va_wrong = round(float(m.group(6)) / 100.0, 4)

        # Check if lr is explicitly printed on the line
        m_lr = lr_pattern.search(line)
        if m_lr:
            lr = float(m_lr.group(1))
            lr_source = "log"
        else:
            # Cosine annealing schedule: lr_t = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * (ep - 1) / T_max))
            # Matching train.py: CosineAnnealingLR(opt, T_max=args.epochs), where eta_min=0.0
            eta_min = 0.0
            lr = round(eta_min + 0.5 * (base_lr - eta_min) * (1.0 + math.cos(math.pi * (ep - 1) / max(tot, 1))), 8)
            lr_source = "derived from cosine schedule, not measured"

        records.append({
            "epoch": ep,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "val_read_rate": va_read,
            "val_wrong_rate": va_wrong,
            "learning_rate": lr,
            "learning_rate_source": lr_source,
        })

    return records


def main():
    parser = argparse.ArgumentParser(description="Parse training log into history.json and history.csv")
    parser.add_argument("log_file", nargs="?", default=None, help="Path to training log file (reads stdin if omitted)")
    parser.add_argument("--out", "-o", default="history.json", help="Path to output history.json (default: history.json)")
    parser.add_argument("--csv", default=None, help="Path to output history.csv (default: history.csv in same dir)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Base learning rate (default: 1e-3)")
    parser.add_argument("--epochs", type=int, default=30, help="Total epochs for LR schedule (default: 30)")
    args = parser.parse_args()

    if args.log_file and Path(args.log_file).is_file():
        text = Path(args.log_file).read_text(encoding="utf-8", errors="replace")
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        print("Paste your training log below (press Ctrl+D or Ctrl+Z on Windows to finish):", file=sys.stderr)
        text = sys.stdin.read()

    records = parse_log_text(text, base_lr=args.lr, default_total_epochs=args.epochs)
    if not records:
        print("Warning: No matching epoch log lines found in input.", file=sys.stderr)
        sys.exit(1)

    out_json_path = Path(args.out)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "provenance": "unseeded, parsed from log",
        "history": records,
        "epochs": records,
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Parsed {len(records)} epochs -> {out_json_path}")

    # Determine CSV path
    csv_path = Path(args.csv) if args.csv else out_json_path.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch", "train_loss", "val_loss", "val_read_rate", "val_wrong_rate",
            "learning_rate", "learning_rate_source"
        ])
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved CSV -> {csv_path}")


if __name__ == "__main__":
    main()
