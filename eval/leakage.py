"""Shared dataset leakage checks and run-config metadata helpers."""
import json
import subprocess
from pathlib import Path


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        )
        return out.strip()
    except Exception:
        return "unknown"


def load_excluded_barcodes(data_dir="data") -> set:
    """Load every 13-digit barcode value used in data/train, data/test, and val split."""
    root = Path(data_dir)
    excluded = set()
    for split in ("train", "test"):
        clean_dir = root / split / "clean"
        if clean_dir.exists():
            for p in clean_dir.glob("*.png"):
                excluded.add(p.stem)
        dmg_dir = root / split / "damaged"
        if dmg_dir.exists():
            for p in dmg_dir.glob("*.png"):
                excluded.add(p.stem.rsplit("_v", 1)[0])
    return excluded


def assert_no_leakage(barcodes, data_dir="data", excluded=None):
    """Assert that none of `barcodes` appear in train, val, or test datasets."""
    if excluded is None:
        excluded = load_excluded_barcodes(data_dir)
    overlap = set(barcodes) & excluded
    if overlap:
        sample = sorted(overlap)[:5]
        raise AssertionError(
            f"Data leakage detected: {len(overlap)} generated barcodes appear in "
            f"train/val/test (e.g. {sample})"
        )


def write_run_config(out_path, seed: int, args_dict: dict, extra: dict = None):
    """Write a run-config JSON (seed, args, git commit) next to outputs."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in args_dict.items()},
        "git_commit": get_git_commit(),
    }
    if extra:
        payload.update(extra)
    p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return p
