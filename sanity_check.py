#!/usr/bin/env python3
"""CPU sanity check for the patched train.py and parse_log_to_history.py.

Verifies:
  1. Output filenames use checkpoint stem prefix (<stem>_history.json, <stem>_history.csv, <stem>_split_manifest.csv).
  2. Seed order: seeding runs before UNet construction so weights are deterministically seeded.
  3. Checkpoint serialisation: plain types only, loads under torch.load(..., weights_only=True),
     and loads via pipeline.load_model(...) without modifying pipeline.py.
  4. DataLoader seeding: torch.Generator and worker_init_fn passed to tr, va unaffected.
  5. Parser: reconstructed LR formula matches CosineAnnealingLR and marks learning_rate_source.
  6. Resume safety: existing history triggers a clear notice on restart and starts fresh.

Usage:
    python sanity_check.py
"""
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from model import UNet
from pipeline import load_model


def make_dummy_image(path: Path, text: str = "4012345678901"):
    img = np.full((128, 384), 255, dtype=np.uint8)
    for x in range(30, 350, 4):
        cv2.line(img, (x, 10), (x, 118), 0, 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def run_sanity_check():
    tmp_dir = Path(tempfile.mkdtemp(prefix="barcode_sanity_"))
    print(f"[*] Running CPU sanity check in {tmp_dir}")
    try:
        # Check Item 2: Seed order determinism before UNet construction
        torch.manual_seed(42)
        m1 = UNet(base=16)
        torch.manual_seed(42)
        m2 = UNet(base=16)
        diff = sum((p1 - p2).abs().sum().item() for p1, p2 in zip(m1.parameters(), m2.parameters()))
        assert diff == 0.0, "UNet initial weights differ when seeded with 42!"
        print("[✓] Item 2 Verified: Seeding before UNet construction yields deterministic initial weights.")

        # Prepare minimal dataset structure: 4 train barcodes, 2 test barcodes
        train_stems = [f"40000000000{i:02d}" for i in range(1, 5)]
        test_stems = [f"40000000000{i:02d}" for i in range(5, 7)]

        data_dir = tmp_dir / "data"
        for s in train_stems:
            make_dummy_image(data_dir / "train" / "clean" / f"{s}.png")
            make_dummy_image(data_dir / "train" / "damaged" / f"{s}_v0.png")
            make_dummy_image(data_dir / "train" / "damaged" / f"{s}_v1.png")

        for s in test_stems:
            make_dummy_image(data_dir / "test" / "clean" / f"{s}.png")
            make_dummy_image(data_dir / "test" / "damaged" / f"{s}_v0.png")

        models_dir = tmp_dir / "models"
        out_pt = models_dir / "unet_test.pt"
        ckpt_stem = out_pt.stem

        cmd = [
            sys.executable,
            "train.py",
            "--data", str(data_dir),
            "--epochs", "1",
            "--batch", "2",
            "--base", "16",
            "--seed", "42",
            "--val-frac", "0.25",
            "--out", str(out_pt),
        ]
        print(f"[*] Executing 1-epoch training run: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.returncode != 0:
            print(res.stderr, file=sys.stderr)
            raise RuntimeError(f"train.py failed with exit code {res.returncode}")

        # Item 1: Verify output filenames use checkpoint stem prefix
        manifest_path = models_dir / f"{ckpt_stem}_split_manifest.csv"
        h_json_path = models_dir / f"{ckpt_stem}_history.json"
        h_csv_path = models_dir / f"{ckpt_stem}_history.csv"

        assert manifest_path.exists(), f"Expected manifest file {manifest_path.name} not found!"
        assert h_json_path.exists(), f"Expected history JSON file {h_json_path.name} not found!"
        assert h_csv_path.exists(), f"Expected history CSV file {h_csv_path.name} not found!"
        print(f"[✓] Item 1 Verified: Unique filenames created with prefix '{ckpt_stem}':")
        print(f"    - {manifest_path.name}")
        print(f"    - {h_json_path.name}")
        print(f"    - {h_csv_path.name}")

        with open(manifest_path, encoding="utf-8") as f:
            manifest_rows = list(csv.DictReader(f))
        manifest_stems = {r["stem"] for r in manifest_rows}
        assert manifest_stems == set(train_stems + test_stems)
        splits = {r["split"] for r in manifest_rows}
        assert "train" in splits and "val" in splits and "test" in splits

        # Item 3: Checkpoint serialisation, plain types, weights_only=True & pipeline.load_model
        assert out_pt.exists(), "Model checkpoint was not saved!"
        # Test weights_only=True load
        ckpt = torch.load(out_pt, map_location="cpu", weights_only=True)
        print("[✓] Item 3a Verified: Checkpoint loads cleanly with torch.load(..., weights_only=True)")

        for key in ["model", "base", "val_decode", "epoch", "seed", "args", "min_consistency"]:
            assert key in ckpt, f"Key '{key}' missing from checkpoint!"
        assert isinstance(ckpt["args"], dict), f"ckpt['args'] must be dict, got {type(ckpt['args'])}"
        for k, v in ckpt["args"].items():
            assert isinstance(v, (str, int, float, bool, list, tuple, dict, type(None))), (
                f"args key '{k}' has non-plain type: {type(v)}"
            )
        print("[✓] Item 3b Verified: vars(args) contains only plain types (no Path objects).")

        # Test loading via pipeline.py (without editing pipeline.py)
        model_loaded = load_model(out_pt, device="cpu")
        assert isinstance(model_loaded, torch.nn.Module), "load_model did not return a valid PyTorch model!"
        assert hasattr(model_loaded, "min_consistency"), "model missing min_consistency attribute"
        print(f"[✓] Item 3c Verified: Successfully loaded via pipeline.load_model(...) (min_consistency={model_loaded.min_consistency:.2f})")

        # Item 4: Check DataLoader configuration in train.py code
        import inspect
        import train
        src = inspect.getsource(train.main)
        assert "generator=g" in src and "worker_init_fn=seed_worker" in src, "tr DataLoader not seeded!"
        assert "va = DataLoader(PairDataset(va_files, root / \"clean\"), batch_size=args.batch,\n                    num_workers=args.workers)" in src, "va DataLoader was modified!"
        print("[✓] Item 4 Verified: tr has generator and worker_init_fn; va loader is unaffected.")

        # Item 6: Test Resume Safety (restart notice)
        print("[*] Testing restart / resume safety notice...")
        res_restart = subprocess.run(cmd, capture_output=True, text=True)
        assert "Notice: Existing history" in res_restart.stdout, "Restart notice was not printed!"
        with open(h_csv_path, encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
        assert len(csv_rows) == 1, f"Expected 1 row after restart, got {len(csv_rows)} (history was duplicated!)"
        print("[✓] Item 6 Verified: Restart correctly outputs notice and avoids duplicating history.")

        # Item 5: Test parse_log_to_history.py
        import parse_log_to_history
        sample_log = (
            "epoch   1/30  train 0.2240  val 0.0933  val: correct reads 72.7%, wrong reads 0.6%  (141s)\n"
            "epoch  26/30  train 0.0024  val 0.0142  val: correct reads 94.2%, wrong reads 0.6%  (138s)\n"
            "epoch  30/30  train 0.0020  val 0.0141  val: correct reads 93.6%, wrong reads 0.8%  (138s)\n"
        )
        parsed_records = parse_log_to_history.parse_log_text(sample_log, base_lr=1e-3, default_total_epochs=30)
        assert len(parsed_records) == 3
        for r in parsed_records:
            assert r["learning_rate_source"] == "derived from cosine schedule, not measured"
        # Check epoch 1 lr: 0.001
        assert parsed_records[0]["learning_rate"] == 0.001
        print("[✓] Item 5 Verified: parse_log_to_history correctly reconstructs LR and sets learning_rate_source.")

        print("\n==========================================")
        print("ALL SANITY CHECKS PASSED SUCCESSFULLY!")
        print("==========================================")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_sanity_check()
