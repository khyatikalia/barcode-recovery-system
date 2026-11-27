# EAN-13 Barcode Recovery & Automated Courier Sorting

An end-to-end computer vision and deep learning pipeline that reconstructs damaged, smudged, blurred, and partially torn EAN-13 shipping barcodes using a 5-stage U-Net and routes parcels through a multi-tier anti-hallucination guard system.

## Folder Layout

```
barcode_recovery/
├── model.py                         # 5-stage U-Net architecture (5.14M params)
├── pipeline.py                      # Inference pipeline: enhance -> ZXing -> U-Net -> guards
├── damage.py                        # Synthetic degradation engine (blur, noise, smudge, tear)
├── generate_barcodes.py             # Clean EAN-13 generator + render_ean13()
├── prepare_dataset.py               # Train/test split & damaged variant generator
├── train.py                         # Seeded U-Net training & consistency-threshold calibration
├── evaluate.py / evaluate_guard.py  # Legacy evaluation scripts
├── app.py                           # 3-tab Streamlit UI (Scanner, Manifest, PIN Routing)
├── eval/                            # Reproducible evaluation harness & analysis scripts
│   ├── guards_trace.py              # Guard-by-guard tracer + side consistency scoring
│   ├── leakage.py                   # Train/val/test leakage checks & run-config writer
│   ├── manifests.py                 # Validated SQLite manifests (random/neighbour/sequential/dense)
│   ├── hard_negatives.py            # 1-digit substitution & transposition hard negatives
│   ├── analyze_direct_path_guard.py # 1,000-barcode direct/enhanced consistency analysis
│   ├── analyze_direct_path_guard_v2.py # 20,000-barcode scaled direct/enhanced + DB guard analysis
│   ├── analyze_db_guard_adversarial.py # Wrong-read profiling, adversarial DB & synthetic injection
│   └── overfit_check.py             # Train vs val vs fresh val2 overfitting check
├── tests/                           # Unit tests (guard coverage, check digit, manifests, hard negatives)
├── results/                         # Tracked CSV/JSON evaluation outputs & run configs
└── models/frozen/                   # Tracked training history & split manifest (weights untracked)
```

## Installation

```bash
pip install -r requirements.txt
```

## Running the App & Tests

```bash
# Run the Streamlit app
python3 -m streamlit run app.py

# Run unit tests
python3 -m unittest discover -s tests -v
```

## Running `eval/` Analysis Scripts

```bash
# 1. Direct & enhanced path consistency analysis (1,000 fresh barcodes)
python3 -m eval.analyze_direct_path_guard --count 1000 --seed 31415 --checkpoint models/unet.pt

# 2. Scaled direct/enhanced + DB guard analysis (up to 20,000 fresh barcodes)
python3 -m eval.analyze_direct_path_guard_v2 --seed 98765 --target-wrong 100 --max-barcodes 20000 --checkpoint models/unet.pt

# 3. Wrong-read profile, adversarial manifests & synthetic error-injection
python3 -m eval.analyze_db_guard_adversarial --seed 54321

# 4. Overfitting check (train sample vs val vs fresh val2)
python3 -m eval.overfit_check

# 5. Generate validated SQLite manifests or hard-negative images
python3 -m eval.manifests --mode dense_valid_neighbours --count 200 --seed 1337 --out results/manifest.db
python3 -m eval.hard_negatives --count 40 --seed 2027 --out-dir results/hard_negatives
```

## Results

All evaluation artifacts in `results/` include matching `.run_config.json` metadata (seed, args, git commit):

- `results/freeze.json` — Frozen checkpoint metadata (`models/frozen/unet_v2_final.pt`, epoch `30`, `val_read_rate=0.94`, `min_consistency=0.600762`) and `train.py` validation audit.
- `results/overfit_check.csv` / `overfit_check_summary.json` — ZXing-failure U-Net evaluation at `min_consistency=0.600762`:
  - `(a) train-sample-1000` (`n=711`): `99.9%` decode, `0.0%` wrong, `46.0%` accepted.
  - `(b) val-split` (`n=327`): `93.0%` decode, `0.9%` wrong, `46.8%` accepted.
  - `(c) val2-fresh-2000` (`n=1,459`): `92.4%` decode, `1.1%` wrong, `46.3%` accepted.
- `results/direct_path_guard_analysis.csv` — `241` direct/enhanced reads from `1,000` fresh barcodes (`236` correct, `5` wrong).
- `results/direct_path_guard_v2.csv` / `direct_path_guard_v2_summary.json` — `5,216` direct/enhanced reads across `20,000` barcodes (`28` wrong reads total; `0/2,608` wrong accepts after DB guard on `eval`).
- `results/wrong_read_profile.csv` — Hamming distance (`2` to `7`) and damage profile of the `28` wrong reads (`0/28` transpositions).
- `results/db_guard_adversarial.csv` / `db_guard_adversarial_summary.json` — Adversarial DB guard results (`0/2,608` real wrong accepts across `random`, `neighbour`, `sequential`, `dense_valid_neighbours`, `missing_10pct`; synthetic error-injection on `5,216` truths yields `2/5,216` wrong accepts on `sequential` and `4/5,216` on `dense_valid_neighbours`).

## Model Weights

Model weights (`*.pt`, `*.pth`, `*.onnx`) are excluded from git via `.gitignore`.
- **Frozen checkpoint**: `models/frozen/unet_v2_final.pt`
  - **SHA-256**: `306f77683fe4502ba60cf452821c282e2d4075939e48c0b59f3ba99b16f356f7`
  - **Size**: `20,615,699` bytes (`min_consistency = 0.600761514079721`, epoch `30`)
- **Regenerate weights**:
  ```bash
  python3 generate_barcodes.py --count 5000 --out data/clean_all
  python3 prepare_dataset.py --src data/clean_all --out data --seed 42 --train-variants 2
  python3 train.py --seed 42 --data data --epochs 30 --batch 16 --out models/unet_v2.pt
  ```
