# Results Provenance

- **Training code commit (post-rewrite)**: `e8b92b5` (*"Add training seeding, tear damage, DB/PIN guards and courier UI (code used to train unet_v2)"*). `models/frozen/unet_v2_final.pt` was trained with `train.py` as of commit `e8b92b5`.
- **Frozen checkpoint SHA-256** (from `results/freeze.json`): `306f77683fe4502ba60cf452821c282e2d4075939e48c0b59f3ba99b16f356f7` (`20,615,699` bytes, epoch `30`, `min_consistency=0.600761514079721`).
- **Pre-rewrite `run_config.json` hashes**: The existing `*.run_config.json` files record pre-rewrite commit hashes (`6ba0276a...`, `3144d1bc...`, `6e47cff4...`, `0cb8eb34...`). Those earlier commits predate the code changes committed in `e8b92b5` and may not run on their own; the current `HEAD` is the supported version.

## `results/` File Mapping

| File | Generating Script / Source | Date |
|---|---|---|
| `results/direct_path_guard_analysis.csv` | `eval/analyze_direct_path_guard.py` | 2026-11-26 / 2026-11-27 |
| `results/direct_path_guard_analysis.run_config.json` | `eval/analyze_direct_path_guard.py` | 2026-11-26 / 2026-11-27 |
| `results/direct_path_guard_v2.csv` | `eval/analyze_direct_path_guard_v2.py` | 2026-11-26 / 2026-11-27 |
| `results/direct_path_guard_v2_summary.json` | `eval/analyze_direct_path_guard_v2.py` | 2026-11-26 / 2026-11-27 |
| `results/direct_path_guard_v2.run_config.json` | `eval/analyze_direct_path_guard_v2.py` | 2026-11-26 / 2026-11-27 |
| `results/wrong_read_profile.csv` | `eval/analyze_db_guard_adversarial.py` | 2026-11-27 |
| `results/db_guard_adversarial.csv` | `eval/analyze_db_guard_adversarial.py` | 2026-11-27 |
| `results/db_guard_adversarial_summary.json` | `eval/analyze_db_guard_adversarial.py` | 2026-11-27 |
| `results/db_guard_adversarial.run_config.json` | `eval/analyze_db_guard_adversarial.py` | 2026-11-27 |
| `results/freeze.json` | Manual freeze from `models/unet_v2.pt` & `train.py` inspection | 2026-11-27 |
| `results/overfit_check.csv` | `eval/overfit_check.py` | 2026-11-27 |
| `results/overfit_check_summary.json` | `eval/overfit_check.py` | 2026-11-27 |
| `results/overfit_check.run_config.json` | `eval/overfit_check.py` | 2026-11-27 |
| `results/PROVENANCE.md` | Provenance documentation | 2026-11-27 |
