"""Tests for eval/hard_negatives.py."""
import sys
import tempfile
import unittest
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.hard_negatives import generate_hard_negatives
from eval.leakage import assert_no_leakage, load_excluded_barcodes
from pipeline import decode


class TestHardNegatives(unittest.TestCase):
    def test_hard_negatives_decode_to_target_and_split_by_barcode(self):
        excluded = load_excluded_barcodes(ROOT / "data")
        with tempfile.TemporaryDirectory() as tmp:
            rows, manifest_csv = generate_hard_negatives(
                num_barcodes=6,
                seed=2027,
                cal_frac=0.5,
                data_dir=str(ROOT / "data"),
                out_dir=tmp,
            )
            self.assertTrue(manifest_csv.exists())
            self.assertEqual(len(rows), 12)  # 6 truths * 2 kinds

            cal_truths = {r["truth"] for r in rows if r["split"] == "cal"}
            eval_truths = {r["truth"] for r in rows if r["split"] == "eval"}
            self.assertEqual(len(cal_truths & eval_truths), 0)
            self.assertGreater(len(cal_truths), 0)
            self.assertGreater(len(eval_truths), 0)

            all_codes = []
            for r in rows:
                self.assertNotEqual(r["truth"], r["hard_negative"])
                img = cv2.imread(r["file"], cv2.IMREAD_GRAYSCALE)
                dec = decode(img)
                self.assertIsNotNone(dec)
                self.assertEqual(dec[0], r["hard_negative"])
                all_codes.extend([r["truth"], r["hard_negative"]])

            assert_no_leakage(all_codes, excluded=excluded)


if __name__ == "__main__":
    unittest.main(verbosity=2)
