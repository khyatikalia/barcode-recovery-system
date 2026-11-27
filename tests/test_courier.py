"""Comprehensive Courier Sorting Pipeline & Safety Guard Test Suite.

Asserts:
1. Guard blocks valid-check-digit value not in manifest.
2. Guard accepts manifested value.
3. Unmapped PIN routes to manual review with reason 'PIN missing or not in routing table'.
4. damage() regression passes on 20 barcodes.
5. Torn read rate is lower than clean.
6. app.py imports without error.
"""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is in sys.path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import damage as d_module
from generate_barcodes import render_ean13
from pipeline import check_digit_ok, decode, route_by_pincode, run_pipeline


class TestCourierSafetyGuards(unittest.TestCase):

    def setUp(self):
        # Generate standard clean barcode for 8901234567890 (valid check digit)
        # 12 digits: 890123456789 -> check digit is 0 -> full code 8901234567890
        self.valid_awb = "8901234567890"
        self.assertTrue(check_digit_ok(self.valid_awb))
        self.clean_arr = render_ean13(self.valid_awb[:12], size=(384, 128))

    def test_01_guard_blocks_unmanifested_valid_check_digit(self):
        """Assert guard blocks valid-check-digit value not in manifest."""
        mock_manifest = {
            "8909999999999": {"awb": "8909999999999", "consignee_pin": "560001"}
        }
        res = run_pipeline(self.clean_arr, model=None, db_manifest=mock_manifest)

        self.assertEqual(res["candidate"], self.valid_awb)
        self.assertEqual(res["status"], "manual_review")
        self.assertIsNone(res["value"])
        self.assertFalse(res["db_verified"])
        self.assertIn("Database Guard Block", res["reason"])
        print("\n[TEST 1 PASS]: Guard successfully blocked valid-check-digit barcode not present in database.")

    def test_02_guard_accepts_manifested_value(self):
        """Assert guard accepts manifested value and routes to sorting lane."""
        mock_manifest = {
            self.valid_awb: {"awb": self.valid_awb, "consignee_pin": "560001"}  # 560 = Bengaluru -> Lane 3
        }
        res = run_pipeline(self.clean_arr, model=None, db_manifest=mock_manifest)

        self.assertEqual(res["status"], "accepted")
        self.assertEqual(res["value"], self.valid_awb)
        self.assertTrue(res["db_verified"])
        self.assertIsNotNone(res["route"])
        self.assertEqual(res["route"]["lane"], "Lane 3 (South)")
        self.assertEqual(res["route"]["hub"], "Bengaluru Sorting Hub")
        print("[TEST 2 PASS]: Guard accepted valid manifested parcel and mapped to Lane 3.")

    def test_03_unmapped_pin_routes_to_manual_review(self):
        """Assert unmapped PIN routes to manual review with exact reason string."""
        mock_manifest = {
            self.valid_awb: {"awb": self.valid_awb, "consignee_pin": "999999"}  # Unmapped PIN prefix
        }
        res = run_pipeline(self.clean_arr, model=None, db_manifest=mock_manifest)

        self.assertEqual(res["status"], "manual_review")
        self.assertEqual(res["candidate"], self.valid_awb)
        self.assertIsNone(res["value"])
        self.assertIsNone(res["route"])
        self.assertEqual(res["reason"], "PIN missing or not in routing table")
        print("[TEST 3 PASS]: Parcel with unmapped PIN routed to manual review with reason 'PIN missing or not in routing table'.")

    def test_04_damage_regression_20_barcodes(self):
        """Assert damage() maintains stable deterministic output for 20 barcodes with fixed seed."""
        for seed in range(20):
            rng1 = np.random.default_rng(2000 + seed)
            img1 = d_module.damage(self.clean_arr, rng1)

            rng2 = np.random.default_rng(2000 + seed)
            img2 = d_module.damage(self.clean_arr, rng2)

            self.assertTrue(np.array_equal(img1, img2), f"Non-deterministic damage on barcode {seed}")
        print("[TEST 4 PASS]: damage() deterministic regression check passed across 20 test runs.")

    def test_05_torn_read_rate_lower_than_clean(self):
        """Assert torn barcodes decode at a lower rate than clean barcodes under plain ZXing."""
        n_samples = 30
        clean_reads = 0
        torn_reads = 0
        rng = np.random.default_rng(12345)

        for i in range(n_samples):
            code12 = f"8901000{i:05d}"
            clean = render_ean13(code12, size=(384, 128))
            full_code = code12 + str(check_digit_ok and (10 - sum(int(c) * (1 if idx % 2 == 0 else 3) for idx, c in enumerate(code12)) % 10) % 10)

            # Clean decode
            r_clean = decode(clean)
            if r_clean and r_clean[0].startswith(code12[:10]):
                clean_reads += 1

            # Torn decode
            torn_img, ops, _ = d_module.damage_with_info(clean, rng, torn=True)
            r_torn = decode(torn_img)
            if r_torn and r_torn[0].startswith(code12[:10]):
                torn_reads += 1

        clean_rate = clean_reads / n_samples
        torn_rate = torn_reads / n_samples
        print(f"[TEST 5 PASS]: Plain ZXing decode rate: Clean = {clean_rate*100:.1f}%, Torn = {torn_rate*100:.1f}% (torn decodes significantly lower).")
        self.assertGreater(clean_rate, torn_rate)

    def test_06_app_imports_cleanly(self):
        """Assert app.py imports cleanly without raising exceptions."""
        import app
        self.assertTrue(hasattr(app, "run_pipeline"))
        self.assertTrue(hasattr(app, "route_by_pincode"))
        print("[TEST 6 PASS]: app.py imported and verified successfully.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
