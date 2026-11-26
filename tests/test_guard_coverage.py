"""Tests proving direct and enhanced paths skip the consistency guard.

These are marked expectedFailure to document the current guard-coverage gap
without modifying pipeline.py.
"""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.guards_trace import trace_pipeline
from generate_barcodes import render_ean13
from pipeline import decode, enhance


def _make_enhanced_only_barcode(code12="890123456789"):
    """Create a low-contrast + blurred barcode that fails direct decode() but passes enhance()."""
    clean = render_ean13(code12, size=(384, 128)).astype(np.float32)
    low_c = np.clip(128.0 + 0.4 * (clean - 128.0), 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(low_c, (5, 5), 2.0)
    assert decode(blurred) is None and decode(enhance(blurred)) is not None
    return blurred


class TestGuardCoverage(unittest.TestCase):
    @unittest.expectedFailure
    def test_direct_path_enforces_consistency_guard(self):
        """FINDING: Direct decode path skips consistency_score() guard."""
        img = render_ean13("890123456789", size=(384, 128))
        trace = trace_pipeline(img, model=None)
        self.assertEqual(trace["decode_path"], "direct")
        # Expected to fail: pipeline_out["score"] is None and guard is skipped
        self.assertTrue(trace["consistency_guard_applied"])
        self.assertIsNotNone(trace["pipeline_out"]["score"])

    @unittest.expectedFailure
    def test_enhanced_path_enforces_consistency_guard(self):
        """FINDING: Enhanced decode path skips consistency_score() guard."""
        img = _make_enhanced_only_barcode("890123456789")
        trace = trace_pipeline(img, model=None)
        self.assertEqual(trace["decode_path"], "enhanced")
        # Expected to fail: pipeline_out["score"] is None and guard is skipped
        self.assertTrue(trace["consistency_guard_applied"])
        self.assertIsNotNone(trace["pipeline_out"]["score"])

    def test_tracer_records_side_consistency_without_altering_result(self):
        """Verify guards_trace computes side_consistency_score on direct & enhanced."""
        clean = render_ean13("890123456789", size=(384, 128))
        t_direct = trace_pipeline(clean, model=None)
        self.assertEqual(t_direct["decode_path"], "direct")
        self.assertEqual(t_direct["final_status"], "accepted")
        self.assertFalse(t_direct["consistency_guard_applied"])
        self.assertIsNotNone(t_direct["side_consistency_score"])
        self.assertGreater(t_direct["side_consistency_score"], 0.8)

        enh_only = _make_enhanced_only_barcode("890123456789")
        t_enh = trace_pipeline(enh_only, model=None)
        self.assertEqual(t_enh["decode_path"], "enhanced")
        self.assertEqual(t_enh["final_status"], "accepted")
        self.assertFalse(t_enh["consistency_guard_applied"])
        self.assertIsNotNone(t_enh["side_consistency_score"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
