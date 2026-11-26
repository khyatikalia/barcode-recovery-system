"""Tests for eval/manifests.py."""
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.leakage import assert_no_leakage, load_excluded_barcodes
from eval.manifests import (generate_neighbour_awbs, generate_random_awbs,
                            validate_awb, write_manifest_db)


class TestManifests(unittest.TestCase):
    def test_random_and_neighbour_manifests_valid_and_no_leakage(self):
        excluded = load_excluded_barcodes(ROOT / "data")
        truths = generate_random_awbs(5, seed=1337, excluded=excluded)
        self.assertEqual(len(truths), 5)
        for t in truths:
            validate_awb(t)

        neighbours = generate_neighbour_awbs(truths, excluded=excluded)
        # At least 100 unique valid-check-digit neighbours per truth
        self.assertGreater(len(neighbours), 400)
        awbs = [a for a, _ in neighbours]
        self.assertEqual(len(awbs), len(set(awbs)))
        assert_no_leakage(awbs, excluded=excluded)

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test_manifest.db"
            write_manifest_db(db_path, neighbours, mode="neighbour", seed=1337)
            conn = sqlite3.connect(str(db_path))
            count = conn.execute("SELECT COUNT(*) FROM shipments").fetchone()[0]
            conn.close()
            self.assertEqual(count, len(neighbours))

    def test_rejects_seed_0_or_42(self):
        with self.assertRaises(ValueError):
            generate_random_awbs(5, seed=42, excluded=set())
        with self.assertRaises(ValueError):
            generate_random_awbs(5, seed=0, excluded=set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
