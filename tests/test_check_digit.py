"""Force a wrong EAN-13 check digit into ZXing and check_digit_ok(), and report which layer rejects it."""
import sys
import unittest
from pathlib import Path

import numpy as np
from stdnum import ean

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import check_digit_ok, decode, run_pipeline

# Standard EAN-13 encoding tables
_L_CODES = {
    "0": "0001101", "1": "0011001", "2": "0010011", "3": "0111101", "4": "0100011",
    "5": "0110001", "6": "0101111", "7": "0111011", "8": "0110111", "9": "0001011",
}
_G_CODES = {
    "0": "0100111", "1": "0110011", "2": "0011011", "3": "0100001", "4": "0011101",
    "5": "0111001", "6": "0000101", "7": "0010001", "8": "0001001", "9": "0010111",
}
_R_CODES = {
    "0": "1110010", "1": "1100110", "2": "1101100", "3": "1000010", "4": "1011100",
    "5": "1001110", "6": "1010000", "7": "1000100", "8": "1001000", "9": "1110100",
}
_PARITY = {
    "0": "LLLLLL", "1": "LLGLGG", "2": "LLGGLG", "3": "LLGGGL", "4": "LGLLGG",
    "5": "LGGLLG", "6": "LGGGLL", "7": "LGLGLG", "8": "LGLGGL", "9": "LGGLGL",
}


def render_raw_ean13_13digits(code13: str, size=(384, 128)) -> np.ndarray:
    """Render an arbitrary 13-digit EAN-13 bar pattern (even with invalid check digit)."""
    assert len(code13) == 13 and code13.isdigit()
    first = code13[0]
    left = code13[1:7]
    right = code13[7:13]
    bits = ["101"]
    for d, p in zip(left, _PARITY[first]):
        bits.append(_L_CODES[d] if p == "L" else _G_CODES[d])
    bits.append("01010")
    for d in right:
        bits.append(_R_CODES[d])
    bits.append("101")
    modules = "".join(bits)  # 95 modules

    w, h = size
    img = np.full((h, w), 255, dtype=np.uint8)
    quiet = 24
    bar_w = (w - 2 * quiet) / 95.0
    for i, m in enumerate(modules):
        if m == "1":
            x0 = int(round(quiet + i * bar_w))
            x1 = int(round(quiet + (i + 1) * bar_w))
            img[16 : h - 16, x0:x1] = 0
    return img


class TestCheckDigitRejection(unittest.TestCase):
    def test_wrong_check_digit_rejected_by_zxing_and_check_digit_ok(self):
        payload12 = "890123456789"
        valid_cd = ean.calc_check_digit(payload12)
        wrong_cd = str((int(valid_cd) + 1) % 10)
        valid_code = payload12 + valid_cd
        wrong_code = payload12 + wrong_cd

        # Sanity: valid image decodes in ZXing and passes check_digit_ok
        valid_img = render_raw_ean13_13digits(valid_code)
        self.assertEqual(decode(valid_img), (valid_code, "EAN13"))
        self.assertTrue(check_digit_ok(valid_code))

        # Layer 1: check_digit_ok() rejects wrong check digit string
        cd_result = check_digit_ok(wrong_code)
        self.assertFalse(cd_result)

        # Layer 2: ZXing decoder rejects wrong check digit image
        wrong_img = render_raw_ean13_13digits(wrong_code)
        zxing_result = decode(wrong_img)
        self.assertIsNone(zxing_result)

        # Pipeline on wrong_img: fails at ZXing decode layer -> manual_review
        pipe_out = run_pipeline(wrong_img, model=None)
        self.assertEqual(pipe_out["status"], "manual_review")
        self.assertIsNone(pipe_out["candidate"])

        print(
            f"\n[CHECK DIGIT REPORT] valid={valid_code} wrong={wrong_code} | "
            f"ZXing decode(wrong_img)={zxing_result} (REJECTED at decoder layer) | "
            f"check_digit_ok(wrong_code)={cd_result} (REJECTED at guard layer)"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
