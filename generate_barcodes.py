"""
Generate clean EAN-13 barcode images for training.

Install:
    pip install python-barcode pillow zxing-cpp opencv-python

Run:
    python generate_barcodes.py --count 5000 --out data/clean_all

Output:
    data/clean_all/<13-digit-value>.png
    The filename IS the true barcode value, so labels are never lost.
"""
import argparse
import random
from pathlib import Path

import barcode
import cv2
import numpy as np
import zxingcpp
from barcode.writer import ImageWriter


BARCODE_OPTIONS = {
    "module_width": 0.4,     # width of the thinnest bar (mm)
    "module_height": 15.0,   # bar height (mm)
    "quiet_zone": 6.5,       # white margin left/right
    "write_text": False,     # no printed digits, so the model must rely on the bars
    "dpi": 300,
}


def render_ean13(code12: str, size: tuple = (384, 128)) -> np.ndarray:
    """Render an EAN-13 barcode into a grayscale uint8 image matching dataset specs."""
    ean = barcode.get("ean13", str(code12)[:12], writer=ImageWriter())
    pil_img = ean.render(writer_options=BARCODE_OPTIONS).convert("L")
    arr = np.array(pil_img, dtype=np.uint8)
    if size is not None:
        arr = cv2.resize(arr, size, interpolation=cv2.INTER_AREA)
    return arr


def make_barcode(code12: str, out_dir: Path) -> Path:
    """Create one EAN-13 PNG. The library appends the correct check digit."""
    ean = barcode.get("ean13", code12, writer=ImageWriter())
    # save() adds the .png extension itself and returns the final path
    path = ean.save(
        str(out_dir / ean.get_fullcode()),
        options=BARCODE_OPTIONS,
    )
    return Path(path)


def decodes_correctly(path: Path) -> bool:
    """Confirm ZXing can read the image and returns the value in the filename."""
    img = cv2.imread(str(path))
    if img is None:
        return False
    results = zxingcpp.read_barcodes(img)
    return bool(results) and results[0].text == path.stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--out", type=str, default="data/clean_all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen = set()
    made = bad = 0
    while made < args.count:
        code12 = "".join(random.choices("0123456789", k=12))
        if code12 in seen:
            continue
        seen.add(code12)

        path = make_barcode(code12, out_dir)
        if decodes_correctly(path):
            made += 1
            if made % 500 == 0:
                print(f"{made}/{args.count} generated")
        else:
            path.unlink(missing_ok=True)  # discard anything ZXing can't read
            bad += 1

    print(f"Done. {made} clean barcodes in {out_dir} ({bad} discarded).")


if __name__ == "__main__":
    main()
