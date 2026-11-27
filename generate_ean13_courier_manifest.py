#!/usr/bin/env python3
"""Generate a realistic EAN-13 courier shipment dataset and database.

Every shipment uses a 13-digit GS1 EAN-13 barcode (starting with 890... with valid check digit).
Outputs:
  <out-dir>/ean13_manifest.xlsx
  <out-dir>/ean13_manifest.csv
  <out-dir>/ean13_manifest.db (SQLite)
  <out-dir>/clean/       (384x128 EAN-13 images)
  <out-dir>/damaged/     (384x128 damaged EAN-13 images using damage())
  <out-dir>/torn/        (384x128 torn EAN-13 images using damage_with_info(torn=True))
  <out-dir>/labels/      (800x560 full courier shipping labels)
  <out-dir>/qr/          (2D QR codes with embedded JSON shipping details)
  <out-dir>/manifest.csv (file, kind, damages, severity)
"""
import argparse
import io
import json
import random
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import qrcode
from PIL import Image, ImageDraw, ImageFont

from damage import damage, damage_with_info
from generate_barcodes import render_ean13
from pipeline import PIN_ROUTING_TABLE

CITIES = [
    ("Delhi", "Delhi", "110001"),
    ("Mumbai", "Maharashtra", "400001"),
    ("Bengaluru", "Karnataka", "560001"),
    ("Chennai", "Tamil Nadu", "600001"),
    ("Kolkata", "West Bengal", "700001"),
    ("Hyderabad", "Telangana", "500001"),
    ("Pune", "Maharashtra", "411001"),
    ("Ahmedabad", "Gujarat", "380001"),
    ("Jaipur", "Rajasthan", "302001"),
    ("Lucknow", "Uttar Pradesh", "226001"),
    ("Kochi", "Kerala", "682001"),
    ("Indore", "Madhya Pradesh", "452001"),
]

CARRIERS = ["Delhivery", "Blue Dart", "DTDC", "India Post", "Xpressbees", "Ecom Express"]
NAMES = ["Riya Sharma", "Arjun Patel", "Meera Iyer", "Karan Singh", "Anika Nair",
         "Rohan Mehta", "Sneha Reddy", "Vikram Das", "Pooja Joshi", "Aditya Khan"]
STREETS = ["MG Road", "Station Road", "Park Street", "Gandhi Nagar", "Lake View Colony", "Brigade Road"]
SHIPPERS = ["Acme Retail", "UrbanKart Logistics", "TechHub Electronics", "StyleWear India", "BookNest Online"]
ITEMS = ["Wireless Earbuds", "Cotton T-Shirt", "Smart Watch", "Running Shoes", "Hardcover Book", "Coffee Maker"]


def compute_ean13_check_digit(digits12: str) -> str:
    """Compute standard GS1 EAN-13 check digit for a 12-digit string."""
    s = 0
    for idx, c in enumerate(digits12):
        d = int(c)
        s += d if idx % 2 == 0 else d * 3
    return str((10 - (s % 10)) % 10)


def generate_ean13_code(rand_gen, sequential_idx: int = None) -> str:
    """Generate 13-digit EAN-13 string starting with 890 with valid check digit."""
    if sequential_idx is not None:
        code12 = f"890{sequential_idx:09d}"
    else:
        code12 = "890" + "".join(rand_gen.choices("0123456789", k=9))
    return code12 + compute_ean13_check_digit(code12)


def make_label_image(r: dict, clean_ean_arr: np.ndarray) -> Image.Image:
    """Generate a clean 800x560 courier shipping label with EAN-13 barcode and QR code."""
    W, H = 800, 560
    im = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(im)

    try:
        font_large = ImageFont.load_default(size=22)
        font_mid = ImageFont.load_default(size=16)
        font_small = ImageFont.load_default(size=12)
    except TypeError:
        font_large = ImageFont.load_default()
        font_mid = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Outer border
    d.rectangle([4, 4, W - 5, H - 5], outline=0, width=3)
    d.text((20, 14), f"COURIER: {r['carrier'].upper()}", font=font_large, fill=0)
    d.text((W - 380, 18), f"{r['service_type']} | {r['payment_mode']}", font=font_mid, fill=0)
    if r['payment_mode'] == "COD":
        d.text((W - 380, 48), f"COLLECT: Rs. {r['cod_amount']}", font=font_mid, fill=0)

    # Paste EAN-13 barcode (scaled to fit left label area)
    clean_pil = Image.fromarray(clean_ean_arr)
    barcode_display = clean_pil.resize((450, 110), Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS)
    im.paste(barcode_display, (20, 75))
    d.text((20, 195), f"EAN-13 / AWB: {r['awb']}", font=font_mid, fill=0)

    # Generate QR Code with embedded JSON shipping payload (NO pre-assigned lane or hub)
    qr_payload = json.dumps({
        "awb": r["awb"],
        "carrier": r["carrier"],
        "order": r["order_id"],
        "to": r["consignee_name"],
        "phone": r["consignee_phone"],
        "pin": r["consignee_pin"],
        "city": r["consignee_city"],
        "pay": f"{r['payment_mode']} (Rs. {r['cod_amount']})",
        "item": r["contents"],
    }, indent=None)

    q = qrcode.QRCode(border=2, box_size=6, error_correction=qrcode.constants.ERROR_CORRECT_M)
    q.add_data(qr_payload)
    q.make(fit=True)
    qr_img = q.make_image(fill_color="black", back_color="white").convert("L").resize((160, 160))
    im.paste(qr_img, (W - 190, 75))

    # Clean Divider (no pre-printed sorting lane banner)
    d.line([20, 235, W - 20, 235], fill=0, width=2)

    # Consignee Details
    d.text((20, 255), "SHIP TO:", font=font_small, fill=0)
    d.text((20, 275), f"{r['consignee_name']}", font=font_large, fill=0)
    d.text((20, 310), f"{r['consignee_address']}, {r['consignee_city']}, {r['consignee_state']} - {r['consignee_pin']}", font=font_mid, fill=0)
    d.text((20, 340), f"Contact: {r['consignee_phone']} | Order: {r['order_id']}", font=font_mid, fill=0)

    # Shipper details & Parcel Info
    d.line([20, 385, W - 20, 385], fill=0, width=2)
    d.text((20, 405), f"FROM: {r['shipper_name']} - PIN: {r['shipper_pin']}", font=font_small, fill=0)
    d.text((20, 435), f"Item: {r['contents']} | Weight: {r['weight_kg']} kg | Dispatch Date: {r['ship_date']}", font=font_small, fill=0)

    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=200, help="Number of EAN-13 shipments")
    ap.add_argument("--out-dir", default="data/ean13_courier", help="Output directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sequential", action="store_true", help="Generate sequential AWBs (890 + counter)")
    args = ap.parse_args()

    rand = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    out_root = Path(args.out_dir)
    clean_dir = out_root / "clean"
    damaged_dir = out_root / "damaged"
    torn_dir = out_root / "torn"
    labels_dir = out_root / "labels"
    qr_dir = out_root / "qr"
    for d in (clean_dir, damaged_dir, torn_dir, labels_dir, qr_dir):
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    seen = set()
    print(f"Generating {args.count} EAN-13 courier shipments (sequential={args.sequential})...")

    seq_idx = 1
    while len(rows) < args.count:
        if args.sequential:
            awb = generate_ean13_code(rand, sequential_idx=seq_idx)
            seq_idx += 1
        else:
            awb = generate_ean13_code(rand)

        if awb in seen:
            continue
        seen.add(awb)

        city_dest = rand.choice(CITIES)
        city_orig = rand.choice(CITIES)

        prefix = city_dest[2][:3]
        if prefix in PIN_ROUTING_TABLE:
            lane, hub, region = PIN_ROUTING_TABLE[prefix]
        else:
            lane, hub, region = "Unassigned", "Unassigned", "Unassigned"

        is_cod = rand.random() < 0.45
        val = rand.randint(3, 50) * 100

        record = {
            "awb": awb,
            "carrier": rand.choice(CARRIERS),
            "order_id": f"ORD-2026-{rand.randint(10000, 99999)}",
            "service_type": rand.choice(["Express Air", "Surface Cargo", "Standard Ground"]),
            "payment_mode": "COD" if is_cod else "Prepaid",
            "cod_amount": val if is_cod else 0,
            "weight_kg": round(rand.uniform(0.25, 5.0), 2),
            "declared_value": val,
            "shipper_name": rand.choice(SHIPPERS),
            "shipper_city": city_orig[0],
            "shipper_pin": city_orig[2],
            "consignee_name": rand.choice(NAMES),
            "consignee_phone": "9" + "".join(rand.choices("0123456789", k=9)),
            "consignee_address": f"{rand.randint(1, 150)} {rand.choice(STREETS)}",
            "consignee_city": city_dest[0],
            "consignee_state": city_dest[1],
            "consignee_pin": city_dest[2],
            "sorting_lane": lane,
            "sorting_hub": hub,
            "sorting_region": region,
            "ship_date": f"2026-11-{rand.randint(10, 16):02d}",
            "contents": rand.choice(ITEMS),
        }
        rows.append(record)

    df = pd.DataFrame(rows)

    # Save to Excel, CSV and SQLite database
    excel_path = out_root / "ean13_manifest.xlsx"
    csv_path = out_root / "ean13_manifest.csv"
    db_path = out_root / "ean13_manifest.db"

    df.to_excel(excel_path, index=False)
    df.to_csv(csv_path, index=False)

    conn = sqlite3.connect(db_path)
    df.to_sql("shipments", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

    print(f"Saved manifest: {excel_path}, {csv_path} & {db_path}")

    # Generate Image dataset
    manifest_records = []
    variants_per_code = 2

    for r in rows:
        awb = r["awb"]
        # 1. Clean EAN-13 (384x128) using render_ean13
        clean_arr = render_ean13(awb, size=(384, 128))
        clean_name = f"{awb}.png"
        cv2.imwrite(str(clean_dir / clean_name), clean_arr)
        manifest_records.append({
            "file": f"clean/{clean_name}",
            "kind": "clean",
            "damages": "none",
            "severity": 0.0,
        })

        # 2. Damaged EAN-13 (384x128) using existing damage()
        # Named <awb>_v1.png, <awb>_v2.png so evaluate-style truth parsing works
        for v in range(1, variants_per_code + 1):
            dmg_arr = damage(clean_arr, np_rng)
            dmg_name = f"{awb}_v{v}.png"
            cv2.imwrite(str(damaged_dir / dmg_name), dmg_arr)
            manifest_records.append({
                "file": f"damaged/{dmg_name}",
                "kind": "damaged",
                "damages": "standard_damage",
                "severity": 0.5,
            })

        # 3. Torn EAN-13 (384x128) using damage_with_info(torn=True)
        # Named <awb>_v1.png, <awb>_v2.png
        for v in range(1, variants_per_code + 1):
            torn_arr, applied_ops, sev = damage_with_info(clean_arr, np_rng, torn=True)
            torn_name = f"{awb}_v{v}.png"
            cv2.imwrite(str(torn_dir / torn_name), torn_arr)
            manifest_records.append({
                "file": f"torn/{torn_name}",
                "kind": "torn",
                "damages": "+".join(applied_ops),
                "severity": sev,
            })

        # 4. Full shipping label (800x560)
        label_img = make_label_image(r, clean_arr)
        label_img.save(labels_dir / f"{awb}.png")

        # 5. QR Code
        qr_data = json.dumps({
            "awb": awb, "carrier": r["carrier"], "to": r["consignee_name"],
            "address": f"{r['consignee_address']}, {r['consignee_city']} - {r['consignee_pin']}"
        })
        qr_obj = qrcode.QRCode(border=2, box_size=6)
        qr_obj.add_data(qr_data)
        qr_obj.make(fit=True)
        qr_obj.make_image(fill_color="black", back_color="white").save(qr_dir / f"{awb}.png")

    manifest_df = pd.DataFrame(manifest_records)
    manifest_df.to_csv(out_root / "manifest.csv", index=False)
    print(f"Generated {len(rows)} clean EAN-13 barcodes, {len(rows)*variants_per_code} damaged, "
          f"{len(rows)*variants_per_code} torn, labels & QR codes.")
    print(f"Saved dataset manifest with {len(manifest_df)} rows to {out_root / 'manifest.csv'}")
    print("Done!")


if __name__ == "__main__":
    main()
