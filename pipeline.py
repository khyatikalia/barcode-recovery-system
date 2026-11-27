"""Core pipeline:
enhance -> ZXing decode -> (U-Net recovery -> ZXing again) -> verification -> accept / manual review.

Anti-hallucination design
-------------------------
The U-Net only restores an IMAGE; the value always comes from ZXing reading that image.
A recovered value is accepted automatically only if ALL of these hold:
  1. ZXing reads it as EAN-13 (which has its own check digit), and the check digit is valid.
  2. Reading the raw and the binarized recovered image does not give two different values.
  3. Consistency check: the bars that SHOULD exist for that value agree with the ORIGINAL
     damaged input (not just with the AI output). A value the model "invented" does not
     match the evidence still visible in the input, so it scores low.
Anything else goes to manual review. Directly-read values (no AI) are accepted as-is.
"""
import cv2
import numpy as np
import zxingcpp
from stdnum import ean

MODEL_W, MODEL_H = 384, 128
EAN13 = zxingcpp.BarcodeFormat.EAN13
DEFAULT_MIN_SCORE = 0.60  # replaced by the value calibrated during training (see train.py)


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def enhance(gray):
    """Classical enhancement: denoise, local contrast (CLAHE), unsharp mask."""
    g = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(g)
    soft = cv2.GaussianBlur(g, (0, 0), 1.5)
    return cv2.addWeighted(g, 1.5, soft, -0.5, 0)


def decode(gray):
    """Returns (text, format_name) or None. Phase 1 reads EAN-13 only."""
    res = zxingcpp.read_barcodes(gray, formats=EAN13)
    if not res:
        return None
    fmt = res[0].format
    return res[0].text, getattr(fmt, "name", str(fmt))


def decode_recovered(gray):
    """Read the restored image as-is and binarized (Otsu). Conflicting answers -> None."""
    found = []
    r = decode(gray)
    if r:
        found.append(r)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    r2 = decode(binary)
    if r2:
        found.append(r2)
    if not found or len({v for v, _ in found}) > 1:
        return None
    return found[0]


def check_digit_ok(value):
    """True/False for numeric EAN-13 values, None if not applicable."""
    if not value or not value.isdigit() or len(value) != 13:
        return False
    try:
        return bool(ean.is_valid(value))
    except Exception:
        return False


# ---------- PIN-to-Sorting-Lane Routing Table ----------
PIN_ROUTING_TABLE = {
    "110": ("Lane 1 (North)", "Delhi NCR Hub", "Delhi NCR"),
    "400": ("Lane 2 (West)", "Mumbai Central Hub", "Maharashtra"),
    "560": ("Lane 3 (South)", "Bengaluru Sorting Hub", "Karnataka"),
    "600": ("Lane 4 (South)", "Chennai Sorting Hub", "Tamil Nadu"),
    "700": ("Lane 5 (East)", "Kolkata Sorting Hub", "West Bengal"),
    "500": ("Lane 6 (South)", "Hyderabad Sorting Hub", "Telangana"),
    "411": ("Lane 7 (West)", "Pune Sorting Hub", "Maharashtra"),
    "380": ("Lane 8 (West)", "Ahmedabad Sorting Hub", "Gujarat"),
    "302": ("Lane 9 (North)", "Jaipur Sorting Hub", "Rajasthan"),
    "226": ("Lane 10 (North)", "Lucknow Sorting Hub", "Uttar Pradesh"),
    "682": ("Lane 11 (South)", "Kochi Sorting Hub", "Kerala"),
    "452": ("Lane 12 (Central)", "Indore Sorting Hub", "Madhya Pradesh"),
}


def route_by_pincode(pin: str):
    """Map 6-digit Indian PIN (first 3 digits = sorting district) to sorting lane.
    
    Returns None if PIN is missing, not 6 digits, or not in PIN_ROUTING_TABLE.
    """
    if pin is None:
        return None
    pin_str = str(pin).strip()
    if len(pin_str) != 6 or not pin_str.isdigit():
        return None
    prefix = pin_str[:3]
    if prefix in PIN_ROUTING_TABLE:
        lane, hub, region = PIN_ROUTING_TABLE[prefix]
        return {"lane": lane, "hub": hub, "region": region, "pin_prefix": prefix, "pin": pin_str}
    return None


def _apply_db_and_route(out: dict, value: str, db_manifest):
    """Verify candidate against shipment database/manifest and compute sorting lane."""
    if db_manifest is None or not value:
        return

    rec = None
    if isinstance(db_manifest, dict):
        rec = db_manifest.get(value)
    elif hasattr(db_manifest, "execute"):
        cur = db_manifest.execute("SELECT * FROM shipments WHERE awb = ?", (value,))
        col_names = [d[0] for d in cur.description] if cur.description else []
        row = cur.fetchone()
        if row:
            rec = dict(zip(col_names, row))
    elif isinstance(db_manifest, (set, list, tuple)):
        if value in db_manifest:
            rec = {"awb": value}
    elif callable(db_manifest):
        rec = db_manifest(value)

    if rec is not None:
        out["db_verified"] = True
        out["shipment"] = rec
        pin = rec.get("consignee_pin") or rec.get("pin") or ""
        route = route_by_pincode(pin)
        if route is None:
            out["status"] = "manual_review"
            out["value"] = None
            out["route"] = None
            out["reason"] = "PIN missing or not in routing table"
        else:
            out["route"] = route
            out["reason"] += f" [Database Guard: Verified in manifest. Routed to {route['lane']}]."
    else:
        out["db_verified"] = False
        out["status"] = "manual_review"
        out["value"] = None
        out["reason"] += f" [Database Guard Block: Decoded EAN-13 '{value}' is not present in registered manifest; blocked for manual verification]."


# ---------- consistency check against the ORIGINAL input ----------
def _ideal_modules(value):
    from barcode import EAN13 as _EAN13
    return _EAN13(value[:12]).build()[0]  # 95 characters of '0'/'1'


def _bar_extent(rec):
    cols = np.where((rec < 128).mean(axis=0) > 0.3)[0]
    if len(cols) < 50:
        return None
    return int(cols[0]), int(cols[-1])


def consistency_score(gray, rec, value):
    """Correlation (-1..1) between the bar pattern of `value` and the original input.

    Position and scale of the bars come from the restored image `rec`; the evidence
    comes from `gray`, the damaged input. High = input supports this value.
    """
    ext = _bar_extent(rec)
    if ext is None:
        return 0.0
    c0, c1 = ext
    mw = (c1 - c0 + 1) / 95.0
    mods = _ideal_modules(value)

    x = cv2.resize(gray, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = 255.0 - x
    d = d - cv2.GaussianBlur(d, (0, 0), 25)      # remove shadows / uneven lighting
    prof = d[20:108, :].mean(axis=0)              # darkness per column (central rows)

    cols = np.arange(MODEL_W)
    idx = np.floor((cols + 0.5 - c0) / mw).astype(int)
    ideal = np.zeros(MODEL_W, np.float32)
    valid = (idx >= 0) & (idx < 95)
    ideal[valid] = [1.0 if mods[i] == "1" else 0.0 for i in idx[valid]]

    lo, hi = max(0, int(c0 - 2 * mw)), min(MODEL_W, int(c1 + 2 * mw) + 1)
    a, b = prof[lo:hi], ideal[lo:hi]
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ---------- model helpers (torch imported lazily) ----------
def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(path, device="cpu"):
    import torch
    from model import UNet
    ckpt = torch.load(path, map_location=device)
    m = UNet(base=ckpt.get("base", 32))
    m.load_state_dict(ckpt["model"])
    m.min_consistency = float(ckpt.get("min_consistency", DEFAULT_MIN_SCORE))
    return m.to(device).eval()


def recover(model, gray, device="cpu"):
    import torch
    x = cv2.resize(gray, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(x).float().div(255).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        y = model(t)[0, 0].cpu().numpy()
    return (y * 255).clip(0, 255).astype(np.uint8)


def run_pipeline(img, model=None, device="cpu", db_manifest=None):
    """
    Returns a dict:
      status      'accepted' (safe to auto-route) or 'manual_review'
      value       accepted value (None when sent to manual review)
      candidate   value ZXing read, even if rejected by the guards (for evaluation)
      source      'direct' | 'enhanced' | 'recovered' | None
      score       consistency score (recovered path only)
      reason      short explanation
      route       routing lane, hub and district mapping (if db_manifest provides PIN)
      db_verified True/False/None
      shipment    dict of shipment record if matched in manifest
    """
    gray = to_gray(img)
    out = {"status": "manual_review", "value": None, "candidate": None, "format": None,
           "source": None, "check_digit": None, "score": None, "reason": "",
           "gray": gray, "enhanced": None, "recovered": None,
           "route": None, "db_verified": None, "shipment": None}

    r = decode(gray)
    source = "direct"
    if r is None:
        out["enhanced"] = enhance(gray)
        r, source = decode(out["enhanced"]), "enhanced"

    if r is not None:  # read from the real pixels, no AI involved
        cd = check_digit_ok(r[0])
        out.update(value=r[0], candidate=r[0], format=r[1], source=source,
                   check_digit=cd, status="accepted" if cd else "manual_review",
                   reason="Read directly from the image (no AI used)." if cd else "Check digit invalid.")
        if out["status"] == "accepted":
            _apply_db_and_route(out, r[0], db_manifest)
        return out

    if model is None:
        out["reason"] = "Could not be read and no recovery model is loaded."
        return out

    rec = recover(model, gray, device)
    out["recovered"] = rec
    r = decode_recovered(rec)
    if r is None:
        out["reason"] = "AI recovery could not produce a readable, unambiguous barcode."
        return out

    value = r[0]
    out.update(candidate=value, format=r[1], source="recovered", check_digit=check_digit_ok(value))
    if out["check_digit"] is False:
        out["reason"] = "Check digit invalid."
        return out
    score = consistency_score(gray, rec, value)
    out["score"] = score
    thr = getattr(model, "min_consistency", DEFAULT_MIN_SCORE)
    if score < thr:
        out["reason"] = (f"Recovered value does not match the original image well "
                         f"(score {score:.2f} < {thr:.2f}); possible AI error.")
        return out
    out.update(value=value, status="accepted",
               reason=f"Recovered by AI; consistent with the original image (score {score:.2f}).")
    _apply_db_and_route(out, value, db_manifest)
    return out
