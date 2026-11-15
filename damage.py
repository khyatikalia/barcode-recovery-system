"""Synthetic damage for clean barcode images (grayscale uint8, HxW).

Three groups, applied in this order: physical damage -> optics -> sensor.
Each op is picked at random, strength scaled by a random severity in [0.3, 1].
"""
import cv2
import numpy as np


# ---------- physical damage ----------
def _smudges(img, rng, s):
    h, w = img.shape
    for _ in range(int(rng.integers(1, 3 + int(4 * s)))):
        mask = np.zeros((h, w), np.float32)
        center = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        axes = (int(rng.integers(6, 21 + int(50 * s))), int(rng.integers(4, 11 + int(25 * s))))
        cv2.ellipse(mask, center, axes, float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), float(rng.uniform(2, 8)))
        alpha = float(rng.uniform(0.4, 0.5 + 0.5 * s))
        color = float(rng.uniform(0, 90)) if rng.random() < 0.7 else float(rng.uniform(170, 255))
        img = img * (1 - alpha * mask) + color * alpha * mask
    return img


def _scratches(img, rng, s):
    h, w = img.shape
    layer = img.copy()
    for _ in range(int(rng.integers(1, 3 + int(5 * s)))):
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        color = float(rng.uniform(0, 80)) if rng.random() < 0.5 else float(rng.uniform(190, 255))
        cv2.line(layer, p1, p2, color, int(rng.integers(1, 3 + int(2 * s))))
    alpha = float(rng.uniform(0.6, 1.0))
    return img * (1 - alpha) + layer * alpha


def _missing_bars(img, rng, s):
    """Ink dropout: whiten a narrow, partial-height strip (bars stay partly visible)."""
    h, w = img.shape
    for _ in range(int(rng.integers(1, 2 + int(4 * s)))):
        x0 = int(rng.integers(int(0.15 * w), int(0.85 * w)))
        bw = int(rng.integers(2, 5 + int(12 * s)))
        y0 = int(rng.integers(0, int(0.5 * h)))
        y1 = min(h, y0 + int(rng.uniform(0.25, 0.7) * h))
        a = float(rng.uniform(0.6, 1.0))
        img[y0:y1, x0:x0 + bw] = img[y0:y1, x0:x0 + bw] * (1 - a) + 255 * a
    return img


# ---------- optics ----------
def _blur(img, rng, s):
    return cv2.GaussianBlur(img, (0, 0), float(rng.uniform(0.7, 0.8 + 2.5 * s)))


def _motion_blur(img, rng, s):
    length = int(rng.integers(3, 6 + int(18 * s))) | 1
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0 / length
    return cv2.filter2D(img, -1, k)


def _low_res(img, rng, s):
    h, w = img.shape
    f = float(rng.uniform(1.3, 1.5 + 2.0 * s))
    small = cv2.resize(img, (max(8, int(w / f)), max(8, int(h / f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


# ---------- sensor / lighting ----------
def _shadow(img, rng, s):
    h, w = img.shape
    th = float(rng.uniform(0, 2 * np.pi))
    X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    ramp = np.cos(th) * X + np.sin(th) * Y
    ramp = (ramp - ramp.min()) / (ramp.max() - ramp.min() + 1e-6)
    strength = float(rng.uniform(0.2, 0.2 + 0.6 * s))
    return img * (1 - strength * ramp)


def _low_contrast(img, rng, s):
    lo = float(rng.uniform(0, 110 * s))
    hi = 255 - float(rng.uniform(0, 90 * s))
    return lo + img * (hi - lo) / 255.0


def _noise(img, rng, s):
    sigma = float(rng.uniform(4, 6 + 40 * s))
    return img + rng.normal(0, sigma, img.shape).astype(np.float32)


def _jpeg(img, rng, s):
    q = int(rng.integers(8, 70))
    _, buf = cv2.imencode(".jpg", np.clip(img, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float32)


ALL_OPS = [_smudges, _scratches, _missing_bars,
           _blur, _motion_blur, _low_res,
           _shadow, _low_contrast, _noise, _jpeg]


def damage(gray, rng, severity=None):
    """gray: uint8 HxW clean image. Returns damaged uint8 HxW."""
    s = float(rng.uniform(0.3, 1.0)) if severity is None else float(severity)
    img = gray.astype(np.float32).copy()
    chosen = [op for op in ALL_OPS if rng.random() < 0.5]
    if not chosen:
        chosen = [ALL_OPS[int(rng.integers(len(ALL_OPS)))]]
    for op in chosen:
        img = np.clip(op(img, rng, s), 0, 255)
    return img.astype(np.uint8)
