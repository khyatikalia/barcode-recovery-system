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


# ---------- tear damage ----------
def _tear_side(img, rng, s):
    """Jagged torn edge on left or right edge. Torn area is near-white paper with edge shadow."""
    h, w = img.shape
    is_left = rng.random() < 0.5
    tear_w = int(rng.uniform(0.12, 0.18 + 0.25 * s) * w)
    n_pts = int(rng.integers(15, 26))
    ys = np.linspace(0, h, n_pts)
    jitter = rng.normal(0, max(2.0, 5.0 * s), n_pts)
    xs = np.clip(tear_w + jitter, 4, w - 4)
    if is_left:
        pts = [(int(x), int(y)) for x, y in zip(xs, ys)]
        poly = np.array([(0, 0)] + pts + [(0, h)], dtype=np.int32)
    else:
        pts = [(int(w - x), int(y)) for x, y in zip(xs, ys)]
        poly = np.array([(w, 0)] + pts + [(w, h)], dtype=np.int32)
    paper = float(rng.uniform(235, 255))
    cv2.fillPoly(img, [poly], paper)
    shadow_color = float(rng.uniform(130, 160))
    cv2.polylines(img, [np.array(pts, dtype=np.int32)], isClosed=False, color=shadow_color, thickness=int(rng.integers(1, 3)))
    return img


def _tear_corner(img, rng, s):
    """Jagged torn corner. Torn area is near-white paper with thin darker edge shadow."""
    h, w = img.shape
    corner_idx = int(rng.integers(0, 4))
    tear_w = int(rng.uniform(0.15, 0.25 + 0.3 * s) * w)
    tear_h = int(rng.uniform(0.2, 0.35 + 0.4 * s) * h)
    n_pts = int(rng.integers(12, 22))

    t = np.linspace(0, 1, n_pts)
    jitter = rng.normal(0, max(2.0, 4.0 * s), n_pts)
    norm = max(float(np.hypot(tear_w, tear_h)), 1e-5)
    nx, ny = tear_h / norm, tear_w / norm
    xs = t * tear_w + jitter * nx
    ys = (1 - t) * tear_h + jitter * ny

    if corner_idx == 0:  # Top-Left
        pts = [(int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))) for x, y in zip(xs, ys)]
        poly = np.array([(0, 0)] + pts, dtype=np.int32)
    elif corner_idx == 1:  # Top-Right
        pts = [(int(np.clip(w - x, 0, w - 1)), int(np.clip(y, 0, h - 1))) for x, y in zip(xs, ys)]
        poly = np.array([(w, 0)] + pts, dtype=np.int32)
    elif corner_idx == 2:  # Bottom-Left
        pts = [(int(np.clip(x, 0, w - 1)), int(np.clip(h - y, 0, h - 1))) for x, y in zip(xs, ys)]
        poly = np.array([(0, h)] + pts, dtype=np.int32)
    else:  # Bottom-Right
        pts = [(int(np.clip(w - x, 0, w - 1)), int(np.clip(h - y, 0, h - 1))) for x, y in zip(xs, ys)]
        poly = np.array([(w, h)] + pts, dtype=np.int32)

    paper = float(rng.uniform(235, 255))
    cv2.fillPoly(img, [poly], paper)
    shadow_color = float(rng.uniform(130, 160))
    cv2.polylines(img, [np.array(pts, dtype=np.int32)], isClosed=False, color=shadow_color, thickness=int(rng.integers(1, 3)))
    return img


ALL_OPS = [_smudges, _scratches, _missing_bars,
           _blur, _motion_blur, _low_res,
           _shadow, _low_contrast, _noise, _jpeg]
TEAR_OPS = [_tear_side, _tear_corner]


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


def damage_with_info(gray, rng, severity=None, torn=False):
    """Apply synthetic damage and return (damaged_uint8_img, applied_op_names, severity).

    If torn=True: always applies one tear operation plus 0-2 of the existing ops.
    If torn=False: uses the standard distribution matching damage().
    """
    s = float(rng.uniform(0.3, 1.0)) if severity is None else float(severity)
    img = gray.astype(np.float32).copy()

    if torn:
        tear_op = TEAR_OPS[int(rng.integers(len(TEAR_OPS)))]
        num_extra = int(rng.integers(0, 3))  # 0, 1, or 2 extra ops
        if num_extra > 0:
            extra_idx = rng.choice(len(ALL_OPS), size=num_extra, replace=False)
            extra_ops = [ALL_OPS[i] for i in extra_idx]
        else:
            extra_ops = []
        chosen = extra_ops + [tear_op]
    else:
        chosen = [op for op in ALL_OPS if rng.random() < 0.5]
        if not chosen:
            chosen = [ALL_OPS[int(rng.integers(len(ALL_OPS)))]]

    for op in chosen:
        img = np.clip(op(img, rng, s), 0, 255)

    op_names = [op.__name__.lstrip("_") for op in chosen]
    return img.astype(np.uint8), op_names, round(s, 3)
