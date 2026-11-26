"""Guard-by-guard tracing wrapper around pipeline.run_pipeline()."""
import cv2

from pipeline import (DEFAULT_MIN_SCORE, MODEL_H, MODEL_W, consistency_score,
                      run_pipeline)


def _resize_for_consistency(gray_img):
    if gray_img is None:
        return None
    if gray_img.shape[:2] == (MODEL_H, MODEL_W):
        return gray_img
    return cv2.resize(gray_img, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)


def trace_pipeline(img, model=None, device="cpu", db_manifest=None):
    """Run pipeline.run_pipeline() unmodified and return a detailed guard trace."""
    out = run_pipeline(img, model=model, device=device, db_manifest=db_manifest)

    source = out["source"]
    candidate = out["candidate"]
    thr = float(getattr(model, "min_consistency", DEFAULT_MIN_SCORE))

    # Dual-decode guard only applies on the recovered path
    if source == "recovered":
        dual_decode_guard = True
    elif out["recovered"] is not None and candidate is None:
        dual_decode_guard = False
    else:
        dual_decode_guard = None

    # Consistency guard: run_pipeline only computes score on the recovered path
    side_score = None
    if source in ("direct", "enhanced") and candidate is not None:
        decoded_img = out["gray"] if source == "direct" else out["enhanced"]
        rec_proxy = _resize_for_consistency(decoded_img)
        if rec_proxy is not None:
            side_score = consistency_score(out["gray"], rec_proxy, candidate)
        consistency_applied = False
        consistency_passed = None
        effective_score = side_score
    elif source == "recovered" and out["score"] is not None:
        consistency_applied = True
        consistency_passed = bool(out["score"] >= thr)
        effective_score = float(out["score"])
    else:
        consistency_applied = False
        consistency_passed = None
        effective_score = None

    consistency_would_pass = (
        bool(effective_score >= thr) if effective_score is not None else None
    )

    # DB and routing guards
    db_guard = out["db_verified"]  # True, False, or None if not reached/no manifest
    if db_guard is True:
        routing_guard = out["route"] is not None
    else:
        routing_guard = None

    return {
        "decode_path": source,
        "candidate": candidate,
        "final_status": out["status"],
        "final_value": out["value"],
        "dual_decode_guard": dual_decode_guard,
        "check_digit_guard": out["check_digit"],
        "consistency_guard_applied": consistency_applied,
        "consistency_guard_passed": consistency_passed,
        "consistency_score": effective_score,
        "side_consistency_score": side_score,
        "consistency_threshold": thr,
        "consistency_would_pass": consistency_would_pass,
        "db_guard": db_guard,
        "routing_guard": routing_guard,
        "reason": out["reason"],
        "pipeline_out": out,
    }
