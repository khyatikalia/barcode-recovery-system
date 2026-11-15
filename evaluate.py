"""Evaluate on the held-out TEST split.   python evaluate.py --data data --model models/unet.pt"""
import argparse
from pathlib import Path

import cv2

from pipeline import decode, load_model, pick_device, run_pipeline, to_gray


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="models/unet.pt")
    args = ap.parse_args()

    files = sorted((Path(args.data) / "test" / "damaged").glob("*.png"))
    model, device = None, "cpu"
    if Path(args.model).exists():
        device = pick_device()
        model = load_model(args.model, device)
    else:
        print("No trained model found; evaluating the non-AI path only.\n")

    n = plain = no_ai_ok = 0
    ok = wrong = review = 0
    ai_used = ai_right = ai_wrong = ai_rej_wrong = ai_rej_right = 0
    for f in files:
        truth = f.stem.rsplit("_v", 1)[0]
        img = cv2.imread(str(f))
        n += 1
        r = decode(to_gray(img))
        plain += int(r is not None and r[0] == truth)
        r0 = run_pipeline(img, None)
        no_ai_ok += int(r0["status"] == "accepted" and r0["value"] == truth)

        r1 = run_pipeline(img, model, device) if model else r0
        if r1["status"] == "accepted":
            ok += int(r1["value"] == truth)
            wrong += int(r1["value"] != truth)
        else:
            review += 1
        if r1["source"] == "recovered":          # the AI produced a readable candidate
            ai_used += 1
            right = r1["candidate"] == truth
            ai_right += int(right)
            ai_wrong += int(not right)
            if r1["status"] != "accepted":
                ai_rej_right += int(right)
                ai_rej_wrong += int(not right)

    pct = lambda k, d=None: f"{100 * k / (d or n):5.1f}%"
    accepted = ok + wrong
    print(f"Test images (never seen in training): {n}\n")
    print("ACCURACY (correct value, automatically accepted)")
    print(f"  Plain ZXing on damaged image      : {pct(plain)}")
    print(f"  + OpenCV enhancement              : {pct(no_ai_ok)}")
    print(f"  + U-Net recovery + safety checks  : {pct(ok)}   <- final system\n")
    print("SAFETY")
    print(f"  WRONG value auto-accepted         : {pct(wrong)}   ({wrong} images)  <- hallucination errors")
    print(f"  Sent to manual review             : {pct(review)}")
    if accepted:
        print(f"  Precision of accepted values      : {pct(ok, accepted)}   (correct / accepted)")
    if model:
        print("\nU-NET DETAILS")
        print(f"  Images where AI gave a readable candidate : {ai_used}")
        print(f"    candidate correct                       : {ai_right}")
        print(f"    candidate WRONG (hallucinated)          : {ai_wrong}")
        print(f"  Safety check blocked: {ai_rej_wrong} wrong candidates, "
              f"but also {ai_rej_right} correct ones (sent to manual review)")


if __name__ == "__main__":
    main()
