"""Streamlit app.   streamlit run app.py"""
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from pipeline import load_model, pick_device, run_pipeline

MODEL_PATH = Path("models/unet.pt")
SOURCE_TEXT = {
    "direct": "Read directly",
    "enhanced": "Read after OpenCV enhancement",
    "recovered": "Read after AI recovery (U-Net)",
}

st.set_page_config(page_title="Barcode Recovery", layout="wide")
st.title("Barcode Recovery")
st.caption("Upload a cropped EAN-13 barcode image. Standard decoding runs first; the AI model is "
           "used only if that fails, and its output is checked before being trusted.")


@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        return None, "cpu"
    device = pick_device()
    return load_model(MODEL_PATH, device), device


model, device = get_model()
if model is None:
    st.warning("No trained model found at models/unet.pt. AI recovery is disabled.")

up = st.file_uploader("Barcode image", type=["png", "jpg", "jpeg", "bmp"])
if up is not None:
    img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        st.error("Could not read this image file.")
        st.stop()

    with st.spinner("Processing..."):
        res = run_pipeline(img, model, device)

    cols = st.columns(3)
    cols[0].image(res["gray"], caption="Uploaded (grayscale)", clamp=True, use_container_width=True)
    if res["enhanced"] is not None:
        cols[1].image(res["enhanced"], caption="After enhancement", clamp=True, use_container_width=True)
    if res["recovered"] is not None:
        cols[2].image(res["recovered"], caption="AI-recovered image", clamp=True, use_container_width=True)

    st.divider()
    if res["status"] == "accepted":
        st.success(f"Decoded value: {res['value']}   |   Decision: AUTO-ROUTE")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Format", res["format"])
        c2.metric("Path used", SOURCE_TEXT[res["source"]])
        cd = res["check_digit"]
        c3.metric("Check digit", "Valid" if cd else ("N/A" if cd is None else "FAILED"))
        c4.metric("Consistency score", f"{res['score']:.2f}" if res["score"] is not None else "n/a (direct read)")
        st.caption(res["reason"])
    else:
        st.error("Decision: MANUAL VERIFICATION. This barcode was not trusted for automatic routing.")
        st.write(res["reason"])
        if res["candidate"]:
            st.warning(f"AI suggestion (NOT verified, do not route automatically): {res['candidate']}")
