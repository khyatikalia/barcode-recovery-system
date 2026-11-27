"""EAN-13 Barcode Recovery & Automated Courier Sorting System

Run with:
    streamlit run app.py
"""
import io
import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from generate_barcodes import render_ean13
from pipeline import (DEFAULT_MIN_SCORE, PIN_ROUTING_TABLE, check_digit_ok,
                      load_model, pick_device, route_by_pincode, run_pipeline)

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "unet.pt"
if not MODEL_PATH.exists() and (ROOT / "models" / "unet_v2.pt").exists():
    MODEL_PATH = ROOT / "models" / "unet_v2.pt"

DATASET_DIR = ROOT / "data" / "ean13_courier"
DB_PATH = DATASET_DIR / "ean13_manifest.db"

st.set_page_config(
    page_title="EAN-13 Barcode Recovery & Courier Sorting",
    page_icon="📦",
    layout="wide"
)

st.title("📦 EAN-13 Barcode Recovery & Automated Courier Sorting")
st.caption(
    "End-to-End AI Logistics Pipeline: "
    "Enhancement (OpenCV) ➔ ZXing ➔ U-Net Reconstruction ➔ Check-Digit Guard ➔ "
    "Consistency Guard ➔ Database Manifest Guard ➔ PIN-to-Lane Sorting."
)


@st.cache_resource
def _load_cached_model(path_str: str, dev: str):
    return load_model(path_str, dev)


def get_model():
    p = ROOT / "models" / "unet.pt"
    if not p.exists():
        p = ROOT / "models" / "unet_v2.pt"
    if not p.exists():
        return None, "cpu"
    dev = pick_device()
    return _load_cached_model(str(p), dev), dev


def get_db_connection():
    if DB_PATH.exists():
        return sqlite3.connect(str(DB_PATH), check_same_thread=False)
    return None


model, device = get_model()
if model is None:
    st.error(
        "⚠️ AI recovery model checkpoint not found at 'models/unet.pt' or 'models/unet_v2.pt'. "
        "AI reconstruction is disabled. Standard OpenCV enhancement, ZXing decoding, and "
        "Database/PIN routing guards will continue to operate."
    )

db_conn = get_db_connection()

tab_scan, tab_manifest, tab_routing = st.tabs([
    "1. Live Scanner & Sorting Pipeline",
    "2. Courier Manifest Database",
    "3. PIN-to-Lane Routing Table"
])

# -----------------------------------------------------------------------------
# TAB 1: Live Scanner & Routing Pipeline
# -----------------------------------------------------------------------------
with tab_scan:
    st.subheader("Scan Barcode / Label & Execute Auto-Routing")

    source_mode = st.radio(
        "Choose Input Source:",
        ["🎯 Pick Test Sample from Dataset", "📁 Upload Image File", "📷 Live Webcam / Camera"],
        horizontal=True
    )

    img_bgr = None

    if source_mode == "🎯 Pick Test Sample from Dataset":
        sample_options = {}
        if (DATASET_DIR / "clean").exists():
            for p in sorted((DATASET_DIR / "clean").glob("*.png"))[:3]:
                sample_options[f"Clean EAN-13 Barcode: {p.stem}"] = str(p)
        if (DATASET_DIR / "damaged").exists():
            for p in sorted((DATASET_DIR / "damaged").glob("*.png"))[:3]:
                sample_options[f"Damaged EAN-13 Barcode: {p.name}"] = str(p)
        if (DATASET_DIR / "torn").exists():
            for p in sorted((DATASET_DIR / "torn").glob("*.png"))[:3]:
                sample_options[f"Torn EAN-13 Barcode: {p.name}"] = str(p)
        if (DATASET_DIR / "labels").exists():
            for p in sorted((DATASET_DIR / "labels").glob("*.png"))[:2]:
                sample_options[f"Courier Shipping Label (with EAN-13 & QR): {p.stem}"] = str(p)

        # Hallucination test case: in-memory valid EAN-13 not in database
        sample_options["⚠️ Anti-Hallucination Test: Valid EAN-13 not in Manifest (8901234567890)"] = "__hallucination_test__"

        chosen_label = st.selectbox("Select Sample Image:", list(sample_options.keys()))
        if chosen_label:
            target_path = sample_options[chosen_label]
            if target_path == "__hallucination_test__":
                # Render in memory using render_ean13("890123456789")
                test_code12 = "890123456789"
                test_arr = render_ean13(test_code12, size=(384, 128))
                img_bgr = cv2.cvtColor(test_arr, cv2.COLOR_GRAY2BGR)

                # Verify rendered barcode value is not in manifest database
                if db_conn:
                    cur = db_conn.cursor()
                    cur.execute("SELECT awb FROM shipments WHERE awb = '8901234567890'")
                    found = cur.fetchone()
                    assert found is None, "Anti-hallucination test barcode must NOT be present in active manifest database!"
            elif Path(target_path).exists():
                img_bgr = cv2.imread(target_path)
            else:
                st.info(
                    "Sample file not found. Generate the dataset by running:\n"
                    "`python generate_ean13_courier_manifest.py --count 200 --sequential`"
                )

    elif source_mode == "📁 Upload Image File":
        up_file = st.file_uploader("Upload an EAN-13 barcode or shipping label", type=["png", "jpg", "jpeg", "bmp"])
        if up_file:
            img_bgr = cv2.imdecode(np.frombuffer(up_file.read(), np.uint8), cv2.IMREAD_COLOR)

    else:
        cam_file = st.camera_input("Capture barcode with camera")
        if cam_file:
            img_bgr = cv2.imdecode(np.frombuffer(cam_file.read(), np.uint8), cv2.IMREAD_COLOR)

    if img_bgr is not None:
        st.markdown("---")
        with st.spinner("Executing Pipeline Guards & Routing..."):
            res = run_pipeline(img_bgr, model=model, device=device, db_manifest=db_conn)

        # Display Pipeline Image Stages
        c1, c2, c3 = st.columns(3)
        c1.image(res["gray"], caption="1. Input Image (Grayscale)", clamp=True, width="stretch")
        if res["enhanced"] is not None:
            c2.image(res["enhanced"], caption="2. OpenCV CLAHE Enhancement", clamp=True, width="stretch")
        else:
            c2.info("OpenCV enhancement not required (direct read successful).")

        if res["recovered"] is not None:
            c3.image(res["recovered"], caption="3. U-Net AI Reconstruction", clamp=True, width="stretch")
        else:
            c3.info("U-Net recovery not needed or no candidate generated.")

        st.markdown("---")

        # Pipeline Decision Banner
        if res["status"] == "accepted" and res.get("route") is not None:
            route = res["route"]
            st.success(
                f"### ✅ DECISION: AUTO-ROUTE ➔ **{route['lane']}** ({route['hub']})\n"
                f"**EAN-13 / AWB:** `{res['value']}` | **Destination PIN:** `{route['pin']}` ({route['region']})"
            )
        elif res["status"] == "accepted":
            st.warning(
                f"### ⚠️ DECISION: ACCEPTED (NO ROUTE - MANUAL DISPATCH)\n"
                f"**EAN-13 / AWB:** `{res['value']}` (Check digit valid, but delivery PIN could not be routed)"
            )
        else:
            st.error(
                f"### ⚠️ DECISION: MANUAL REVIEW REQUIRED\n"
                f"**Reason:** {res['reason']}"
            )
            if res.get("candidate"):
                st.warning(f"Candidate value read: `{res['candidate']}` (BLOCKED from auto-routing by safety guards)")

        # Diagnostic Metric Badges
        SOURCE_NAMES = {
            "direct": "Plain ZXing (Direct)",
            "enhanced": "OpenCV Enhancement",
            "recovered": "U-Net Reconstruction + ZXing",
            None: "Unreadable",
        }
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Format", res.get("format") or "N/A")
        m2.metric("Pipeline Stage", SOURCE_NAMES.get(res.get("source"), "None"))
        cd = res.get("check_digit")
        m3.metric("Check Digit (GS1)", "VALID" if cd else ("INVALID" if cd is False else "N/A"))
        score = res.get("score")
        m4.metric("Consistency", f"{score:.2f}" if score is not None else "N/A (Direct Read)")
        db_match = res.get("db_verified")
        m5.metric("Database Match", "VERIFIED" if db_match else ("NOT FOUND" if db_match is False else "N/A"))

        # Render Shipment Dispatch Card if present in database
        if res.get("shipment"):
            s = res["shipment"]
            st.markdown("### 📋 Verified Shipment Dispatch Details")
            col_left, col_right = st.columns(2)
            with col_left:
                st.info(
                    f"🚚 **Courier Partner:** {s.get('carrier')}\n\n"
                    f"🏷️ **Tracking (EAN-13):** `{s.get('awb')}`\n\n"
                    f"📦 **Order ID:** `{s.get('order_id')}`\n\n"
                    f"⚖️ **Weight:** {s.get('weight_kg')} kg | **Items:** {s.get('contents')}"
                )
            with col_right:
                pay_str = f"COD (₹{s.get('cod_amount')})" if str(s.get('payment_mode')).upper() == "COD" else s.get('payment_mode')
                st.warning(
                    f"💳 **Payment:** {pay_str}\n\n"
                    f"📅 **Dispatch Date:** {s.get('ship_date')}\n\n"
                    f"🏢 **Origin:** {s.get('shipper_name')} ({s.get('shipper_city')} - {s.get('shipper_pin')})"
                )

            st.markdown(
                f"**📍 Delivery Destination:** **{s.get('consignee_name')}** (Ph: `{s.get('consignee_phone')}`)\n\n"
                f"{s.get('consignee_address')}, {s.get('consignee_city')}, {s.get('consignee_state')} - **{s.get('consignee_pin')}**"
            )

# -----------------------------------------------------------------------------
# TAB 2: Courier Manifest Database
# -----------------------------------------------------------------------------
with tab_manifest:
    st.subheader("Courier Manifest Database & Shipment Records")

    # File uploader to accept user's Excel or CSV manifest
    with st.expander("📤 Upload Custom Courier Manifest File (.xlsx / .csv)", expanded=False):
        uploaded_doc = st.file_uploader(
            "Upload Excel or CSV file containing shipment barcode records:",
            type=["xlsx", "xls", "csv"],
            key="user_manifest_upload"
        )
        if uploaded_doc is not None:
            try:
                if uploaded_doc.name.endswith(".csv"):
                    new_df = pd.read_csv(uploaded_doc)
                else:
                    new_df = pd.read_excel(uploaded_doc)

                # Normalize column names: ensure awb / barcode exists
                col_map = {}
                for c in new_df.columns:
                    cl = str(c).strip().lower()
                    if cl in ("barcode", "ean13", "tracking_number", "tracking_id") and "awb" not in new_df.columns:
                        col_map[c] = "awb"
                    elif cl in ("pin", "pincode", "postal_code") and "consignee_pin" not in new_df.columns:
                        col_map[c] = "consignee_pin"
                if col_map:
                    new_df = new_df.rename(columns=col_map)

                if "awb" not in new_df.columns:
                    st.error("Uploaded document must contain an 'awb' or 'barcode' column.")
                else:
                    DATASET_DIR.mkdir(parents=True, exist_ok=True)
                    store_conn = sqlite3.connect(str(DB_PATH))
                    new_df.to_sql("shipments", store_conn, if_exists="replace", index=False)
                    store_conn.commit()
                    store_conn.close()
                    new_df.to_excel(DATASET_DIR / "ean13_manifest.xlsx", index=False)
                    new_df.to_csv(DATASET_DIR / "ean13_manifest.csv", index=False)
                    st.success(f"✅ Successfully stored {len(new_df)} shipment records from '{uploaded_doc.name}' into active database!")
                    db_conn = get_db_connection()
            except Exception as e:
                st.error(f"Error processing uploaded document: {e}")

    if db_conn:
        df_manifest = pd.read_sql_query("SELECT * FROM shipments", db_conn)
        st.write(f"Total Active Registered Shipments in Database: **{len(df_manifest)}**")

        # Metric Summary
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Registered Parcels", len(df_manifest))
        cod_series = df_manifest[df_manifest["payment_mode"].astype(str).str.upper() == "COD"]["cod_amount"] if "payment_mode" in df_manifest.columns else []
        cod_sum = pd.to_numeric(cod_series, errors="coerce").fillna(0).sum() if len(cod_series) else 0
        c2.metric("Total COD Amount", f"₹{cod_sum:,.0f}")
        c3.metric("Carriers", df_manifest["carrier"].nunique() if "carrier" in df_manifest.columns else 1)
        c4.metric("Sorting Lanes", df_manifest["sorting_lane"].nunique() if "sorting_lane" in df_manifest.columns else "Auto")

        search_q = st.text_input("Search Manifest (by EAN-13, Recipient, City, PIN, Hub):")
        if search_q:
            mask = df_manifest.apply(lambda col: col.astype(str).str.contains(search_q, case=False, regex=False)).any(axis=1)
            df_manifest = df_manifest[mask]

        st.dataframe(df_manifest, width="stretch")

        # Download button
        buf = io.BytesIO()
        df_manifest.to_excel(buf, index=False)
        st.download_button(
            "⬇ Download Active Manifest as Excel (.xlsx)",
            buf.getvalue(),
            "ean13_courier_manifest.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("Database not connected. Please upload a manifest file above or run `generate_ean13_courier_manifest.py`.")

# -----------------------------------------------------------------------------
# TAB 3: PIN-to-Lane Routing Table
# -----------------------------------------------------------------------------
with tab_routing:
    st.subheader("Postal Sorting District & Lane Configuration")
    st.markdown("""
    In the Indian postal and logistics network, the first 3 digits of a 6-digit PIN code define the **Sorting District**.
    The sorting router automates package routing by mapping PIN prefixes directly to sorting lanes and regional transit hubs:
    """)
    route_rows = []
    for prefix, (lane, hub, zone) in PIN_ROUTING_TABLE.items():
        route_rows.append({
            "PIN Prefix (3 Digits)": f"{prefix}xxx",
            "Sorting Lane": lane,
            "Primary Sorting Hub": hub,
            "Logistics Zone": zone,
        })
    st.table(pd.DataFrame(route_rows))
