import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
import mysql.connector
from mysql.connector import Error
import pandas as pd
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🩺",
    layout="wide"
)

IMG_SIZE    = (224, 224)
CLASS_NAMES = ["benign", "malignant"]
MODEL_PATH  = "breast_cancer_vgg16.keras"

# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE CONNECTION & SETUP
# ══════════════════════════════════════════════════════════════════════════════

DB_CONFIG = {
    "host":     "localhost",
    "user":     "root",
    "password": "admin",
    "database": "breast_cancer_db"
}

def get_connection():
    return mysql.connector.connect(**DB_CONFIG)

def init_db():
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id               INT          NOT NULL AUTO_INCREMENT,
            image_name       VARCHAR(255) NOT NULL,
            predicted_class  ENUM('Benign','Malignant') NOT NULL,
            confidence_score FLOAT        NOT NULL,
            model_version    VARCHAR(50)  NOT NULL DEFAULT 'VGG16-v1',
            is_verified      TINYINT(1)   NOT NULL DEFAULT 0,
            notes            TEXT,
            timestamp        DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (id),
            CHECK (confidence_score BETWEEN 0.0 AND 1.0)
        ) ENGINE=InnoDB;
    """)
    conn.commit()
    cur.close()
    conn.close()

try:
    init_db()
    DB_OK = True
except Error as e:
    DB_OK  = False
    DB_ERR = str(e)

# ══════════════════════════════════════════════════════════════════════════════
#  CRUD FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── CREATE ────────────────────────────────────────────────────────────────────
def db_insert_prediction(image_name, predicted_class, confidence_score):
    conn = get_connection()
    cur  = conn.cursor()
    sql  = """
        INSERT INTO prediction_logs
            (image_name, predicted_class, confidence_score)
        VALUES (%s, %s, %s)
    """
    cur.execute(sql, (image_name, predicted_class, round(confidence_score, 4)))
    conn.commit()
    new_id = cur.lastrowid
    cur.close()
    conn.close()
    return new_id

# ── READ ──────────────────────────────────────────────────────────────────────
def db_get_all(filter_class=None, min_conf=0.0, only_unverified=False):
    conn   = get_connection()
    cur    = conn.cursor(dictionary=True)
    where  = ["confidence_score >= %s"]
    params = [min_conf]
    if filter_class and filter_class != "All":
        where.append("predicted_class = %s")
        params.append(filter_class)
    if only_unverified:
        where.append("is_verified = 0")
    sql = f"""
        SELECT id, image_name, predicted_class,
               ROUND(confidence_score * 100, 2) AS confidence_pct,
               model_version, is_verified, notes, timestamp
        FROM prediction_logs
        WHERE {' AND '.join(where)}
        ORDER BY timestamp DESC
    """
    cur.execute(sql, params)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def db_get_by_id(pred_id):
    conn = get_connection()
    cur  = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM prediction_logs WHERE id = %s", (pred_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row

def db_summary():
    conn = get_connection()
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT
            predicted_class,
            COUNT(*)                              AS total,
            SUM(is_verified)                      AS verified,
            COUNT(*) - SUM(is_verified)           AS pending,
            ROUND(AVG(confidence_score)*100, 2)   AS avg_conf_pct
        FROM prediction_logs
        GROUP BY predicted_class
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

# ── UPDATE ────────────────────────────────────────────────────────────────────
def db_update_prediction(pred_id, new_class, new_conf, notes):
    conn = get_connection()
    cur  = conn.cursor()
    sql  = """
        UPDATE prediction_logs
        SET predicted_class  = %s,
            confidence_score = %s,
            notes            = %s,
            is_verified      = 1
        WHERE id = %s
    """
    cur.execute(sql, (new_class, round(new_conf / 100, 4), notes, pred_id))
    conn.commit()
    affected = cur.rowcount
    cur.close()
    conn.close()
    return affected

# ── DELETE ────────────────────────────────────────────────────────────────────
def db_delete_prediction(pred_id):
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("DELETE FROM prediction_logs WHERE id = %s", (pred_id,))
    conn.commit()
    affected = cur.rowcount
    cur.close()
    conn.close()
    return affected

def db_delete_unverified_low_conf(threshold):
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("""
        DELETE FROM prediction_logs
        WHERE is_verified = 0 AND confidence_score < %s
    """, (threshold / 100,))
    conn.commit()
    affected = cur.rowcount
    cur.close()
    conn.close()
    return affected

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(MODEL_PATH)

def predict(img_array, model):
    arr  = np.array(img_array.resize(IMG_SIZE).convert("RGB"), dtype=np.float32)
    arr  = preprocess_input(arr)
    arr  = np.expand_dims(arr, axis=0)
    probs     = model.predict(arr, verbose=0)[0]
    pred_idx  = int(np.argmax(probs))
    return CLASS_NAMES[pred_idx], float(probs[pred_idx]) * 100, probs

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Breast_cancer_cell_line.jpg/320px-Breast_cancer_cell_line.jpg",
        use_column_width=True
    )
    st.markdown("## 🩺 About")
    st.info(
        "AI tool classifying breast ultrasound images as **Benign** or "
        "**Malignant** using a VGG16 CNN trained on the BUSI dataset.\n\n"
        "⚠️ *For educational/research use only.*"
    )
    st.markdown("**Model:** VGG16 Transfer Learning")
    st.markdown("**Dataset:** BUSI (780 images)")
    st.markdown("**Classes:** Benign | Malignant")
    st.divider()
    if DB_OK:
        st.success("🟢 MySQL Connected")
    else:
        st.error(f"🔴 MySQL Error:\n{DB_ERR}")

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════

st.title("🩺 Breast Cancer Detection System")
st.markdown("VGG16 Transfer Learning · BUSI Dataset · MySQL Database Integration")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Predict & Save",
    "📋 View History (READ)",
    "✏️ Update / Verify",
    "🗑️ Delete Records"
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — PREDICT & SAVE  (CREATE)
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    upload_col, result_col = st.columns([1, 1], gap="large")

    with upload_col:
        st.subheader("📤 Upload Image")
        uploaded = st.file_uploader(
            "Choose a PNG or JPG ultrasound image",
            type=["png", "jpg", "jpeg"]
        )
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded Image", use_column_width=True)
            st.markdown(f"**Resolution:** {img.size[0]} × {img.size[1]} px")

    with result_col:
        st.subheader("🔍 Prediction")
        if uploaded:
            if not os.path.exists(MODEL_PATH):
                st.error(f"Model file not found: `{MODEL_PATH}`")
            else:
                with st.spinner("Analysing image..."):
                    model = load_my_model()
                    pred_class, confidence, probs = predict(img, model)

                color = "🔴" if pred_class == "malignant" else "🟢"
                st.markdown(f"### {color} {pred_class.upper()}")
                st.metric(label="Confidence", value=f"{confidence:.1f}%")
                st.progress(int(confidence))

                st.divider()
                st.markdown("**Class Probabilities:**")
                for cls, prob in zip(CLASS_NAMES, probs):
                    bar_color = "🔴" if cls == "malignant" else "🟢"
                    st.markdown(f"{bar_color} **{cls.capitalize()}:** {prob*100:.2f}%")
                    st.progress(float(prob))

                st.divider()

                # ── AUTO-SAVE TO DATABASE (CREATE) ────────────────────────
                if DB_OK:
                    db_class = pred_class.capitalize()
                    new_id   = db_insert_prediction(
                        image_name       = uploaded.name,
                        predicted_class  = db_class,
                        confidence_score = confidence / 100
                    )
                    st.success(
                        f"✅ Prediction saved to database — Record ID: **{new_id}**"
                    )
                else:
                    st.warning("⚠️ DB not connected — prediction not saved.")

                if pred_class == "malignant" and confidence > 70:
                    st.warning(
                        "⚠️ High-confidence malignant result. "
                        "Please consult a qualified radiologist or oncologist."
                    )
        else:
            st.info("⬅️ Upload an ultrasound image to begin.")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — VIEW HISTORY  (READ)
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("📋 Prediction History")

    if not DB_OK:
        st.error("Database not connected.")
    else:
        f1, f2, f3 = st.columns(3)
        with f1:
            filter_class = st.selectbox("Filter by Class", ["All", "Benign", "Malignant"])
        with f2:
            min_conf = st.slider("Minimum Confidence (%)", 0, 100, 0)
        with f3:
            only_unverified = st.checkbox("Show Unverified Only")

        rows = db_get_all(filter_class, min_conf / 100, only_unverified)

        if rows:
            df = pd.DataFrame(rows)
            df["is_verified"] = df["is_verified"].apply(
                lambda x: "✅ Yes" if x else "⏳ Pending"
            )
            df.columns = [c.replace("_", " ").title() for c in df.columns]
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(f"Showing **{len(rows)}** record(s)")
        else:
            st.info("No records match the selected filters.")

        st.divider()
        st.subheader("📊 Summary Statistics")
        summary = db_summary()
        if summary:
            s_df = pd.DataFrame(summary)
            s_df.columns = ["Class", "Total", "Verified", "Pending Review", "Avg Confidence (%)"]
            st.dataframe(s_df, use_container_width=True, hide_index=True)

            col_b, col_m = st.columns(2)
            for row in summary:
                col = col_b if row["predicted_class"] == "Benign" else col_m
                with col:
                    emoji = "🟢" if row["predicted_class"] == "Benign" else "🔴"
                    st.metric(
                        label=f"{emoji} {row['predicted_class']} Predictions",
                        value=row["total"],
                        delta=f"Avg conf: {row['avg_conf_pct']}%"
                    )

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — UPDATE / VERIFY
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("✏️ Update / Verify a Prediction")

    if not DB_OK:
        st.error("Database not connected.")
    else:
        st.markdown("Enter a Record ID to correct a mislabelled prediction or mark it as verified.")

        pred_id_input = st.number_input(
            "Enter Record ID to Update", min_value=1, step=1, value=1
        )

        if st.button("🔍 Fetch Record"):
            record = db_get_by_id(pred_id_input)
            if record:
                st.session_state["edit_record"] = record
            else:
                st.error(f"No record found with ID {pred_id_input}")

        if "edit_record" in st.session_state:
            rec = st.session_state["edit_record"]
            st.divider()
            st.markdown(f"**Editing Record ID:** `{rec['id']}` — `{rec['image_name']}`")

            e1, e2 = st.columns(2)
            with e1:
                new_class = st.selectbox(
                    "Corrected Class",
                    ["Benign", "Malignant"],
                    index=0 if rec["predicted_class"] == "Benign" else 1
                )
            with e2:
                current_conf = round(float(rec["confidence_score"]) * 100, 2)
                new_conf = st.slider(
                    "Corrected Confidence (%)",
                    min_value=0.0, max_value=100.0,
                    value=current_conf, step=0.1
                )

            notes = st.text_area(
                "Review Notes (reason for correction)",
                value=rec.get("notes") or ""
            )

            if st.button("💾 Save Update"):
                affected = db_update_prediction(pred_id_input, new_class, new_conf, notes)
                if affected:
                    st.success(
                        f"✅ Record **{pred_id_input}** updated → "
                        f"**{new_class}** at **{new_conf:.1f}%** confidence. Marked as Verified."
                    )
                    del st.session_state["edit_record"]
                else:
                    st.error("Update failed — record may not exist.")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — DELETE
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("🗑️ Delete Prediction Records")

    if not DB_OK:
        st.error("Database not connected.")
    else:
        st.markdown("#### Delete a Single Record")
        del_id = st.number_input("Enter Record ID to Delete", min_value=1, step=1, value=1)

        if st.button("🗑️ Delete This Record"):
            rec = db_get_by_id(del_id)
            if rec:
                st.session_state["pending_delete"] = rec
            else:
                st.error(f"No record found with ID {del_id}")

        if "pending_delete" in st.session_state:
            rec = st.session_state["pending_delete"]
            st.warning(
                f"⚠️ You are about to delete:  \n"
                f"**ID {rec['id']}** | {rec['image_name']} | "
                f"{rec['predicted_class']} | "
                f"{round(float(rec['confidence_score'])*100, 1)}% confidence"
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Confirm Delete"):
                    db_delete_prediction(rec["id"])
                    st.success(f"Record **{rec['id']}** deleted successfully.")
                    del st.session_state["pending_delete"]
            with c2:
                if st.button("❌ Cancel"):
                    del st.session_state["pending_delete"]

        st.divider()
        st.markdown("#### Bulk Delete — Low Confidence Unverified Records")
        st.caption(
            "Removes all **unverified** predictions below a confidence threshold. "
            "Verified records are never deleted."
        )
        bulk_threshold = st.slider(
            "Delete unverified records below (%) confidence",
            min_value=10, max_value=90, value=60, step=5
        )

        if st.button("🧹 Run Bulk Delete"):
            deleted = db_delete_unverified_low_conf(bulk_threshold)
            if deleted:
                st.success(
                    f"✅ Deleted **{deleted}** unverified record(s) "
                    f"with confidence < {bulk_threshold}%"
                )
            else:
                st.info("No records matched the bulk-delete criteria.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Breast Cancer Detection System · VGG16 Transfer Learning · "
    "BUSI Dataset · MySQL CRUD Integration"
)