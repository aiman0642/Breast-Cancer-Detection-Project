import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
import time
import mysql.connector
from mysql.connector import Error
import pandas as pd
from datetime import datetime
import hashlib

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Breast Cancer Detection System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .role-badge-admin {
        background: linear-gradient(90deg,#e94560,#c0392b);
        color:white; padding:4px 12px; border-radius:20px;
        font-size:12px; font-weight:600;
    }
    .role-badge-doctor {
        background: linear-gradient(90deg,#2980b9,#1a5276);
        color:white; padding:4px 12px; border-radius:20px;
        font-size:12px; font-weight:600;
    }
    .role-badge-radiologist {
        background: linear-gradient(90deg,#27ae60,#1e8449);
        color:white; padding:4px 12px; border-radius:20px;
        font-size:12px; font-weight:600;
    }
    .role-badge-researcher {
        background: linear-gradient(90deg,#8e44ad,#6c3483);
        color:white; padding:4px 12px; border-radius:20px;
        font-size:12px; font-weight:600;
    }
    .welcome-card {
        background: linear-gradient(135deg,#1a1a2e,#16213e);
        border:1px solid #2d3748; border-radius:12px;
        padding:16px 20px; margin-bottom:20px;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
IMG_SIZE    = (224, 224)
CLASS_NAMES = ["benign", "malignant"]
MODEL_PATH  = "breast_cancer_vgg16.keras"

ROLE_ICONS = {
    "admin":       "👑",
    "doctor":      "👨‍⚕️",
    "radiologist": "🔬",
    "researcher":  "🧪"
}

# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
DB_CONFIG = {
    "host":     "localhost",
    "user":     "root",
    "password": "admin",
    "database": "breast_cancer_db"
}

def get_connection(silent=False):
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Error as e:
        if not silent:
            st.session_state["db_last_error"] = str(e)
        return None

def _column_exists(cur, table, column):
    cur.execute("""
        SELECT COUNT(*) FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND COLUMN_NAME = %s
    """, (DB_CONFIG["database"], table, column))
    return cur.fetchone()[0] > 0

def migrate_schema(cur):
    if not _column_exists(cur, "prediction_logs", "reviewed_by"):
        cur.execute("ALTER TABLE prediction_logs ADD COLUMN reviewed_by INT NULL")
    if not _column_exists(cur, "prediction_logs", "created_by"):
        cur.execute("ALTER TABLE prediction_logs ADD COLUMN created_by INT NULL")
    if not _column_exists(cur, "prediction_logs", "image_id"):
        cur.execute("ALTER TABLE prediction_logs ADD COLUMN image_id INT NULL")
    if not _column_exists(cur, "prediction_logs", "processing_time_ms"):
        cur.execute("ALTER TABLE prediction_logs ADD COLUMN processing_time_ms INT NULL")

# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE INIT
# ══════════════════════════════════════════════════════════════════════════════
def init_db():
    conn = get_connection(silent=True)
    if not conn:
        return False, "Could not connect to MySQL. Is the server running?"
    cur = conn.cursor()

    # ── users ─────────────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id       INT           NOT NULL AUTO_INCREMENT,
            username      VARCHAR(100)  NOT NULL UNIQUE,
            full_name     VARCHAR(255)  NOT NULL,
            email         VARCHAR(255)  NOT NULL UNIQUE,
            role          ENUM('admin','doctor','radiologist','researcher')
                          NOT NULL DEFAULT 'doctor',
            password_hash VARCHAR(64),
            created_at    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
            is_active     TINYINT(1)    NOT NULL DEFAULT 1,
            PRIMARY KEY (user_id)
        ) ENGINE=InnoDB;
    """)

    # ── images_metadata ───────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS images_metadata (
            image_id          INT          NOT NULL AUTO_INCREMENT,
            image_name        VARCHAR(255) NOT NULL,
            original_filename VARCHAR(255) NOT NULL,
            file_size_kb      FLOAT,
            image_width_px    INT,
            image_height_px   INT,
            file_format       ENUM('JPEG','PNG','BMP','TIFF') NOT NULL DEFAULT 'PNG',
            upload_path       VARCHAR(500),
            uploaded_by       INT,
            uploaded_at       DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
            notes             TEXT,
            PRIMARY KEY (image_id),
            CONSTRAINT fk_image_user
                FOREIGN KEY (uploaded_by) REFERENCES users(user_id)
                ON DELETE SET NULL
        ) ENGINE=InnoDB;
    """)

    # ── prediction_logs ───────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id                 INT          NOT NULL AUTO_INCREMENT,
            image_id           INT,
            image_name         VARCHAR(255) NOT NULL,
            predicted_class    ENUM('Benign','Malignant') NOT NULL,
            confidence_score   FLOAT        NOT NULL,
            model_version      VARCHAR(50)  NOT NULL DEFAULT 'VGG16-v1',
            processing_time_ms INT,
            is_verified        TINYINT(1)   NOT NULL DEFAULT 0,
            notes              TEXT,
            reviewed_by        INT          NULL,
            created_by         INT          NULL,
            timestamp          DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (id),
            CONSTRAINT fk_pred_image
                FOREIGN KEY (image_id) REFERENCES images_metadata(image_id)
                ON DELETE SET NULL,
            CONSTRAINT fk_pred_reviewer
                FOREIGN KEY (reviewed_by) REFERENCES users(user_id)
                ON DELETE SET NULL,
            CHECK (confidence_score BETWEEN 0.0 AND 1.0)
        ) ENGINE=InnoDB;
    """)

    migrate_schema(cur)

    # ── demo users ────────────────────────────────────────────────────────────
    demo_users = [
        ("admin",        "Admin User",          "admin@hospital.com",       "admin"),
        ("doctor1",      "Dr. Tayyaba Akhtar",  "tayyaba@hospital.com",     "doctor"),
        ("dr_aiman",     "Dr. Aiman Ijaz",      "aiman@hospital.com",       "radiologist"),
        ("researcher1",  "Researcher One",       "research@hospital.com",    "researcher"),
        ("dr_sarah",     "Dr. Sarah Ahmed",      "sarah@medcenter.pk",       "doctor"),
        ("dr_kamran",    "Dr. Kamran Malik",     "kamran@medcenter.pk",      "radiologist"),
        ("dr_nadia",     "Dr. Nadia Hussain",    "nadia@medcenter.pk",       "doctor"),
        ("admin_ali",    "Ali Hassan",           "ali@medcenter.pk",         "admin"),
    ]
    for uname, fname, email, role in demo_users:
        pwd_hash = hashlib.sha256(uname.encode()).hexdigest()
        cur.execute("""
            INSERT IGNORE INTO users
                (username, full_name, email, role, password_hash)
            VALUES (%s, %s, %s, %s, %s)
        """, (uname, fname, email, role, pwd_hash))
        cur.execute("""
            UPDATE users
            SET password_hash = %s
            WHERE username = %s
              AND (password_hash IS NULL OR password_hash = '')
        """, (pwd_hash, uname))

    conn.commit()
    cur.close()
    conn.close()
    return True, ""

def refresh_db_status():
    conn = get_connection(silent=True)
    if conn:
        try:
            cur = conn.cursor()
            migrate_schema(cur)
            conn.commit()
            cur.close()
        finally:
            conn.close()
        st.session_state["db_ok"]  = True
        st.session_state["db_err"] = ""
        return True
    err = st.session_state.get("db_last_error", "MySQL unreachable")
    st.session_state["db_ok"]  = False
    st.session_state["db_err"] = err
    return False

if "db_ok" not in st.session_state:
    _ok, _err = init_db()
    st.session_state["db_ok"]  = _ok
    st.session_state["db_err"] = _err

DB_OK  = st.session_state["db_ok"]
DB_ERR = st.session_state.get("db_err", "")

# ══════════════════════════════════════════════════════════════════════════════
#  AUTH FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def login_user(username, password):
    conn = get_connection(silent=True)
    if not conn:
        return None
    try:
        cur = conn.cursor(dictionary=True)
        pwd_hash = hashlib.sha256(password.encode()).hexdigest()
        cur.execute("""
            SELECT user_id, username, full_name, email, role
            FROM users
            WHERE username = %s
              AND password_hash = %s
              AND is_active = 1
        """, (username, pwd_hash))
        return cur.fetchone()
    except Error:
        return None
    finally:
        conn.close()

def get_all_users():
    conn = get_connection(silent=True)
    if not conn:
        return pd.DataFrame()
    try:
        return pd.read_sql("""
            SELECT user_id, username, full_name, email,
                   role, created_at,
                   CASE WHEN is_active=1 THEN 'Active' ELSE 'Inactive' END AS status
            FROM users ORDER BY created_at DESC
        """, conn)
    finally:
        conn.close()

def add_user(username, full_name, email, role, password):
    conn = get_connection(silent=True)
    if not conn:
        return False
    try:
        cur = conn.cursor()
        pwd_hash = hashlib.sha256(password.encode()).hexdigest()
        cur.execute("""
            INSERT INTO users (username, full_name, email, role, password_hash)
            VALUES (%s, %s, %s, %s, %s)
        """, (username, full_name, email, role, pwd_hash))
        conn.commit()
        return True
    except Error as e:
        st.error(f"Add user failed: {e}")
        return False
    finally:
        conn.close()

def deactivate_user(user_id):
    conn = get_connection(silent=True)
    if not conn:
        return False
    try:
        cur = conn.cursor()
        cur.execute("UPDATE users SET is_active=0 WHERE user_id=%s", (user_id,))
        conn.commit()
        return True
    finally:
        conn.close()

# ══════════════════════════════════════════════════════════════════════════════
#  CRUD FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── CREATE ────────────────────────────────────────────────────────────────────
def db_insert_prediction(image_name, predicted_class, confidence_score,
                         user_id=None, file_size_kb=None,
                         image_width_px=None, image_height_px=None,
                         processing_time_ms=None):
    conn = get_connection(silent=True)
    if not conn:
        return None
    cur = conn.cursor()

    # Step 1 — insert into images_metadata first
    cur.execute("""
        INSERT INTO images_metadata
            (image_name, original_filename, file_size_kb,
             image_width_px, image_height_px, file_format, uploaded_by)
        VALUES (%s, %s, %s, %s, %s, 'PNG', %s)
    """, (image_name, image_name, file_size_kb,
          image_width_px, image_height_px, user_id))
    image_id = cur.lastrowid

    # Step 2 — insert into prediction_logs with the new image_id
    cur.execute("""
        INSERT INTO prediction_logs
            (image_id, image_name, predicted_class, confidence_score,
             processing_time_ms, created_by)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (image_id, image_name, predicted_class,
          round(confidence_score, 4), processing_time_ms, user_id))

    conn.commit()
    new_id = cur.lastrowid
    cur.close()
    conn.close()
    return new_id

# ── READ ──────────────────────────────────────────────────────────────────────
def db_get_all(filter_class=None, min_conf=0.0, only_unverified=False):
    conn = get_connection(silent=True)
    if not conn:
        return None
    cur = conn.cursor(dictionary=True)
    where_plain = ["confidence_score >= %s"]
    params = [min_conf]
    if filter_class and filter_class != "All":
        where_plain.append("predicted_class = %s")
        params.append(filter_class)
    if only_unverified:
        where_plain.append("is_verified = 0")
    where_prefixed = " AND ".join(
        c.replace("confidence_score", "p.confidence_score")
         .replace("predicted_class", "p.predicted_class")
         .replace("is_verified", "p.is_verified")
        for c in where_plain
    )
    try:
        cur.execute(f"""
            SELECT p.id, p.image_name, p.predicted_class,
                   ROUND(p.confidence_score*100,2) AS confidence_pct,
                   p.model_version, p.is_verified, p.notes,
                   COALESCE(u_save.full_name, u_rev.full_name) AS saved_by,
                   p.timestamp
            FROM prediction_logs p
            LEFT JOIN users u_save ON p.created_by  = u_save.user_id
            LEFT JOIN users u_rev  ON p.reviewed_by = u_rev.user_id
            WHERE {where_prefixed}
            ORDER BY p.timestamp DESC
        """, params)
    except Error:
        where_sql = " AND ".join(where_plain)
        cur.execute(f"""
            SELECT id, image_name, predicted_class,
                   ROUND(confidence_score*100,2) AS confidence_pct,
                   model_version, is_verified, notes,
                   NULL AS saved_by, timestamp
            FROM prediction_logs
            WHERE {where_sql}
            ORDER BY timestamp DESC
        """, params)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def db_get_by_id(pred_id):
    conn = get_connection(silent=True)
    if not conn:
        return None
    cur  = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM prediction_logs WHERE id=%s", (pred_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row

def db_summary():
    conn = get_connection(silent=True)
    if not conn:
        return []
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT
            predicted_class,
            COUNT(*)                            AS total,
            SUM(is_verified)                    AS verified,
            COUNT(*) - SUM(is_verified)         AS pending,
            ROUND(AVG(confidence_score)*100,2)  AS avg_conf_pct
        FROM prediction_logs
        GROUP BY predicted_class
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

# ── UPDATE ────────────────────────────────────────────────────────────────────
def db_update_prediction(pred_id, new_class, new_conf, notes, reviewer_id=None):
    conn = get_connection(silent=True)
    if not conn:
        return 0
    cur  = conn.cursor()
    cur.execute("""
        UPDATE prediction_logs
        SET predicted_class  = %s,
            confidence_score = %s,
            notes            = %s,
            is_verified      = 1,
            reviewed_by      = COALESCE(%s, reviewed_by)
        WHERE id = %s
    """, (new_class, round(new_conf / 100, 4), notes, reviewer_id, pred_id))
    conn.commit()
    affected = cur.rowcount
    cur.close()
    conn.close()
    return affected

# ── DELETE ────────────────────────────────────────────────────────────────────
def db_delete_prediction(pred_id):
    conn = get_connection(silent=True)
    if not conn:
        return 0
    cur  = conn.cursor()
    cur.execute("DELETE FROM prediction_logs WHERE id=%s", (pred_id,))
    conn.commit()
    affected = cur.rowcount
    cur.close()
    conn.close()
    return affected

def db_delete_unverified_low_conf(threshold):
    conn = get_connection(silent=True)
    if not conn:
        return 0
    cur  = conn.cursor()
    cur.execute("""
        DELETE FROM prediction_logs
        WHERE is_verified=0 AND confidence_score < %s
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

def is_mask_image(img):
    arr = np.array(img.convert("L"))
    if len(np.unique(arr)) <= 5:
        return True
    if np.sum(arr < 10) / arr.size > 0.95:
        return True
    return False

def predict(img_array, model):
    start = time.time()
    arr   = np.array(img_array.resize(IMG_SIZE).convert("RGB"), dtype=np.float32)
    arr   = preprocess_input(arr)
    arr   = np.expand_dims(arr, axis=0)
    probs = model.predict(arr, verbose=0)[0]
    elapsed_ms = int((time.time() - start) * 1000)
    pred_idx   = int(np.argmax(probs))
    return CLASS_NAMES[pred_idx], float(probs[pred_idx]) * 100, probs, elapsed_ms

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user      = None

# ══════════════════════════════════════════════════════════════════════════════
#  LOGIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
def show_login():
    col1, col2, col3 = st.columns([1, 1.1, 1])
    with col2:
        st.markdown("""
        <div style='text-align:center; margin-top:40px; margin-bottom:30px;'>
            <div style='font-size:64px;'>🩺</div>
            <h1 style='color:#ffffff; font-size:30px; font-weight:700; margin:10px 0 4px;'>
                Breast Cancer Detection
            </h1>
            <p style='color:#a0aec0; font-size:15px; margin:0;'>
                AI-Powered Diagnostic System
            </p>
            <p style='color:#718096; font-size:12px; margin-top:6px;'>
                VGG16 Transfer Learning &nbsp;·&nbsp; BUSI Dataset &nbsp;·&nbsp; MySQL
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("#### 🔐 Sign In to Your Account")
            st.markdown("")
            username = st.text_input("👤  Username", placeholder="Enter your username")
            password = st.text_input("🔑  Password", type="password",
                                      placeholder="Enter your password")
            st.markdown("")
            if st.button("Sign In →", type="primary", use_container_width=True):
                if not username or not password:
                    st.error("Please enter both username and password.")
                else:
                    with st.spinner("Authenticating..."):
                        user = login_user(username, password)
                    if user:
                        st.session_state.logged_in = True
                        st.session_state.user      = user
                        st.success(f"Welcome, {user['full_name']}!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password.")

        st.markdown("")
        with st.expander("👥 Demo Credentials (click to view)"):
            st.markdown("""
            | Role | Username | Password |
            |------|----------|----------|
            | 👑 Admin | `admin` | `admin` |
            | 👨‍⚕️ Doctor | `doctor1` | `doctor1` |
            | 🔬 Radiologist | `dr_aiman` | `dr_aiman` |
            | 🧪 Researcher | `researcher1` | `researcher1` |
            """)
            st.caption("Password is the same as username for all demo accounts.")

        st.markdown("""
        <p style='text-align:center; color:#4a5568; font-size:11px; margin-top:16px;'>
            ⚠️ For educational and research use only.<br>
            Not a substitute for professional medical diagnosis.
        </p>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def show_sidebar():
    user = st.session_state.user
    role = user["role"]
    icon = ROLE_ICONS.get(role, "👤")

    with st.sidebar:
        st.markdown(f"""
        <div class='welcome-card'>
            <div style='font-size:40px; text-align:center;'>{icon}</div>
            <div style='text-align:center; color:white; font-weight:600;
                        font-size:16px; margin:8px 0 4px;'>{user['full_name']}</div>
            <div style='text-align:center; color:#a0aec0;
                        font-size:12px; margin-bottom:10px;'>@{user['username']}</div>
            <div style='text-align:center;'>
                <span class='role-badge-{role}'>{icon} {role.capitalize()}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        if refresh_db_status():
            st.success("🟢 MySQL Connected")
        else:
            err = st.session_state.get("db_err") or st.session_state.get("db_last_error", "Unknown")
            st.error(f"🔴 MySQL Error\n{err}")

        st.markdown("**Model:** VGG16 Transfer Learning")
        st.markdown("**Dataset:** BUSI (780 images)")
        st.markdown("**Classes:** Benign | Malignant")
        st.divider()

        st.markdown("#### 🗂️ Navigation")
        pages = ["🔬 Predict & Save", "📋 View History", "📊 Dashboard"]

        if role in ["doctor", "radiologist", "admin"]:
            pages.append("✏️ Update / Verify")

        if role == "admin":
            pages.append("🗑️ Delete Records")
            pages.append("👥 Manage Users")

        page = st.radio("", pages, label_visibility="collapsed")
        st.divider()

        if st.button("🚪 Sign Out", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user      = None
            st.rerun()

        st.markdown(f"""
        <p style='text-align:center; color:#4a5568; font-size:10px; margin-top:16px;'>
            Logged in as <b>{role}</b><br>
            {datetime.now().strftime('%Y-%m-%d %H:%M')}
        </p>
        """, unsafe_allow_html=True)

    return page

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE — PREDICT & SAVE  (CREATE)
# ══════════════════════════════════════════════════════════════════════════════
def page_predict():
    user = st.session_state.user
    st.title("🔬 Predict & Save")
    st.markdown("Upload a breast ultrasound image to classify and save to database.")
    st.divider()

    upload_col, result_col = st.columns([1, 1], gap="large")

    with upload_col:
        st.subheader("📤 Upload Image")
        uploaded = st.file_uploader(
            "Choose a PNG or JPG ultrasound image",
            type=["png", "jpg", "jpeg"]
        )
        if uploaded:
            img = Image.open(uploaded)

            if is_mask_image(img):
                st.error("⛔ This looks like a mask image, not a real ultrasound!")
                st.warning("Please upload the original image, not the _mask version.")
                return

            st.image(img, caption="Uploaded Image", use_container_width=True)
            st.markdown(f"**File:** {uploaded.name}")
            st.markdown(f"**Size:** {round(uploaded.size/1024, 2)} KB")
            st.markdown(f"**Resolution:** {img.size[0]} × {img.size[1]} px")

    with result_col:
        st.subheader("🔍 Prediction")
        if uploaded:
            if not os.path.exists(MODEL_PATH):
                st.error(f"Model file not found: `{MODEL_PATH}`")
                return

            with st.spinner("Analysing image..."):
                model = load_my_model()
                pred_class, confidence, probs, elapsed_ms = predict(img, model)

            color = "🔴" if pred_class == "malignant" else "🟢"
            st.markdown(f"### {color} {pred_class.upper()}")
            st.metric("Confidence", f"{confidence:.1f}%")
            st.progress(int(confidence))
            st.caption(f"⏱️ Processing time: {elapsed_ms} ms")

            st.divider()
            st.markdown("**Class Probabilities:**")
            for cls, prob in zip(CLASS_NAMES, probs):
                bc = "🔴" if cls == "malignant" else "🟢"
                st.markdown(f"{bc} **{cls.capitalize()}:** {prob*100:.2f}%")
                st.progress(float(prob))

            st.divider()

            # Auto-save — now populates BOTH images_metadata AND prediction_logs
            if refresh_db_status():
                db_class = pred_class.capitalize()
                new_id   = db_insert_prediction(
                    image_name         = uploaded.name,
                    predicted_class    = db_class,
                    confidence_score   = confidence / 100,
                    user_id            = user["user_id"],
                    file_size_kb       = round(uploaded.size / 1024, 2),
                    image_width_px     = img.size[0],
                    image_height_px    = img.size[1],
                    processing_time_ms = elapsed_ms
                )
                st.success(f"✅ Prediction saved to database — Record ID: **{new_id}**")
                st.caption(f"Saved by: {user['full_name']} ({user['role']})")
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
#  PAGE — VIEW HISTORY  (READ)
# ══════════════════════════════════════════════════════════════════════════════
def page_records():
    st.title("📋 Prediction History")
    st.divider()

    if not refresh_db_status():
        err = st.session_state.get("db_err") or st.session_state.get("db_last_error", "")
        st.error(f"Database not connected. {err}")
        return

    if st.button("🔄 Refresh"):
        refresh_db_status()
        st.rerun()

    f1, f2, f3 = st.columns(3)
    with f1:
        filter_class = st.selectbox("Filter by Class", ["All", "Benign", "Malignant"])
    with f2:
        min_conf = st.slider("Minimum Confidence (%)", 0, 100, 0)
    with f3:
        only_unverified = st.checkbox("Show Unverified Only")

    rows = db_get_all(filter_class, min_conf / 100, only_unverified)

    if rows is None:
        err = st.session_state.get("db_last_error", "Could not reach MySQL.")
        st.error(f"Query failed — database unreachable. {err}")
        return

    if rows:
        df = pd.DataFrame(rows)
        df["is_verified"] = df["is_verified"].apply(
            lambda x: "✅ Verified" if x else "⏳ Pending"
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total",        len(df))
        m2.metric("🟢 Benign",    len(df[df["predicted_class"] == "Benign"]))
        m3.metric("🔴 Malignant", len(df[df["predicted_class"] == "Malignant"]))
        m4.metric("✅ Verified",  len(df[df["is_verified"] == "✅ Verified"]))

        st.divider()
        st.dataframe(df, use_container_width=True, hide_index=True, height=400)
        st.caption(f"Showing **{len(rows)}** record(s)")

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "predictions.csv", "text/csv")

        st.divider()
        st.subheader("📊 Summary Statistics")
        summary = db_summary()
        if summary:
            s_df = pd.DataFrame(summary)
            s_df.columns = ["Class", "Total", "Verified", "Pending", "Avg Confidence (%)"]
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
    else:
        st.info("No records match the selected filters (or prediction_logs is empty).")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    st.title("📊 Dashboard")
    st.divider()

    if not refresh_db_status():
        st.error("Database not connected.")
        return

    summary = db_summary()
    if not summary:
        st.info("No predictions yet. Go to 🔬 Predict & Save to add some.")
        return

    total_b = next((r["total"] for r in summary if r["predicted_class"] == "Benign"), 0)
    total_m = next((r["total"] for r in summary if r["predicted_class"] == "Malignant"), 0)
    total   = total_b + total_m
    avg_b   = next((r["avg_conf_pct"] for r in summary if r["predicted_class"] == "Benign"), 0)
    avg_m   = next((r["avg_conf_pct"] for r in summary if r["predicted_class"] == "Malignant"), 0)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Predictions", total)
    c2.metric("🟢 Benign",         total_b)
    c3.metric("🔴 Malignant",      total_m)
    c4.metric("Avg Conf Benign",   f"{avg_b}%")
    c5.metric("Avg Conf Malignant",f"{avg_m}%")

    st.divider()

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#0e1117')
    for ax in axes:
        ax.set_facecolor('#0e1117')

    axes[0].pie(
        [total_b, total_m],
        labels=["Benign", "Malignant"],
        colors=["#27ae60", "#e74c3c"],
        autopct="%1.1f%%", startangle=140,
        textprops={"color": "white"}
    )
    axes[0].set_title("Class Distribution", color="white", fontweight="bold")

    rows = db_get_all()
    if rows:
        df = pd.DataFrame(rows).head(10)
        colors = ["#e74c3c" if c == "Malignant" else "#27ae60"
                  for c in df["predicted_class"]]
        axes[1].barh(df["image_name"], df["confidence_pct"], color=colors)
        axes[1].set_xlabel("Confidence (%)", color="white")
        axes[1].set_title("Last 10 Predictions", color="white", fontweight="bold")
        axes[1].tick_params(colors="white")
        axes[1].set_xlim(0, 100)
        for spine in axes[1].spines.values():
            spine.set_edgecolor('#2d3748')

    plt.tight_layout()
    st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE — UPDATE / VERIFY
# ══════════════════════════════════════════════════════════════════════════════
def page_update():
    st.title("✏️ Update / Verify a Prediction")
    st.markdown("Correct a mislabelled prediction or mark it as clinically verified.")
    st.divider()

    if not refresh_db_status():
        st.error("Database not connected.")
        return

    user = st.session_state.user
    pred_id_input = st.number_input("Enter Record ID to Update", min_value=1, step=1, value=1)

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
            "Clinical Notes (reason for correction)",
            value=rec.get("notes") or ""
        )

        if st.button("💾 Save Update", type="primary"):
            affected = db_update_prediction(
                pred_id_input, new_class, new_conf, notes,
                reviewer_id=user["user_id"]
            )
            if affected:
                st.success(
                    f"✅ Record **{pred_id_input}** updated → "
                    f"**{new_class}** at **{new_conf:.1f}%** confidence. Marked as Verified."
                )
                del st.session_state["edit_record"]
            else:
                st.error("Update failed.")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE — DELETE RECORDS
# ══════════════════════════════════════════════════════════════════════════════
def page_delete():
    st.title("🗑️ Delete Prediction Records")
    st.divider()

    if not refresh_db_status():
        st.error("Database not connected.")
        return

    st.markdown("#### Delete a Single Record")
    del_id = st.number_input("Enter Record ID to Delete", min_value=1, step=1, value=1)

    if st.button("🔍 Preview Record"):
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
            if st.button("✅ Confirm Delete", type="primary"):
                db_delete_prediction(rec["id"])
                st.success(f"Record **{rec['id']}** deleted successfully.")
                del st.session_state["pending_delete"]
        with c2:
            if st.button("❌ Cancel"):
                del st.session_state["pending_delete"]

    st.divider()
    st.markdown("#### Bulk Delete — Low Confidence Unverified Records")
    st.caption(
        "Removes all **unverified** predictions below the selected confidence. "
        "Verified records are never touched."
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

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE — MANAGE USERS
# ══════════════════════════════════════════════════════════════════════════════
def page_manage_users():
    st.title("👥 Manage Users")
    st.divider()

    if not refresh_db_status():
        st.error("Database not connected.")
        return

    st.subheader("All Users")
    df = get_all_users()
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No users found.")

    st.divider()
    st.subheader("➕ Add New User")

    with st.form("add_user_form"):
        col1, col2 = st.columns(2)
        with col1:
            new_username  = st.text_input("Username")
            new_fullname  = st.text_input("Full Name")
            new_email     = st.text_input("Email")
        with col2:
            new_role      = st.selectbox("Role", ["doctor", "radiologist", "researcher", "admin"])
            new_password  = st.text_input("Password", type="password")
            new_password2 = st.text_input("Confirm Password", type="password")

        submitted = st.form_submit_button("➕ Add User", type="primary")
        if submitted:
            if not all([new_username, new_fullname, new_email, new_password]):
                st.error("All fields are required.")
            elif new_password != new_password2:
                st.error("Passwords do not match.")
            else:
                if add_user(new_username, new_fullname, new_email, new_role, new_password):
                    st.success(f"✅ User '{new_username}' added as {new_role}.")
                    st.rerun()

    st.divider()
    st.subheader("🚫 Deactivate User")
    deact_id = st.number_input("Enter User ID to deactivate", min_value=1, step=1)
    if st.button("Deactivate User"):
        if deactivate_user(deact_id):
            st.success(f"User ID {deact_id} deactivated.")
        else:
            st.error("Failed to deactivate user.")

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ROUTER
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    show_login()
else:
    page = show_sidebar()

    st.title("🩺 Breast Cancer Detection System")
    st.markdown(
        "VGG16 Transfer Learning · BUSI Dataset · MySQL Database Integration"
    )
    st.divider()

    if page == "🔬 Predict & Save":
        page_predict()
    elif page == "📋 View History":
        page_records()
    elif page == "📊 Dashboard":
        page_dashboard()
    elif page == "✏️ Update / Verify":
        page_update()
    elif page == "🗑️ Delete Records":
        page_delete()
    elif page == "👥 Manage Users":
        page_manage_users()

    st.divider()
    st.caption(
        "Breast Cancer Detection System · VGG16 Transfer Learning · "
        "BUSI Dataset · MySQL CRUD Integration"
    )