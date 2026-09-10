import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import random
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# ============================================================
# --- Konfigurasi Halaman ---
# ============================================================
st.set_page_config(
    page_title="Sleep Quality & Stress Level Predictor",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# --- Helper: generate posisi bintang secara acak (deterministik) ---
# ============================================================
def generate_star_shadows(n, seed, color="#FFF7DE", x_range=(0, 100), y_range=(0, 100), unit=("%", "%")):
    rng = random.Random(seed)
    shadows = []
    for _ in range(n):
        x = rng.randint(*x_range)
        y = rng.randint(*y_range)
        shadows.append(f"{x}{unit[0]} {y}{unit[1]} {color}")
    return ", ".join(shadows)

# Bintang di dalam hero banner (area gelap di atas)
HERO_STARS_1 = generate_star_shadows(35, seed=42, color="#FFFFFF")
HERO_STARS_2 = generate_star_shadows(18, seed=7, color="#FFE9A8")


# ============================================================
# --- CSS Tema: area utama terang & rapi, sidebar + hero gelap bertema malam ---
# ============================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@600;700;800&family=Poppins:wght@300;400;500;600;700&display=swap');

/* ============ Latar utama: terang & bersih ============ */
.stApp {{
    background: linear-gradient(160deg, #F6F5FF 0%, #FFFFFF 40%, #F1F5FF 100%);
    font-family: 'Poppins', sans-serif;
}}
header[data-testid="stHeader"] {{
    background: transparent;
}}
.block-container {{
    padding-top: 1.6rem;
    padding-bottom: 3rem;
    max-width: 1180px;
}}
.stApp h1, .stApp h2, .stApp h3 {{
    font-family: 'Baloo 2', sans-serif;
    color: #232946;
}}

/* ============ Blob gradasi lembut yang mengambang (dekorasi, tidak ganggu bacaan) ============ */
.bg-blobs {{
    position: fixed;
    inset: 0;
    z-index: -3;
    overflow: hidden;
    pointer-events: none;
}}
.blob {{
    position: absolute;
    border-radius: 50%;
    filter: blur(70px);
    opacity: 0.35;
}}
.blob-1 {{
    width: 420px; height: 420px;
    background: #C9B6FF;
    top: -120px; left: -100px;
    animation: driftA 16s ease-in-out infinite;
}}
.blob-2 {{
    width: 380px; height: 380px;
    background: #FFC3DE;
    bottom: -140px; right: -80px;
    animation: driftB 19s ease-in-out infinite;
}}
.blob-3 {{
    width: 300px; height: 300px;
    background: #B7D4FF;
    top: 40%; right: 12%;
    animation: driftC 14s ease-in-out infinite;
}}
@keyframes driftA {{
    0%, 100% {{ transform: translate(0, 0) scale(1); }}
    50%      {{ transform: translate(60px, 40px) scale(1.12); }}
}}
@keyframes driftB {{
    0%, 100% {{ transform: translate(0, 0) scale(1); }}
    50%      {{ transform: translate(-50px, -30px) scale(1.08); }}
}}
@keyframes driftC {{
    0%, 100% {{ transform: translate(0, 0) scale(1); }}
    50%      {{ transform: translate(-30px, 50px) scale(0.94); }}
}}

@keyframes twinkle {{
    0%, 100% {{ opacity: 0.25; }}
    50%      {{ opacity: 1; }}
}}
@keyframes floatMoon {{
    0%, 100% {{ transform: translateY(0px); }}
    50%      {{ transform: translateY(-16px); }}
}}
@keyframes glowPulse {{
    0%, 100% {{ box-shadow: 0 0 24px 8px rgba(255, 233, 168, 0.35), 0 0 55px 20px rgba(255, 233, 168, 0.12); }}
    50%      {{ box-shadow: 0 0 36px 14px rgba(255, 233, 168, 0.55), 0 0 80px 28px rgba(255, 233, 168, 0.22); }}
}}

/* ============ Hero banner (kartu gelap bertema malam, kontras dijaga di sini saja) ============ */
.hero-box {{
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
    padding: 34px 40px;
    margin-bottom: 26px;
    border-radius: 22px;
    background: linear-gradient(135deg, #0b1330 0%, #16224a 55%, #1f2e5c 100%);
    box-shadow: 0 16px 40px rgba(20, 25, 60, 0.25);
}}
.hero-stars-1, .hero-stars-2 {{
    position: absolute; top: 0; left: 0;
    width: 2px; height: 2px;
    background: transparent;
    border-radius: 50%;
    pointer-events: none;
}}
.hero-stars-1 {{ box-shadow: {HERO_STARS_1}; animation: twinkle 3.2s ease-in-out infinite; }}
.hero-stars-2 {{ box-shadow: {HERO_STARS_2}; width: 3px; height: 3px; animation: twinkle 4.6s ease-in-out infinite 0.6s; }}
.hero-text {{ position: relative; z-index: 2; flex: 1; min-width: 260px; }}
.hero-title {{
    font-family: 'Baloo 2', sans-serif;
    font-weight: 800;
    font-size: 2.3rem;
    line-height: 1.08;
    margin: 0;
    color: #FFFFFF !important;
    letter-spacing: 0.4px;
}}
.hero-title .accent {{ color: #F6B8CB !important; }}
.hero-sub {{
    margin-top: 10px;
    font-size: 0.98rem;
    color: #D9DCF5 !important;
    max-width: 560px;
}}
.hero-badge {{
    display: inline-block;
    margin-top: 16px;
    padding: 6px 16px;
    border-radius: 999px;
    background: rgba(246, 184, 203, 0.18);
    border: 1px solid rgba(246, 184, 203, 0.45);
    color: #F6B8CB !important;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.4px;
}}
.hero-moon-area {{
    position: relative;
    z-index: 2;
    width: 140px;
    height: 140px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
}}
.hero-moon {{
    width: 92px; height: 92px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 32%, #FFFBEA 0%, #FFE9A8 55%, #FFD98A 100%);
    animation: floatMoon 6s ease-in-out infinite, glowPulse 4s ease-in-out infinite;
}}

/* ============ Sidebar: disamakan dengan tema terang area utama ============ */
[data-testid="stSidebar"] {{
    position: relative;
    overflow: hidden;
    background: linear-gradient(180deg, #FBFAFF 0%, #FFFFFF 55%, #F4F2FF 100%);
    border-right: 1px solid #ECEBFB;
}}
[data-testid="stSidebar"] > div {{ position: relative; z-index: 1; }}

/* Blob dekorasi lembut, kontinu dgn animasi area utama tapi lebih kalem */
.sidebar-blob-1, .sidebar-blob-2 {{
    position: absolute;
    border-radius: 50%;
    filter: blur(55px);
    opacity: 0.4;
    z-index: 0;
    pointer-events: none;
}}
.sidebar-blob-1 {{ width: 230px; height: 230px; background: #C9B6FF; top: -70px; right: -90px; animation: driftA 15s ease-in-out infinite; }}
.sidebar-blob-2 {{ width: 210px; height: 210px; background: #FFC3DE; bottom: 8%; left: -90px; animation: driftB 18s ease-in-out infinite; }}

/* Teks sidebar: gelap & jelas di atas latar terang */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] .stMarkdown {{
    color: #3A3F63 !important;
}}

/* Kotak selectbox bergaya kartu modern, senada dengan area utama */
[data-testid="stSidebar"] [data-baseweb="select"] {{
    background-color: #FFFFFF !important;
    border: 1px solid #E4E1FA !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 14px rgba(90, 90, 160, 0.08);
}}
[data-testid="stSidebar"] [data-baseweb="select"] * {{
    color: #232946 !important;
    -webkit-text-fill-color: #232946 !important;
}}
[data-testid="stSidebar"] [data-baseweb="select"] svg {{
    fill: #8B7CF6 !important;
}}

/* Slider: thumb & fokus senada dengan aksen ungu-pink tema */
[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {{
    background-color: #8B7CF6 !important;
    border-color: #8B7CF6 !important;
    box-shadow: 0 0 0 4px rgba(139, 124, 246, 0.18) !important;
}}

.sidebar-hero {{
    position: relative; z-index: 1;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 14px 6px 18px 6px;
    border-bottom: 1px solid #ECEBFB;
    margin-bottom: 14px;
}}
.sidebar-hero .moon-mini {{
    width: 30px; height: 30px;
    flex-shrink: 0;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 32%, #FFFBEA 0%, #FFE9A8 55%, #FFD98A 100%);
    box-shadow: 0 0 16px 4px rgba(255, 217, 138, 0.55);
    animation: floatMoon 5s ease-in-out infinite, glowPulse 4s ease-in-out infinite;
}}
.sidebar-hero h2 {{ font-family: 'Baloo 2', sans-serif; font-size: 1.15rem; margin: 0; color: #232946 !important; }}
.sidebar-hero p {{ font-size: 0.8rem; color: #8A8FB5 !important; margin-top: 3px; }}
.sidebar-section {{
    position: relative; z-index: 1;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'Baloo 2', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    color: #6B4FD8 !important;
    background: rgba(139, 124, 246, 0.10);
    padding: 5px 12px;
    border-radius: 999px;
    margin: 18px 0 8px 0;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}}

/* ============ Kartu putih untuk konten utama ============ */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background: #FFFFFF !important;
    border: 1px solid #ECEBFB !important;
    border-radius: 18px !important;
    box-shadow: 0 6px 22px rgba(90, 90, 160, 0.08);
}}

/* ============ Tombol ============ */
/* Tombol Utama (Primary) - Disamakan dengan tema gelap Hero Banner */
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, #0b1330 0%, #1f2e5c 100%) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    border: 1px solid #2a3a70 !important;
    border-radius: 14px !important;
    padding: 0.7rem 1.4rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 8px 20px rgba(11, 19, 48, 0.25) !important;
}}

.stButton > button[kind="primary"]:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 26px rgba(11, 19, 48, 0.4) !important;
    background: linear-gradient(135deg, #16224a 0%, #2a3a70 100%) !important;
    color: #F6B8CB !important; /* Aksen teks pink ala badge banner saat di-hover */
    border-color: #F6B8CB !important;
}}

/* Tombol Sekunder (Secondary) - Untuk tombol "Kosongkan Riwayat", dll */
.stButton > button[kind="secondary"] {{
    background: #FFFFFF !important;
    color: #5B3FBF !important;
    font-weight: 700 !important;
    border: 1px solid #ECEBFB !important;
    border-radius: 14px !important;
    padding: 0.7rem 1.4rem !important;
    box-shadow: 0 4px 14px rgba(90, 90, 160, 0.08) !important;
    transition: all 0.2s ease !important;
}}

.stButton > button[kind="secondary"]:hover {{
    transform: translateY(-2px) !important;
    border-color: #8B7CF6 !important;
    box-shadow: 0 8px 20px rgba(139, 124, 246, 0.15) !important;
    color: #8B7CF6 !important;
}}

/* ============ Metric ============ */
[data-testid="stMetric"] {{
    background: #FFFFFF;
    border: 1px solid #ECEBFB;
    border-radius: 16px;
    padding: 14px 16px;
    box-shadow: 0 6px 18px rgba(90, 90, 160, 0.07);
    transition: transform 0.15s ease;
}}
[data-testid="stMetric"]:hover {{ transform: translateY(-3px); }}

/* ============ Tabs ============ */
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
.stTabs [data-baseweb="tab"] {{
    background: #F2F1FC;
    border-radius: 12px 12px 0 0;
    padding: 10px 18px;
    color: #6B7099 !important;
    font-weight: 600;
}}
.stTabs [aria-selected="true"] {{
    background: #EADCFF !important;
    color: #5B3FBF !important;
}}

/* ============ Dataframe ============ */
[data-testid="stDataFrame"] {{ border-radius: 14px; overflow: hidden; }}

/* ============ Progress bar ============ */
.stProgress > div > div {{ background: linear-gradient(90deg, #8B7CF6, #EC7FB0, #FFC98A); }}

/* ============ Grid ringkasan data (kartu kecil, anti-berantakan) ============ */
.chip-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
}}
.chip {{
    background: #FFFFFF;
    border: 1px solid #ECEBFB;
    border-radius: 14px;
    padding: 12px 14px;
    box-shadow: 0 4px 14px rgba(90, 90, 160, 0.06);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
.chip:hover {{ transform: translateY(-3px); box-shadow: 0 10px 22px rgba(90, 90, 160, 0.14); }}
.chip-icon {{ font-size: 1.25rem; }}
.chip-label {{ font-size: 0.74rem; color: #8A8FB5; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.4px; }}
.chip-value {{ font-size: 1.02rem; color: #232946; font-weight: 700; margin-top: 2px; }}

/* ============ Kartu status hasil prediksi ============ */
.status-card {{
    background: #FFFFFF;
    border-radius: 16px;
    padding: 20px 22px;
    border-left: 7px solid #8B7CF6;
    box-shadow: 0 8px 22px rgba(90, 90, 160, 0.10);
}}
.status-card.ok    {{ border-left-color: #34C77B; }}
.status-card.warn  {{ border-left-color: #F2A93B; }}
.status-card.risk  {{ border-left-color: #F05C6B; }}
.status-title {{ font-family: 'Baloo 2', sans-serif; font-size: 1.3rem; margin: 0 0 6px 0; color: #232946; }}
.status-desc {{ color: #5B6081; font-size: 0.93rem; margin: 0; }}

/* ============ Divider bintang ============ */
.star-divider {{
    text-align: center;
    color: #C7B4FF;
    letter-spacing: 12px;
    margin: 26px 0 10px 0;
    font-size: 0.9rem;
}}

/* ============ Footer ============ */
.footer-note {{
    text-align: center;
    margin-top: 34px;
    padding: 14px;
    color: #9AA0C2;
    font-size: 0.8rem;
}}
</style>

<div class="bg-blobs">
    <div class="blob blob-1"></div>
    <div class="blob blob-2"></div>
    <div class="blob blob-3"></div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# --- Konfigurasi & Inisialisasi Database SQLite ---
# ============================================================
def init_db():
    # Membuat atau menghubungkan ke file database lokal 'sleep_health.db'
    conn = sqlite3.connect('sleep_health.db', check_same_thread=False)
    cursor = conn.cursor()
    # Membuat tabel riwayat prediksi jika belum ada
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            gender TEXT,
            age INTEGER,
            occupation TEXT,
            sleep_duration REAL,
            quality_of_sleep INTEGER,
            physical_activity INTEGER,
            stress_level INTEGER,
            bmi_category TEXT,
            heart_rate INTEGER,
            daily_steps INTEGER,
            systolic_bp INTEGER,
            diastolic_bp INTEGER,
            prediction TEXT
        )
    ''')
    conn.commit()
    return conn

conn = init_db()

# ============================================================
# --- Memuat Model & Bundle yang Telah Disimpan ---
# ============================================================
@st.cache_resource
def load_bundle():
    bundle = joblib.load('sleep_disorder_model.pkl')
    return bundle['model'], bundle['scaler'], bundle['encoders'], bundle['feature_cols']

model, scaler, encoders, feature_cols = load_bundle()

# Memuat dataset bersih untuk sistem rekomendasi berbasis kemiripan
@st.cache_data
def load_data():
    return pd.read_csv('sleep_health_clean.csv')

df_clean = load_data()

# ============================================================
# --- Hero Header ---
# ============================================================
st.markdown("""
<div class="hero-box">
    <div class="hero-stars-1"></div>
    <div class="hero-stars-2"></div>
    <div class="hero-text">
        <div class="hero-title">SLEEP QUALITY <span class="accent">PREDICTOR</span></div>
        <div class="hero-sub">
            Aplikasi interaktif berbasis Machine Learning yang terhubung dengan database <b>SQLite</b>
            untuk memprediksi risiko gangguan tidur serta memberikan rekomendasi gaya hidup
            berdasarkan kemiripan profil pengguna lain yang sehat.
        </div>
        <div class="hero-badge">✨ No Disorder &nbsp;·&nbsp; Insomnia &nbsp;·&nbsp; Sleep Apnea</div>
    </div>
    <div class="hero-moon-area">
        <div class="hero-moon"></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# --- Sidebar: Form Input ---
# ============================================================
st.sidebar.markdown("""
<div class="sidebar-blob-1"></div>
<div class="sidebar-blob-2"></div>
<div class="sidebar-hero">
    <div class="moon-mini"></div>
    <div>
        <h2>Data Diri Kamu</h2>
        <p>Isi data kesehatan & gaya hidup di bawah untuk mulai memprediksi.</p>
    </div>
</div>
""", unsafe_allow_html=True)

def user_input_features():
    st.sidebar.markdown('<div class="sidebar-section">👤 Data Pribadi & Pekerjaan</div>', unsafe_allow_html=True)
    gender = st.sidebar.selectbox("Gender", encoders['Gender'].classes_)
    age = st.sidebar.slider("Usia (Tahun)", 20, 70, 30)
    occupation = st.sidebar.selectbox("Pekerjaan", encoders['Occupation'].classes_)

    st.sidebar.markdown('<div class="sidebar-section">💤 Pola Tidur</div>', unsafe_allow_html=True)
    sleep_duration = st.sidebar.slider("Durasi Tidur (Jam/Hari)", 4.0, 10.0, 7.0, 0.1)
    quality_of_sleep = st.sidebar.slider("Kualitas Tidur (Skala 1-10)", 1, 10, 7)
    stress_level = st.sidebar.slider("Tingkat Stres (Skala 1-10)", 1, 10, 5)

    st.sidebar.markdown('<div class="sidebar-section">🏃 Aktivitas & Fisik</div>', unsafe_allow_html=True)
    physical_activity = st.sidebar.slider("Aktivitas Fisik (Menit/Hari)", 10, 120, 60)
    bmi_category = st.sidebar.selectbox("Kategori BMI", encoders['BMI Category'].classes_)
    daily_steps = st.sidebar.slider("Jumlah Langkah Harian", 1000, 15000, 7000, 500)

    st.sidebar.markdown('<div class="sidebar-section">❤️ Tanda Vital</div>', unsafe_allow_html=True)
    heart_rate = st.sidebar.slider("Detak Jantung (bpm)", 50, 100, 70)
    systolic_bp = st.sidebar.slider("Tekanan Darah Sistolik (mmHg)", 90, 180, 120)
    diastolic_bp = st.sidebar.slider("Tekanan Darah Diastolik (mmHg)", 60, 120, 80)

    data = {
        'Gender': gender,
        'Age': age,
        'Occupation': occupation,
        'Sleep Duration': sleep_duration,
        'Quality of Sleep': quality_of_sleep,
        'Physical Activity Level': physical_activity,
        'Stress Level': stress_level,
        'BMI Category': bmi_category,
        'Heart Rate': heart_rate,
        'Daily Steps': daily_steps,
        'Systolic_BP': systolic_bp,
        'Diastolic_BP': diastolic_bp
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# ============================================================
# --- Preprocessing ---
# ============================================================
input_processed = input_df.copy()
for col, le in encoders.items():
    input_processed[col] = le.transform(input_processed[col])
input_scaled = scaler.transform(input_processed[feature_cols])

# ============================================================
# --- Layout Utama: Tabs ---
# ============================================================
tab_predict, tab_history = st.tabs(["🔮 Prediksi & Rekomendasi", "📜 Riwayat Database"])

# ------------------------------------------------------------
# TAB 1 — PREDIKSI
# ------------------------------------------------------------
with tab_predict:
    st.subheader("📊 Ringkasan Data Input Kamu")

    chip_items = [
        ("👤", "Gender", str(input_df['Gender'].values[0])),
        ("🎂", "Usia", f"{int(input_df['Age'].values[0])} th"),
        ("💼", "Pekerjaan", str(input_df['Occupation'].values[0])),
        ("🛏️", "Durasi Tidur", f"{input_df['Sleep Duration'].values[0]:.1f} jam"),
        ("⭐", "Kualitas Tidur", f"{int(input_df['Quality of Sleep'].values[0])}/10"),
        ("🏃", "Aktivitas Fisik", f"{int(input_df['Physical Activity Level'].values[0])} mnt"),
        ("🧘", "Tingkat Stres", f"{int(input_df['Stress Level'].values[0])}/10"),
        ("⚖️", "BMI", str(input_df['BMI Category'].values[0])),
        ("❤️", "Detak Jantung", f"{int(input_df['Heart Rate'].values[0])} bpm"),
        ("👟", "Langkah Harian", f"{int(input_df['Daily Steps'].values[0])}"),
        ("🩺", "Sistolik", f"{int(input_df['Systolic_BP'].values[0])} mmHg"),
        ("🩺", "Diastolik", f"{int(input_df['Diastolic_BP'].values[0])} mmHg"),
    ]
    chips_html = "".join(
        f"""<div class="chip">
                <div class="chip-icon">{icon}</div>
                <div class="chip-label">{label}</div>
                <div class="chip-value">{value}</div>
            </div>"""
        for icon, label, value in chip_items
    )
    st.markdown(f'<div class="chip-grid">{chips_html}</div>', unsafe_allow_html=True)

    with st.expander("Lihat sebagai tabel mentah"):
        st.dataframe(input_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🔍 Prediksi Risiko, Simpan ke DB & Cari Rekomendasi", type="primary", use_container_width=True)

    if predict_clicked:
        # Lakukan Prediksi Model
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)

        # Simpan data input dan hasil prediksi ke database SQLite
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO prediction_history (
                timestamp, gender, age, occupation, sleep_duration, quality_of_sleep,
                physical_activity, stress_level, bmi_category, heart_rate, daily_steps,
                systolic_bp, diastolic_bp, prediction
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp,
            input_df['Gender'].values[0],
            int(input_df['Age'].values[0]),
            input_df['Occupation'].values[0],
            float(input_df['Sleep Duration'].values[0]),
            int(input_df['Quality of Sleep'].values[0]),
            int(input_df['Physical Activity Level'].values[0]),
            int(input_df['Stress Level'].values[0]),
            input_df['BMI Category'].values[0],
            int(input_df['Heart Rate'].values[0]),
            int(input_df['Daily Steps'].values[0]),
            int(input_df['Systolic_BP'].values[0]),
            int(input_df['Diastolic_BP'].values[0]),
            prediction
        ))
        conn.commit()
        st.success("✅ Data berhasil dianalisis dan disimpan ke database SQLite (`sleep_health.db`)!")

        st.markdown('<div class="star-divider">✦ ✧ ✦ ✧ ✦</div>', unsafe_allow_html=True)
        st.subheader("🎯 Hasil Analisis & Prediksi Model")

        col1, col2 = st.columns([1, 1.2], gap="large")

        with col1:
            if prediction == "No Disorder":
                st.markdown("""
                <div class="status-card ok">
                    <p class="status-title">🎉 Status: No Disorder</p>
                    <p class="status-desc">Kondisi tidur kamu terpantau sehat. Pertahankan pola gaya hidup saat ini!</p>
                </div>
                """, unsafe_allow_html=True)
            elif prediction == "Insomnia":
                st.markdown("""
                <div class="status-card warn">
                    <p class="status-title">⚠️ Status: Insomnia</p>
                    <p class="status-desc">Terdeteksi indikasi risiko Insomnia. Perhatikan tingkat stres dan durasi istirahat kamu.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="status-card risk">
                    <p class="status-title">🚨 Status: Sleep Apnea</p>
                    <p class="status-desc">Terdeteksi indikasi risiko Sleep Apnea. Disarankan berkonsultasi dengan tenaga medis.</p>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            with st.container(border=True):
                st.write("**Tingkat Keyakinan Prediksi (Probabilitas):**")
                classes = list(model.classes_)
                probs = prediction_proba[0]

                prob_colors = {
                    "No Disorder": "#34C77B",
                    "Insomnia": "#F2A93B",
                    "Sleep Apnea": "#F05C6B",
                }

                bars_html = ""
                for cls, p in zip(classes, probs):
                    pct = p * 100
                    color = prob_colors.get(cls, "#8B7CF6")
                    bars_html += f"""
                    <div style="margin-bottom:16px;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                            <span style="font-weight:600; color:#232946;">{cls}</span>
                            <span style="font-weight:700; color:{color};">{pct:.1f}%</span>
                        </div>
                        <div style="background:#F0EFFB; border-radius:999px; height:14px; overflow:hidden;">
                            <div style="width:{pct}%; background:{color}; height:100%; border-radius:999px;"></div>
                        </div>
                    </div>
                    """
                st.markdown(bars_html, unsafe_allow_html=True)

                with st.expander("Lihat tabel detail probabilitas"):
                    proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
                    st.dataframe(proba_df.T.rename(columns={0: "Probabilitas"}), use_container_width=True)

        # --- Sistem Rekomendasi Berbasis Kemiripan (Similarity-Based) ---
        st.markdown('<div class="star-divider">✦ ✧ ✦ ✧ ✦</div>', unsafe_allow_html=True)
        st.subheader("💡 Sistem Rekomendasi Gaya Hidup Sehat")

        df_healthy = df_clean[df_clean['Sleep Disorder'] == 'No Disorder'].copy()

        if len(df_healthy) > 0:
            rec_features = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
                            'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic_BP', 'Diastolic_BP']

            rec_scaler = StandardScaler()
            healthy_scaled = rec_scaler.fit_transform(df_healthy[rec_features])
            user_rec_scaled = rec_scaler.transform(input_df[rec_features])

            nn = NearestNeighbors(n_neighbors=min(3, len(df_healthy)))
            nn.fit(healthy_scaled)
            distances, indices = nn.kneighbors(user_rec_scaled)

            similar_profiles = df_healthy.iloc[indices[0]]

            target_sleep = similar_profiles['Sleep Duration'].mean()
            target_activity = similar_profiles['Physical Activity Level'].mean()
            target_steps = similar_profiles['Daily Steps'].mean()

            st.caption("Dibandingkan dengan 3 profil pengguna paling mirip yang tidak memiliki gangguan tidur:")

            rcol1, rcol2, rcol3 = st.columns(3)
            rcol1.metric(
                "🛏️ Target Tidur",
                f"{target_sleep:.1f} jam",
                delta=f"{target_sleep - input_df['Sleep Duration'].values[0]:+.1f} jam vs kamu"
            )
            rcol2.metric(
                "🏃 Target Aktivitas",
                f"{target_activity:.0f} menit",
                delta=f"{target_activity - input_df['Physical Activity Level'].values[0]:+.0f} menit vs kamu"
            )
            rcol3.metric(
                "👟 Target Langkah",
                f"{target_steps:.0f}",
                delta=f"{target_steps - input_df['Daily Steps'].values[0]:+.0f} vs kamu"
            )
    else:
        st.info("👈 Isi data di sidebar, lalu klik tombol di atas untuk melihat hasil prediksi dan rekomendasi.")

# ------------------------------------------------------------
# TAB 2 — RIWAYAT
# ------------------------------------------------------------
with tab_history:
    st.subheader("📜 Riwayat Prediksi Tersimpan (SQLite Database)")

    show_history = st.checkbox("Tampilkan Tabel Riwayat Prediksi")

    if show_history:
        history_df = pd.read_sql("SELECT * FROM prediction_history ORDER BY id DESC", conn)

        if not history_df.empty:
            with st.container(border=True):
                st.dataframe(history_df, use_container_width=True, hide_index=True)

            if st.button("🗑️ Kosongkan Riwayat Database"):
                cursor = conn.cursor()
                cursor.execute("DELETE FROM prediction_history")
                conn.commit()
                st.success("Riwayat berhasil dibersihkan!")
                st.rerun()
        else:
            st.info("Belum ada riwayat prediksi yang tersimpan di dalam database.")

# ============================================================
# --- Footer ---
# ============================================================
st.markdown("""
<div class="footer-note">🌙 Sleep Health Analytics &nbsp;·&nbsp; Team 9 &nbsp;·&nbsp; Powered by Machine Learning</div>
""", unsafe_allow_html=True)