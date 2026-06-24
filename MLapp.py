import streamlit as st
import pandas as pd, numpy as np, joblib, json, os
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="😴 SleepSense – Prediksi Gangguan Tidur",
    page_icon="😴", layout="wide", initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Quicksand', sans-serif; }
.stApp { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%); }

.hero {
    background: linear-gradient(135deg, #4c1d95 0%, #6d28d9 50%, #a855f7 100%);
    border-radius: 22px; padding: 2.5rem 2rem; margin-bottom: 1.5rem;
    box-shadow: 0 0 40px rgba(168,85,247,0.25);
}
.hero h1 { font-size: 2.4rem; font-weight: 700; margin: 0; color: white; }
.hero p  { font-size: 1rem; color: #e9d5ff; margin: .5rem 0 0; }

.card {
    background: #20203a; border-radius: 18px; padding: 1.6rem;
    border: 1px solid rgba(168,85,247,0.15);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3); margin-bottom: 1rem;
}
.card h3 { font-weight: 700; margin: 0 0 1rem; font-size: 1.05rem; color: #c4b5fd; }

.metric-box {
    background: linear-gradient(135deg, #20203a, #2a2a4a);
    border-radius: 16px; padding: 1.3rem; text-align: center;
    border: 1px solid rgba(168,85,247,0.2);
}
.metric-box .val { font-size: 1.8rem; font-weight: 700; color: #c4b5fd; }
.metric-box .lbl { font-size: .72rem; color: #9ca3af; font-weight: 600;
                   text-transform: uppercase; letter-spacing: 1px; margin-top: .3rem; }

.result-good {
    background: linear-gradient(135deg, #064e3b, #047857);
    border: 2px solid #34d399; border-radius: 18px; padding: 2rem; text-align: center;
    box-shadow: 0 0 30px rgba(52,211,153,0.2);
}
.result-warn {
    background: linear-gradient(135deg, #7c2d12, #c2410c);
    border: 2px solid #fb923c; border-radius: 18px; padding: 2rem; text-align: center;
    box-shadow: 0 0 30px rgba(251,146,60,0.2);
}
.result-icon  { font-size: 4rem; }
.result-label { font-size: 1.8rem; font-weight: 700; color: white; }
.result-sub   { font-size: .9rem; color: #e5e7eb; margin-top: .3rem; }

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #4c1d95 100%) !important;
}
div[data-testid="stSidebar"] p,
div[data-testid="stSidebar"] span,
div[data-testid="stSidebar"] label { color: #e9d5ff !important; }
div[data-testid="stSidebar"] h2 { color: #c4b5fd !important; }

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #7c3aed, #c026d3) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 700 !important;
    width: 100% !important; font-size: 1rem !important; padding: .7rem !important;
}

p, span, li { color: #e5e7eb !important; }
h1, h2, h3, h4 { color: #f3f4f6 !important; }
.stTabs [data-baseweb="tab"] { color: #9ca3af !important; }
.stTabs [aria-selected="true"] { color: #c4b5fd !important; border-bottom-color: #c4b5fd !important; }
</style>
""", unsafe_allow_html=True)

BASE = os.path.dirname(__file__)

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE, 'data', 'sleep_data.csv'))

@st.cache_data
def load_results():
    with open(os.path.join(BASE, 'model', 'results.json')) as f:
        return json.load(f)

@st.cache_data
def load_fi():
    with open(os.path.join(BASE, 'model', 'feature_importance.json')) as f:
        return json.load(f)

@st.cache_resource
def load_models():
    scaler  = joblib.load(os.path.join(BASE, 'model', 'scaler.pkl'))
    enc     = joblib.load(os.path.join(BASE, 'model', 'encoders.pkl'))
    tgt_enc = joblib.load(os.path.join(BASE, 'model', 'target_encoder.pkl'))
    fcols   = joblib.load(os.path.join(BASE, 'model', 'feature_cols.pkl'))
    models  = {}
    for name in ['Decision Tree','KNN','Random Forest','XGBoost']:
        fname = name.lower().replace(' ', '_')
        models[name] = joblib.load(os.path.join(BASE, 'model', f'{fname}_model.pkl'))
    return scaler, enc, tgt_enc, fcols, models

df = load_data()
df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
results = load_results()
feat_importance = load_fi()
scaler, encoders, target_le, FEATURE_COLS, models = load_models()

PLOT_THEME = dict(plot_bgcolor='#20203a', paper_bgcolor='#20203a',
                  font_color='#e5e7eb', margin=dict(t=30,b=10,l=10,r=10))
DISORDER_COLORS = {'None': '#34d399', 'Insomnia': '#fb923c', 'Sleep Apnea': '#f87171'}
DISORDER_EMOJI  = {'None': '😴', 'Insomnia': '😵', 'Sleep Apnea': '😮\u200d💨'}

with st.sidebar:
    st.markdown("## 😴 SleepSense")
    st.markdown("---")
    page = st.radio("Navigasi", [
        "🏠 Beranda", "📊 Analisis Data", "🤖 Perbandingan Model", "🔍 Cek Kesehatan Tidur"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Kelompok EAS**")
    st.markdown("Mata Kuliah: Pembelajaran Mesin")
    st.markdown("Semester Genap 2025/2026")

# ════════════════════════════════════════
# BERANDA
# ════════════════════════════════════════
if page == "🏠 Beranda":
    st.markdown("""<div class="hero">
        <h1>😴 SleepSense</h1>
        <p>Sistem Prediksi Gangguan Tidur Berbasis Machine Learning · Gaya Hidup Sehat untuk Semua</p>
    </div>""", unsafe_allow_html=True)

    best_model = max(results, key=lambda x: results[x]['accuracy'])
    c1,c2,c3,c4 = st.columns(4)
    for col,(val,lbl) in zip([c1,c2,c3,c4],[
        (str(len(df)), "Total Data"),
        (str(df['Sleep Disorder'].nunique()), "Kategori Gangguan"),
        (f"{results[best_model]['accuracy']*100:.1f}%", "Akurasi Terbaik"),
        ("4", "Algoritma ML"),
    ]):
        col.markdown(f'<div class="metric-box"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="card"><h3>📌 Tentang Proyek</h3>', unsafe_allow_html=True)
        st.markdown("""
**SleepSense** membantu mendeteksi risiko gangguan tidur berdasarkan gaya hidup sehari-hari.

**Fitur yang dianalisis:**
- 💼 Pekerjaan & usia
- 😰 Level stress
- 🏃 Aktivitas fisik & langkah harian
- ⏰ Durasi & kualitas tidur
- ⚖️ BMI & tekanan darah

**Output:** None / Insomnia / Sleep Apnea
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    with cb:
        vc = df['Sleep Disorder'].value_counts()
        fig = go.Figure(go.Pie(
            labels=vc.index, values=vc.values, hole=.55,
            marker_colors=[DISORDER_COLORS.get(x,'#888') for x in vc.index],
        ))
        fig.update_layout(title='Distribusi Gangguan Tidur', **PLOT_THEME, height=320,
                          legend=dict(font_color='#ccc'))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="card"><h3>📋 Preview Dataset</h3>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════
# ANALISIS DATA
# ════════════════════════════════════════
elif page == "📊 Analisis Data":
    st.markdown("""<div class="hero">
        <h1>📊 Analisis Data</h1>
        <p>Eksplorasi pola gangguan tidur berdasarkan gaya hidup</p>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Distribusi Fitur", "🔗 Korelasi", "💼 Per Pekerjaan"])

    with tab1:
        feat = st.selectbox("Pilih Fitur Numerik",
                            ['Age','Sleep Duration','Quality of Sleep','Physical Activity Level',
                             'Stress Level','Heart Rate','Daily Steps'])
        fig = go.Figure()
        for disorder in df['Sleep Disorder'].unique():
            sub = df[df['Sleep Disorder']==disorder]
            fig.add_trace(go.Box(y=sub[feat], name=disorder,
                                 marker_color=DISORDER_COLORS.get(disorder,'#888')))
        fig.update_layout(title=f'{feat} berdasarkan Gangguan Tidur', **PLOT_THEME, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        num_cols = ['Age','Sleep Duration','Quality of Sleep','Physical Activity Level',
                   'Stress Level','Heart Rate','Daily Steps']
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale='Purples', height=480)
        fig.update_layout(**PLOT_THEME, title='Heatmap Korelasi Antar Fitur')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        occ_disorder = pd.crosstab(df['Occupation'], df['Sleep Disorder'])
        fig = go.Figure()
        for disorder in df['Sleep Disorder'].unique():
            if disorder in occ_disorder.columns:
                fig.add_trace(go.Bar(name=disorder, x=occ_disorder.index,
                                     y=occ_disorder[disorder],
                                     marker_color=DISORDER_COLORS.get(disorder,'#888')))
        fig.update_layout(barmode='stack', title='Gangguan Tidur per Pekerjaan',
                          xaxis_tickangle=-40, **PLOT_THEME, height=450)
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════
# PERBANDINGAN MODEL
# ════════════════════════════════════════
elif page == "🤖 Perbandingan Model":
    st.markdown("""<div class="hero">
        <h1>🤖 Perbandingan Model</h1>
        <p>Evaluasi Decision Tree, KNN, Random Forest, dan XGBoost</p>
    </div>""", unsafe_allow_html=True)

    names = list(results.keys())
    accs  = [results[n]['accuracy'] for n in names]

    cols = st.columns(len(names))
    for col, name, acc in zip(cols, names, accs):
        col.markdown(f'<div class="metric-box"><div class="lbl">{name}</div><div class="val">{acc*100:.1f}%</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📈 Akurasi", "🎯 Confusion Matrix", "💡 Feature Importance"])

    with tab1:
        fig = go.Figure(go.Bar(
            x=names, y=[a*100 for a in accs],
            marker_color=['#a78bfa','#f472b6','#60a5fa','#34d399'],
            text=[f"{a*100:.2f}%" for a in accs], textposition='outside', textfont_color='white',
        ))
        fig.update_layout(title='Akurasi 4 Model', yaxis=dict(range=[0,110]), **PLOT_THEME, height=400)
        st.plotly_chart(fig, use_container_width=True)
        best = max(results, key=lambda x: results[x]['accuracy'])
        st.success(f"🏆 Model terbaik: **{best}** dengan akurasi **{results[best]['accuracy']*100:.2f}%**")

    with tab2:
        model_sel = st.selectbox("Pilih Model", names)
        cm = np.array(results[model_sel]['cm'])
        labels = target_le.classes_.tolist()
        fig = px.imshow(cm, text_auto=True, x=labels, y=labels,
                        labels=dict(x="Prediksi", y="Aktual"),
                        color_continuous_scale='Purples', height=420)
        fig.update_layout(**PLOT_THEME, title=f'Confusion Matrix – {model_sel}')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fi_sorted = dict(sorted(feat_importance.items(), key=lambda x: x[1]))
        fig = go.Figure(go.Bar(
            x=list(fi_sorted.values()), y=list(fi_sorted.keys()), orientation='h',
            marker_color='#a78bfa',
        ))
        fig.update_layout(title='Feature Importance (Random Forest)', **PLOT_THEME, height=420)
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════
# PREDIKSI
# ════════════════════════════════════════
elif page == "🔍 Cek Kesehatan Tidur":
    st.markdown("""<div class="hero">
        <h1>🔍 Cek Kesehatan Tidur</h1>
        <p>Isi data diri kamu untuk memeriksa risiko gangguan tidur</p>
    </div>""", unsafe_allow_html=True)

    model_sel = st.selectbox("Pilih Model", list(models.keys()))
    clf = models[model_sel]

    st.markdown('<div class="card"><h3>👤 Data Diri</h3>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    gender = c1.selectbox("Gender", encoders['Gender'].classes_)
    age    = c2.slider("Usia", 18, 70, 28)
    occ_options = encoders['Occupation'].classes_
    occupation = c3.selectbox("Pekerjaan", occ_options)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>😴 Pola Tidur</h3>', unsafe_allow_html=True)
    c4, c5 = st.columns(2)
    sleep_dur = c4.slider("Durasi Tidur (jam)", 4.0, 10.0, 7.0, step=0.1)
    sleep_qual = c5.slider("Kualitas Tidur (1-10)", 1, 10, 7)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>🏃 Gaya Hidup & Kesehatan</h3>', unsafe_allow_html=True)
    c6, c7, c8 = st.columns(3)
    activity = c6.slider("Aktivitas Fisik (menit/hari)", 0, 120, 45)
    stress   = c7.slider("Level Stress (1-10)", 1, 10, 5)
    steps    = c8.number_input("Langkah Harian", 1000, 20000, 6000, step=500)

    c9, c10, c11 = st.columns(3)
    bmi_cat  = c9.selectbox("Kategori BMI", encoders['BMI Category'].classes_)
    hr       = c10.slider("Detak Jantung (bpm)", 50, 110, 72)
    bp_sys   = c11.number_input("Tekanan Darah Sistolik", 90, 180, 120)
    bp_dia   = st.number_input("Tekanan Darah Diastolik", 60, 120, 80)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔍 Analisis Kesehatan Tidur Saya"):
        input_dict = {
            'Gender': encoders['Gender'].transform([gender])[0],
            'Age': age,
            'Occupation': encoders['Occupation'].transform([occupation])[0],
            'Sleep Duration': sleep_dur,
            'Quality of Sleep': sleep_qual,
            'Physical Activity Level': activity,
            'Stress Level': stress,
            'BMI Category': encoders['BMI Category'].transform([bmi_cat])[0],
            'Heart Rate': hr,
            'Daily Steps': steps,
            'BP_Systolic': bp_sys,
            'BP_Diastolic': bp_dia,
        }
        input_df = pd.DataFrame([input_dict])[FEATURE_COLS]
        input_scaled = scaler.transform(input_df)

        pred_idx = clf.predict(input_scaled)[0]
        pred_label = target_le.inverse_transform([pred_idx])[0]
        proba = clf.predict_proba(input_scaled)[0] if hasattr(clf,'predict_proba') else None

        box_class = "result-good" if pred_label == "None" else "result-warn"
        emoji = DISORDER_EMOJI.get(pred_label, "😴")

        if pred_label == "None":
            sub_text = "Pola tidurmu terlihat sehat! Pertahankan gaya hidup ini 👍"
        elif pred_label == "Insomnia":
            sub_text = "Terdeteksi indikasi insomnia. Pertimbangkan kelola stress & rutinitas tidur."
        else:
            sub_text = "Terdeteksi indikasi sleep apnea. Disarankan konsultasi ke dokter spesialis tidur."

        st.markdown(f"""
        <div class="{box_class}">
            <div class="result-icon">{emoji}</div>
            <div class="result-label">{pred_label}</div>
            <div class="result-sub">{sub_text}</div>
        </div>""", unsafe_allow_html=True)

        if proba is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="card"><h3>📊 Probabilitas Tiap Kategori</h3>', unsafe_allow_html=True)
            classes = target_le.classes_
            fig = go.Figure(go.Bar(
                x=classes, y=[p*100 for p in proba],
                marker_color=[DISORDER_COLORS.get(c,'#888') for c in classes],
                text=[f"{p*100:.1f}%" for p in proba], textposition='outside', textfont_color='white',
            ))
            fig.update_layout(yaxis=dict(range=[0,115]), **PLOT_THEME, height=320)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.info("⚠️ **Disclaimer:** Hasil ini hanya prediksi berbasis data dan bukan diagnosis medis. Konsultasikan ke dokter untuk pemeriksaan lebih lanjut.")
