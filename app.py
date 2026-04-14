import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Load model and scaler ──────────────────────────────────────────────
model = joblib.load('models/stroke_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="🧠",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    .hero-title {
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(90deg, #f953c6, #b91d73, #f953c6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        text-align: center;
        color: #aaa;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }

    .card-title {
        font-size: 1rem;
        font-weight: 700;
        color: #f953c6;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 1rem;
    }

    .risk-high {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        font-size: 1.8rem;
        font-weight: 900;
    }

    .risk-low {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        font-size: 1.8rem;
        font-weight: 900;
    }

    .metric-box {
        background: rgba(249, 83, 198, 0.15);
        border: 1px solid #f953c6;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 900;
        color: #f953c6;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton > button {
        background: linear-gradient(90deg, #f953c6, #b91d73) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        padding: 0.75rem 2rem !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 8px 25px rgba(249, 83, 198, 0.4) !important;
    }

    .disclaimer {
        text-align: center;
        color: #666;
        font-size: 0.8rem;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid rgba(255,255,255,0.1);
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stNumberInput"] label {
        color: #ddd !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Hero Header ────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🧠 Stroke Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Enter patient details below to assess stroke risk using AI</div>', unsafe_allow_html=True)
st.divider()

# ── Input Form ─────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="card"><div class="card-title">👤 Demographics</div>', unsafe_allow_html=True)
    gender = st.selectbox("Gender", options=["Female", "Male"])
    age = st.slider("Age", min_value=1, max_value=100, value=45)
    ever_married = st.selectbox("Ever Married", options=["No", "Yes"])
    residence_type = st.selectbox("Residence Type", options=["Rural", "Urban"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card"><div class="card-title">❤️ Health Conditions</div>', unsafe_allow_html=True)
    hypertension = st.selectbox("Hypertension", options=["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", options=["No", "Yes"])
    avg_glucose_level = st.number_input("Average Glucose Level",
                                         min_value=50.0,
                                         max_value=300.0,
                                         value=100.0)
    bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=28.0)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card"><div class="card-title">🏃 Lifestyle</div>', unsafe_allow_html=True)
    work_type = st.selectbox("Work Type", options=[
        "Govt_job", "Never_worked", "Private",
        "Self-employed", "children"
    ])
    smoking_status = st.selectbox("Smoking Status", options=[
        "Unknown", "formerly smoked", "never smoked", "smokes"
    ])
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# ── Predict Button ─────────────────────────────────────────────────────
predict = st.button("🔍 Predict Stroke Risk", type="primary", use_container_width=True)

if predict:
    # ── Build input ────────────────────────────────────────────────────
    input_dict = {
        'gender': 1 if gender == 'Male' else 0,
        'age': age,
        'hypertension': 1 if hypertension == 'Yes' else 0,
        'heart_disease': 1 if heart_disease == 'Yes' else 0,
        'ever_married': 1 if ever_married == 'Yes' else 0,
        'Residence_type': 1 if residence_type == 'Urban' else 0,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'work_type_Govt_job': 1 if work_type == 'Govt_job' else 0,
        'work_type_Never_worked': 1 if work_type == 'Never_worked' else 0,
        'work_type_Private': 1 if work_type == 'Private' else 0,
        'work_type_Self-employed': 1 if work_type == 'Self-employed' else 0,
        'work_type_children': 1 if work_type == 'children' else 0,
        'smoking_status_Unknown': 1 if smoking_status == 'Unknown' else 0,
        'smoking_status_formerly smoked': 1 if smoking_status == 'formerly smoked' else 0,
        'smoking_status_never smoked': 1 if smoking_status == 'never smoked' else 0,
        'smoking_status_smokes': 1 if smoking_status == 'smokes' else 0,
    }

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    probability = model.predict_proba(input_scaled)[0][1]
    prediction = int(probability >= 0.3)

    st.subheader("📊 Results")
    res_col1, res_col2, res_col3 = st.columns([2, 1, 2])

    # ── Risk label ─────────────────────────────────────────────────────
    with res_col1:
        if prediction == 1:
            st.markdown(f'<div class="risk-high">🔴 High Risk</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk-low">🟢 Low Risk</div>', unsafe_allow_html=True)

    # ── Probability metric ─────────────────────────────────────────────
    with res_col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{probability:.1%}</div>
            <div class="metric-label">Risk Probability</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Risk gauge ─────────────────────────────────────────────────────
    with res_col3:
        fig, ax = plt.subplots(figsize=(4, 2.2))
        fig.patch.set_alpha(0)
        ax.set_facecolor('none')

        # Background bar
        ax.barh(0, 1, color='#333', height=0.4, edgecolor='none')

        # Filled bar
        color = '#ff416c' if prediction == 1 else '#38ef7d'
        ax.barh(0, probability, color=color, height=0.4, edgecolor='none')

        # Threshold line
        ax.axvline(x=0.3, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(0.3, 0.35, 'Threshold', color='white', fontsize=7,
                ha='center', va='bottom', alpha=0.7)

        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], color='white', fontsize=8)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title('Risk Gauge', color='white', fontsize=10, fontweight='bold')

        st.pyplot(fig)

    # ── Feature importance ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔬 What's Driving This Prediction?")

    feature_names = list(input_dict.keys())
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True).tail(10)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    fig2.patch.set_alpha(0)
    ax2.set_facecolor('none')

    colors = ['#f953c6' if i >= len(feat_df) - 3 else '#7b2ff7'
              for i in range(len(feat_df))]
    bars = ax2.barh(feat_df['Feature'], feat_df['Importance'],
                    color=colors, edgecolor='none')

    ax2.set_xlabel('Importance Score', color='white')
    ax2.tick_params(colors='blue')
    ax2.xaxis.label.set_color('white')
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.set_title('Top 10 Feature Importances', color='darkblue',
                  fontsize=12, fontweight='bold')

    top_patch = mpatches.Patch(color='#f953c6', label='Top 3 features')
    other_patch = mpatches.Patch(color='#7b2ff7', label='Other features')
    ax2.legend(handles=[top_patch, other_patch], facecolor='#1a1a2e',
               labelcolor='white', fontsize=8)

    st.pyplot(fig2)

    # ── Risk flags ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚠️ Risk Flags")

    flags = {
        "Age above 60": age >= 60,
        "Has Hypertension": hypertension == "Yes",
        "Has Heart Disease": heart_disease == "Yes",
        "High Glucose (>140)": avg_glucose_level > 140,
        "Obese BMI (>30)": bmi > 30,
        "Formerly or Currently Smokes": smoking_status in ["formerly smoked", "smokes"],
    }

    flag_cols = st.columns(3)
    for i, (flag, triggered) in enumerate(flags.items()):
        with flag_cols[i % 3]:
            if triggered:
                st.error(f"🚨 {flag}")
            else:
                st.success(f"✅ {flag}")

    # ── Disclaimer ─────────────────────────────────────────────────────
    st.markdown(
        '<div class="disclaimer">⚠️ This tool is for educational purposes only '
        'and is not a substitute for professional medical advice.</div>',
        unsafe_allow_html=True
    )