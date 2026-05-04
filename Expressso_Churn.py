import pandas as pd
import streamlit as st
import numpy as np
import warnings
import os
import pickle

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Expresso Churn Predictor",
    page_icon="📡",
    layout="wide",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; }

  .hero-title { font-size: 2.8rem; font-weight: 700; color: #860A35; text-align: center; margin-bottom: 0; }
  .hero-sub   { font-size: 1.05rem; color: #888; text-align: center; margin-top: 4px; }
  .section-header {
    font-size: 1.4rem; font-weight: 600; color: #860A35;
    border-bottom: 2px solid #860A35; padding-bottom: 6px; margin-top: 1.8rem;
  }
  .feature-card {
    background: #fff8f9; border-left: 4px solid #860A35;
    border-radius: 6px; padding: 14px 18px; margin-bottom: 10px;
  }
  .feature-card h4 { color: #860A35; margin: 0 0 4px 0; font-size: 1rem; }
  .feature-card p  { color: #555; margin: 0; font-size: 0.9rem; }
  .stat-card {
    background: #860A35; color: white; border-radius: 10px;
    padding: 18px; text-align: center;
  }
  .stat-card .value { font-size: 1.8rem; font-weight: 700; }
  .stat-card .label { font-size: 0.85rem; opacity: 0.85; margin-top: 2px; }
  .result-churn {
    background: linear-gradient(135deg, #860A35, #5c0724);
    color: white; border-radius: 12px; padding: 28px; text-align: center; margin-top: 1.5rem;
  }
  .result-safe {
    background: linear-gradient(135deg, #2e7d32, #1b5e20);
    color: white; border-radius: 12px; padding: 28px; text-align: center; margin-top: 1.5rem;
  }
  .result-verdict { font-size: 2.2rem; font-weight: 700; }
  .result-label   { font-size: 0.95rem; opacity: 0.85; }
  hr.divider { border: none; border-top: 1px solid #e0e0e0; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Data + model ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('Expresso_churn_dataset.csv')


@st.cache_resource
def load_model():
    model_path = 'Expresso_churn.pkl'
    if os.path.exists(model_path):
        return pickle.load(open(model_path, 'rb'))

    data = load_data()
    df = data.copy()

    categoricals = df.select_dtypes(include=['object', 'category'])
    numericals   = df.select_dtypes(include='number')

    def remove_outliers(df):
        for col in df.columns:
            if df[col].dtype != 'O':
                Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                IQR = Q3 - Q1
                df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
        return df

    df = remove_outliers(df)

    scaler  = StandardScaler()
    encoder = LabelEncoder()

    for col in numericals.columns:
        if col in df.drop('CHURN', axis=1).columns:
            df[col] = scaler.fit_transform(df[[col]])

    for col in categoricals.columns:
        if col in df.drop('CHURN', axis=1).columns:
            df[col] = encoder.fit_transform(df[col])

    sel_cols = ['REGULARITY', 'DATA_VOLUME', 'REVENUE', 'ORANGE', 'ON_NET', 'MONTANT', 'FREQUENCE']
    X = df[sel_cols]
    y = df['CHURN']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.20, random_state=75, stratify=y)
    mdl = RandomForestClassifier(n_estimators=100, random_state=42)
    mdl.fit(X_train, y_train)
    pickle.dump(mdl, open(model_path, 'wb'))
    return mdl


@st.cache_data
def get_scaled_ranges():
    data = load_data()
    df = data.copy()

    def remove_outliers(df):
        for col in df.columns:
            if df[col].dtype != 'O':
                Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                IQR = Q3 - Q1
                df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
        return df

    df = remove_outliers(df)
    sel = df[['REGULARITY', 'DATA_VOLUME', 'REVENUE', 'ORANGE', 'ON_NET', 'MONTANT', 'FREQUENCE']]
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(sel), columns=sel.columns)
    return sel, scaled, scaler


data  = load_data()
model = load_model()
raw_df, scaled_df, fit_scaler = get_scaled_ranges()

sel_cols = ['REGULARITY', 'DATA_VOLUME', 'REVENUE', 'ORANGE', 'ON_NET', 'MONTANT', 'FREQUENCE']


# ── Feature metadata ───────────────────────────────────────────────────────────
FEATURES = [
    {
        "key": "REGULARITY",
        "label": "Regularity",
        "icon": "📅",
        "desc": "How consistently the customer uses Expresso services over a period. High regularity means frequent, habitual usage — a strong retention signal.",
    },
    {
        "key": "DATA_VOLUME",
        "label": "Data Volume",
        "icon": "📶",
        "desc": "Total mobile data consumed by the customer (in MB or GB). Higher consumption typically indicates a more engaged user.",
    },
    {
        "key": "REVENUE",
        "label": "Revenue",
        "icon": "💰",
        "desc": "Total revenue generated from the customer (calls, data, etc.). Customers with low revenue are often higher churn risks.",
    },
    {
        "key": "ORANGE",
        "label": "Orange Calls",
        "icon": "🟠",
        "desc": "Number of calls made to the Orange network (a competing telecom). High inter-network calls can indicate divided loyalty.",
    },
    {
        "key": "ON_NET",
        "label": "On-Net Calls",
        "icon": "📞",
        "desc": "Number of calls made within the Expresso network. Higher on-net activity suggests stronger in-network engagement.",
    },
    {
        "key": "MONTANT",
        "label": "Top-Up Amount (Montant)",
        "icon": "💳",
        "desc": "Total recharge/top-up amount. Customers who top up frequently and in larger amounts are generally less likely to churn.",
    },
    {
        "key": "FREQUENCE",
        "label": "Recharge Frequency",
        "icon": "🔄",
        "desc": "Number of times the customer recharged their account. A high recharge frequency is a positive engagement indicator.",
    },
]


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">📡 Expresso Churn Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Customer Churn Prediction for Expresso Telecom &nbsp;|&nbsp; Built by Datapsalm</p>', unsafe_allow_html=True)
st.markdown('<br>', unsafe_allow_html=True)

tab_home, tab_predict, tab_data = st.tabs(["🏠 Overview", "🔮 Predict Churn", "📊 Dataset Explorer"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_home:
    col_img, col_desc = st.columns([1, 1.6], gap="large")

    with col_img:
        st.image('bg.jpeg', use_column_width=True)

    with col_desc:
        st.markdown('<p class="section-header">About This App</p>', unsafe_allow_html=True)
        st.markdown("""
**Expresso** is a major telecommunications provider in Africa under the Sudatel Group, connecting
millions of users across multiple nations with mobile and internet services.

This app uses a **Random Forest Classifier** trained on over 2.5 million Expresso customer records
to predict the likelihood of churn — whether a customer is about to leave the network.

Telecoms that can anticipate churn can act early with targeted retention campaigns, better offers,
and proactive customer support before customers walk away.
        """)

        st.markdown('<br>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        churn_rate = data['CHURN'].mean() * 100 if 'CHURN' in data.columns else 0
        with c1:
            st.markdown(f'<div class="stat-card"><div class="value">{len(data):,}</div><div class="label">Customer Records</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-card"><div class="value">{churn_rate:.1f}%</div><div class="label">Churn Rate</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="stat-card"><div class="value">7</div><div class="label">Model Features</div></div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">Feature Guide</p>', unsafe_allow_html=True)
    st.markdown("The model uses seven behavioral signals to assess churn risk. Understanding each helps you enter accurate values.")
    st.markdown('<br>', unsafe_allow_html=True)

    left, right = st.columns(2, gap="medium")
    for i, feat in enumerate(FEATURES):
        col = left if i % 2 == 0 else right
        with col:
            st.markdown(f"""
            <div class="feature-card">
              <h4>{feat['icon']} {feat['label']}</h4>
              <p>{feat['desc']}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">How the Model Works</p>', unsafe_allow_html=True)
    cols = st.columns(4, gap="medium")
    for col, (step, label) in zip(cols, [
        ("1️⃣", "Outliers are removed using the IQR method to clean the data"),
        ("2️⃣", "Numeric features are scaled; categorical features are label-encoded"),
        ("3️⃣", "100 decision trees are trained on 80% of the data"),
        ("4️⃣", "Your inputs are scaled and voted on by all trees — majority wins"),
    ]):
        with col:
            st.info(f"**{step}** {label}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown('<p class="section-header">Customer Profile</p>', unsafe_allow_html=True)
    st.markdown("Enter the customer's usage data below and press **Predict** to assess churn risk.")
    st.markdown('<br>', unsafe_allow_html=True)

    input_type = st.radio("Input style", ["Sliders", "Number Inputs"], horizontal=True, label_visibility="collapsed")
    st.markdown('<br>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    def make_input(label, key, col, help_text=""):
        mn, mx = float(raw_df[key].min()), float(raw_df[key].max())
        default = float(raw_df[key].median())
        if input_type == "Sliders":
            return col.slider(label, mn, mx, default, help=help_text)
        return col.number_input(label, mn, mx, default, help=help_text)

    with col1:
        regularity  = make_input("📅 Regularity",              "REGULARITY",  col1, "How consistently the customer uses Expresso")
        data_volume = make_input("📶 Data Volume",             "DATA_VOLUME", col1, "Total data consumed")
        revenue     = make_input("💰 Revenue",                 "REVENUE",     col1, "Total revenue generated from the customer")
        orange      = make_input("🟠 Orange Calls",            "ORANGE",      col1, "Calls made to the Orange network")

    with col2:
        on_net      = make_input("📞 On-Net Calls",            "ON_NET",      col2, "Calls made within the Expresso network")
        montant     = make_input("💳 Top-Up Amount (Montant)", "MONTANT",     col2, "Total recharge amount")
        frequence   = make_input("🔄 Recharge Frequency",     "FREQUENCE",   col2, "Number of times the customer recharged")

    st.markdown('<br>', unsafe_allow_html=True)

    raw_input = pd.DataFrame([{
        'REGULARITY': regularity, 'DATA_VOLUME': data_volume,
        'REVENUE': revenue, 'ORANGE': orange,
        'ON_NET': on_net, 'MONTANT': montant, 'FREQUENCE': frequence,
    }])

    scaled_input = pd.DataFrame(
        fit_scaler.transform(raw_input),
        columns=raw_input.columns
    )

    with st.expander("Review raw input values"):
        st.dataframe(raw_input, use_container_width=True)

    if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
        pred = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0]
        churn_prob = prob[1] * 100

        if pred == 1:
            st.markdown(f"""
            <div class="result-churn">
              <div class="result-label">Prediction Result</div>
              <div class="result-verdict">⚠️ Likely to Churn</div>
              <div class="result-label" style="margin-top:10px">Churn probability: {churn_prob:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
              <div class="result-label">Prediction Result</div>
              <div class="result-verdict">✅ Not Likely to Churn</div>
              <div class="result-label" style="margin-top:10px">Retention probability: {100 - churn_prob:.1f}%</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown('<p class="section-header">Dataset Overview</p>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records", f"{len(data):,}")
    if 'CHURN' in data.columns:
        m2.metric("Churned",    f"{data['CHURN'].sum():,}")
        m3.metric("Retained",   f"{(data['CHURN'] == 0).sum():,}")
        m4.metric("Churn Rate", f"{data['CHURN'].mean()*100:.1f}%")

    st.markdown('<br>', unsafe_allow_html=True)
    preview_cols = sel_cols + (['CHURN'] if 'CHURN' in data.columns else [])
    st.markdown("**Sample rows (selected features)**")
    st.dataframe(data[preview_cols].head(20), use_container_width=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown("**Descriptive statistics**")
    st.dataframe(data[preview_cols].describe().round(2), use_container_width=True)

    if 'CHURN' in data.columns:
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown("**Churn Distribution**")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        churn_counts = data['CHURN'].value_counts()
        axes[0].pie(churn_counts, labels=['Retained', 'Churned'], autopct='%1.1f%%',
                    colors=['#2e7d32', '#860A35'], startangle=90)
        axes[0].set_title('Churn Distribution')

        sns.histplot(data=data, x='REVENUE', hue='CHURN', bins=30,
                     palette={0: '#2e7d32', 1: '#860A35'}, ax=axes[1])
        axes[1].set_title('Revenue by Churn Status')
        axes[1].set_xlabel('Revenue')
        fig.tight_layout()
        st.pyplot(fig)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image('pngwing.com (8).png')
    st.markdown("### Expresso Churn Predictor")
    st.markdown("Use the tabs to explore the data or make a churn prediction.")
    st.markdown("---")
    st.markdown("**Model:** Random Forest Classifier")
    st.markdown("**Dataset:** Expresso Telecom (Africa)")
    st.markdown("**Features:** 7 behavioral signals")
    st.markdown(f"**Records:** {len(data):,}")
    st.markdown("---")
    st.caption("Built by Datapsalm")
