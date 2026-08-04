"""
Walmart Weekly Sales Predictor — Streamlit App (v7)
====================================================
Supports THREE prediction pipelines:
  • Normalized RF      → loads model_norm.pkl + scaler.pkl + feature_names_norm.json
  • Unnormalized RF    → loads model_raw.pkl  + feature_names_raw.json
  • Linear Regression  → loads linear_regression_model.pkl + scaler.pkl + feature_names_linear.json

Critical fixes vs v4 app:
  ① Scaler is now loaded and applied before prediction in the normalized pipeline.
     Previously: raw Temperature/CPI/etc. were sent to a model trained on [0,1] values.
     Now: scaler.transform() is called on the 5 economic columns before predict().
  ② Pipeline selector lets the user choose which model to use.
  ③ build_input_row() is now pipeline-aware: normalized path applies scaler,
     raw path passes values directly.
  ④ Feature column ordering is enforced from the saved JSON files — never guessed.
  ⑤ Full error handling: missing files, invalid inputs, prediction exceptions.
  ⑥ Explanation panel shows which inputs currently have low effect and why.

v6 fixes (MarkDown):
  ⑦ MarkDown slider max raised to 200,000 — matches full training range (~$160K).
  ⑧ Debug panel now shows MarkDown raw + scaled values and % of training range covered.
  ⑨ MarkDown insight card updated to explain the slider range requirement and sparsity.

v7 addition (Linear Regression):
  ⑩ Linear Regression pipeline added (notebook Section 17).
     Uses same normalized features as Normalized RF — reuses scaler.pkl.
     Pipeline key: 'lr'. Coefficients replace feature importance.
     No fit_transform() during inference — only scaler.transform() on SCALER_COLS.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Walmart Sales Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0b0f1a;
    color: #e8eaf0;
}
.block-container { padding: 2rem 3rem 4rem; max-width: 1200px; }

.hero {
    background: linear-gradient(135deg, #0d47a1 0%, #1565c0 40%, #0a2a6e 100%);
    border-radius: 20px;
    padding: 2.5rem 3rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(13,71,161,0.4);
}
.hero::before {
    content:''; position:absolute; top:-60px; right:-60px;
    width:280px; height:280px;
    background:rgba(255,255,255,0.04); border-radius:50%;
}
.hero-tag {
    display:inline-block; background:rgba(255,255,255,0.15);
    color:#90caf9; font-size:0.72rem; font-weight:600;
    letter-spacing:0.18em; text-transform:uppercase;
    padding:0.3rem 0.9rem; border-radius:100px; margin-bottom:0.8rem;
}
.hero h1 {
    font-family:'Syne',sans-serif; font-size:2.4rem; font-weight:800;
    color:#fff; margin:0 0 0.5rem; line-height:1.15;
}
.hero p { color:#bbdefb; font-size:0.95rem; font-weight:300; margin:0; line-height:1.6; }

.section-title {
    font-family:'Syne',sans-serif; font-size:0.7rem; font-weight:700;
    letter-spacing:0.2em; text-transform:uppercase; color:#5c9ee8;
    margin:1.8rem 0 0.9rem; padding-bottom:0.4rem; border-bottom:1px solid #1e2d4a;
}
.card {
    background:#111827; border:1px solid #1e2d4a;
    border-radius:16px; padding:1.6rem 1.8rem; margin-bottom:1.2rem;
}
.badge {
    background:#0d1e3a; border:1px solid #1e3a5f;
    border-radius:10px; padding:0.6rem 1rem;
    font-size:0.82rem; color:#7aabdf; margin-bottom:1rem;
}
.chip-row { display:flex; gap:0.5rem; flex-wrap:wrap; margin-bottom:1.2rem; }
.chip {
    background:#0d1e3a; border:1px solid #1e3a5f;
    border-radius:100px; padding:0.22rem 0.75rem;
    font-size:0.72rem; color:#7aabdf; font-weight:500;
}
.chip-norm  { border-color:#1565c0; color:#90caf9; background:#0a1e3a; }
.chip-raw   { border-color:#e65100; color:#ffcc80; background:#1a0e00; }
.chip-warn  { border-color:#7a4f00; color:#ffd54f; background:#1a1200; }

label, .stNumberInput label, .stSelectbox label {
    font-size:0.82rem !important; font-weight:500 !important;
    color:#8ba8cc !important; letter-spacing:0.04em; text-transform:uppercase;
}
.stNumberInput input, .stSelectbox select {
    background:#0d1526 !important; border:1px solid #1e3a5f !important;
    border-radius:10px !important; color:#e8eaf0 !important;
}
.stButton > button {
    background:linear-gradient(135deg,#1565c0,#0d47a1) !important;
    color:#fff !important; font-family:'Syne',sans-serif !important;
    font-size:0.95rem !important; font-weight:700 !important;
    border:none !important; border-radius:12px !important;
    padding:0.8rem 2rem !important; width:100% !important;
    box-shadow:0 6px 24px rgba(13,71,161,0.5) !important;
    transition:all 0.2s ease !important;
}
.stButton > button:hover { transform:translateY(-2px) !important; }

.result-box {
    background:linear-gradient(135deg,#0d3b7a,#0a2a6e);
    border:1px solid #1565c0; border-radius:16px;
    padding:2rem 2.2rem; text-align:center; margin-top:1.2rem;
    box-shadow:0 12px 40px rgba(13,71,161,0.35);
    animation:fadeUp 0.5s ease forwards;
}
.result-norm { border-color:#1565c0; }
.result-raw  { border-color:#e65100; background:linear-gradient(135deg,#1a0e00,#0d0500); }
.result-label { font-size:0.72rem; font-weight:600; letter-spacing:0.2em; text-transform:uppercase; color:#90caf9; margin-bottom:0.4rem; }
.result-value { font-family:'Syne',sans-serif; font-size:3rem; font-weight:800; color:#fff; line-height:1.1; }
.result-sub   { font-size:0.8rem; color:#64b5f6; margin-top:0.5rem; }

.error-box {
    background:#1a0a0a; border:1px solid #7f1d1d;
    border-radius:12px; padding:1.2rem 1.5rem;
    color:#fca5a5; font-size:0.88rem; margin-top:0.8rem;
}
.info-box {
    background:#0a1e12; border:1px solid #1a4a2a;
    border-radius:12px; padding:1.2rem 1.5rem;
    color:#86efac; font-size:0.85rem; margin-top:0.8rem;
}
.warn-box {
    background:#1a1200; border:1px solid #7a4f00;
    border-radius:12px; padding:1.2rem 1.5rem;
    color:#ffd54f; font-size:0.85rem; margin-top:0.8rem;
}
.debug-row {
    font-family: monospace; font-size:0.78rem; color:#5c9ee8;
    background:#0a0f1a; padding:0.2rem 0.5rem; border-radius:4px;
    margin:0.1rem 0;
}

@keyframes fadeUp {
    from { opacity:0; transform:translateY(12px); }
    to   { opacity:1; transform:translateY(0); }
}
hr { border-color:#1e2d4a; margin:1.2rem 0; }
#MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STORE METADATA  (mirrors stores.csv — used to auto-resolve Type & Size)
# ══════════════════════════════════════════════════════════════════════════════
STORE_META = {
     1:("A",151315),  2:("A",202307),  3:("B", 37392),  4:("A",205863),
     5:("B", 34875),  6:("A",202505),  7:("B", 70713),  8:("A",155078),
     9:("B",125833), 10:("B",126512), 11:("A",207499), 12:("B",112238),
    13:("A",219622), 14:("A",200898), 15:("B",123737), 16:("B", 57197),
    17:("B", 93188), 18:("B",120653), 19:("A",203819), 20:("A",203742),
    21:("B",140167), 22:("B",119557), 23:("B",114533), 24:("A",203819),
    25:("B",128107), 26:("A",152513), 27:("A",204184), 28:("A",206302),
    29:("B", 93638), 30:("C", 42988), 31:("A",203750), 32:("A",203007),
    33:("A", 39690), 34:("A",158114), 35:("B",103681), 36:("A", 39910),
    37:("C", 39910), 38:("C", 39690), 39:("A",184109), 40:("A",155083),
    41:("A",196321), 42:("C", 39690), 43:("C", 41062), 44:("C", 39910),
    45:("B",118221),
}

# Columns the scaler was trained on (must match notebook NORM_COLS exactly)
SCALER_COLS = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'MarkDown']


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE LOADER
#  Each pipeline is a dict with: model, feature_names, scaler (or None), label
# ══════════════════════════════════════════════════════════════════════════════

def get_dir():
    """Return the directory where model files live (same folder as app.py)."""
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


@st.cache_resource(show_spinner=False)
def load_pipelines():
    """
    Attempt to load both pipelines.
    Returns dict: {
        'norm': {'model':..., 'scaler':..., 'features':..., 'label':..., 'error': None|str},
        'raw':  {'model':..., 'scaler': None, 'features':..., 'label':..., 'error': None|str},
    }
    Falls back gracefully: if new files not found, tries legacy model.pkl.
    """
    d = get_dir()
    pipelines = {}

    # ── Normalized pipeline ───────────────────────────────────────────────────
    norm_model_path  = os.path.join(d, 'model_norm.pkl')
    norm_scaler_path = os.path.join(d, 'scaler.pkl')
    norm_feat_path   = os.path.join(d, 'feature_names_norm.json')

    # Fallback to legacy model.pkl if new file not present
    if not os.path.exists(norm_model_path):
        norm_model_path = os.path.join(d, 'model.pkl')
        norm_feat_path  = os.path.join(d, 'feature_names.json')

    try:
        if not os.path.exists(norm_model_path):
            raise FileNotFoundError(f"model_norm.pkl (or model.pkl) not found in {d}")
        if not os.path.exists(norm_feat_path):
            raise FileNotFoundError(f"feature_names_norm.json (or feature_names.json) not found in {d}")
        if not os.path.exists(norm_scaler_path):
            raise FileNotFoundError(
                f"scaler.pkl not found in {d}.\n"
                "Re-run the notebook Section 15 to save scaler.pkl alongside model_norm.pkl.\n"
                "Without scaler.pkl the normalized model will receive raw values and produce wrong predictions."
            )

        model  = joblib.load(norm_model_path)
        scaler = joblib.load(norm_scaler_path)
        with open(norm_feat_path) as f:
            features = json.load(f)

        pipelines['norm'] = {
            'model': model, 'scaler': scaler, 'features': features,
            'label': 'Normalized RF', 'error': None,
            'model_file': os.path.basename(norm_model_path),
        }
    except Exception as e:
        pipelines['norm'] = {
            'model': None, 'scaler': None, 'features': None,
            'label': 'Normalized RF', 'error': str(e), 'model_file': '—',
        }

    # ── Unnormalized pipeline ─────────────────────────────────────────────────
    raw_model_path = os.path.join(d, 'model_raw.pkl')
    raw_feat_path  = os.path.join(d, 'feature_names_raw.json')

    try:
        if not os.path.exists(raw_model_path):
            raise FileNotFoundError(
                f"model_raw.pkl not found in {d}.\n"
                "Run the notebook to train and save the unnormalized model."
            )
        if not os.path.exists(raw_feat_path):
            raise FileNotFoundError(f"feature_names_raw.json not found in {d}.")

        model = joblib.load(raw_model_path)
        with open(raw_feat_path) as f:
            features = json.load(f)

        pipelines['raw'] = {
            'model': model, 'scaler': None, 'features': features,
            'label': 'Unnormalized RF', 'error': None, 'model_file': 'model_raw.pkl',
        }
    except Exception as e:
        pipelines['raw'] = {
            'model': None, 'scaler': None, 'features': None,
            'label': 'Unnormalized RF', 'error': str(e), 'model_file': '—',
        }

    # ── Linear Regression pipeline ───────────────────────────────────────────
    lr_model_path = os.path.join(d, 'linear_regression_model.pkl')
    lr_feat_path  = os.path.join(d, 'feature_names_linear.json')
    # LR reuses the same scaler.pkl as the Normalized RF pipeline

    try:
        if not os.path.exists(lr_model_path):
            raise FileNotFoundError(
                f"linear_regression_model.pkl not found in {d}.\n"
                "Run notebook Section 17 to train and save the Linear Regression model."
            )
        if not os.path.exists(lr_feat_path):
            raise FileNotFoundError(f"feature_names_linear.json not found in {d}.")
        if not os.path.exists(norm_scaler_path):
            raise FileNotFoundError(
                f"scaler.pkl not found in {d}.\n"
                "Linear Regression requires the same scaler.pkl used by the Normalized RF.\n"
                "Re-run notebook Section 15 to save it."
            )

        lr_mdl    = joblib.load(lr_model_path)
        lr_scaler = joblib.load(norm_scaler_path)
        with open(lr_feat_path) as f_lr:
            lr_features = json.load(f_lr)

        pipelines['lr'] = {
            'model': lr_mdl, 'scaler': lr_scaler, 'features': lr_features,
            'label': 'Linear Regression', 'error': None,
            'model_file': 'linear_regression_model.pkl',
        }
    except Exception as e:
        pipelines['lr'] = {
            'model': None, 'scaler': None, 'features': None,
            'label': 'Linear Regression', 'error': str(e), 'model_file': '—',
        }

    return pipelines


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE: build_input_row + predict
# ══════════════════════════════════════════════════════════════════════════════

def build_input_row(pipeline_key, pipeline, store, dept, store_type,
                    is_holiday_int, store_size, temperature,
                    fuel_price, cpi, unemployment, markdown_val):
    """
    Build the exact feature vector the model expects.

    Training used:
        pd.get_dummies(df, columns=['Type','Store','Dept'], drop_first=True)
    drop_first=True drops reference levels:
        Store → Store_1 is reference → dummies are Store_2 … Store_45
        Dept  → Dept_1  is reference → dummies are Dept_2  … Dept_99
        Type  → Type_A  is reference → dummies are Type_B, Type_C

    For the NORMALIZED pipeline:
        scaler.transform() is applied to SCALER_COLS before returning.
        This matches how X_train_norm was built in the notebook.

    For the UNNORMALIZED pipeline:
        raw values are passed directly — no transformation.
    """
    feature_names = pipeline['features']
    row = {f: 0 for f in feature_names}

    # ── Continuous / binary features (raw values) ─────────────────────────────
    raw_values = {
        'IsHoliday'   : is_holiday_int,
        'Temperature' : temperature,
        'Fuel_Price'  : fuel_price,
        'CPI'         : cpi,
        'Unemployment': unemployment,
        'MarkDown'    : markdown_val,
        'Size'        : store_size,
    }
    for key, val in raw_values.items():
        if key in row:
            row[key] = val

    # ── One-hot: Store ────────────────────────────────────────────────────────
    store_col = f'Store_{store}'
    if store_col in row:
        row[store_col] = 1
    # Store_1 is the reference level — no dummy column exists for it

    # ── One-hot: Dept ─────────────────────────────────────────────────────────
    dept_col = f'Dept_{dept}'
    if dept_col in row:
        row[dept_col] = 1
    # Dept_1 is the reference level

    # ── One-hot: Type ─────────────────────────────────────────────────────────
    type_col = f'Type_{store_type}'
    if type_col in row:
        row[type_col] = 1
    # Type_A is the reference level

    # ── Build DataFrame in exact column order ──────────────────────────────────
    input_df = pd.DataFrame([row])[feature_names]

    # ── NORMALIZED / LINEAR REGRESSION PIPELINE: apply scaler ───────────────
    # FIX: This was missing in v4 — raw values were sent to a normalized model.
    # Now scaler.transform() is called on exactly the columns it was fit on.
    if pipeline_key in ('norm', 'lr') and pipeline['scaler'] is not None:
        scaler = pipeline['scaler']
        cols_present = [c for c in SCALER_COLS if c in input_df.columns]
        input_df[cols_present] = scaler.transform(input_df[cols_present])

    return input_df


def make_prediction(pipeline_key, pipeline, store, dept, store_type,
                    is_holiday_int, store_size, temperature,
                    fuel_price, cpi, unemployment, markdown_val):
    """Run inference and return (prediction_float, debug_dict, error_str|None)."""
    try:
        input_df = build_input_row(
            pipeline_key, pipeline, store, dept, store_type,
            is_holiday_int, store_size, temperature,
            fuel_price, cpi, unemployment, markdown_val
        )

        prediction = pipeline['model'].predict(input_df)[0]

        # Debug info — values sent to the model
        scaler_applied = (pipeline_key == 'norm' and pipeline['scaler'] is not None)
        debug = {
            'n_features_sent'   : input_df.shape[1],
            'n_features_expected': pipeline['model'].n_features_in_,
            'scaler_applied'    : scaler_applied,
            'Temperature_sent'  : float(input_df['Temperature'].iloc[0]) if 'Temperature' in input_df else '?',
            'CPI_sent'          : float(input_df['CPI'].iloc[0]) if 'CPI' in input_df else '?',
        }

        if input_df.shape[1] != pipeline['model'].n_features_in_:
            return None, debug, (
                f"Feature count mismatch: sent {input_df.shape[1]} columns but "
                f"model expects {pipeline['model'].n_features_in_}. "
                "Re-run the notebook and re-save the model and feature_names JSON."
            )

        return float(prediction), debug, None

    except Exception as e:
        return None, {}, str(e)


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD PIPELINES
# ══════════════════════════════════════════════════════════════════════════════

pipelines = load_pipelines()
norm_ok = pipelines['norm']['error'] is None
raw_ok  = pipelines['raw']['error']  is None
lr_ok   = pipelines['lr']['error']   is None
any_ok  = norm_ok or raw_ok or lr_ok


# ══════════════════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
    <div class="hero-tag">🌲 Random Forest · 📐 Linear Regression · Triple Pipeline</div>
    <h1>Walmart Weekly<br>Sales Predictor</h1>
    <p>Choose a preprocessing pipeline, enter store conditions,
       and generate an instant ML prediction of weekly revenue.</p>
</div>
""", unsafe_allow_html=True)

# ── Pipeline status chips ─────────────────────────────────────────────────────
chips_html = '<div class="chip-row">'
if norm_ok:
    n_feat = len(pipelines['norm']['features'])
    chips_html += f'<span class="chip chip-norm">🔵 Normalized RF loaded · {n_feat} features</span>'
else:
    chips_html += '<span class="chip chip-warn">⚠️ Normalized RF unavailable</span>'
if raw_ok:
    n_feat = len(pipelines['raw']['features'])
    chips_html += f'<span class="chip chip-raw">🟠 Unnormalized RF loaded · {n_feat} features</span>'
else:
    chips_html += '<span class="chip chip-warn">⚠️ Unnormalized RF unavailable</span>'
if lr_ok:
    n_feat = len(pipelines['lr']['features'])
    chips_html += f'<span class="chip" style="border-color:#2e7d32;color:#a5d6a7;background:#0a1a0a;">🟢 Linear Regression loaded · {n_feat} features</span>'
else:
    chips_html += '<span class="chip chip-warn">⚠️ Linear Regression unavailable</span>'
chips_html += '</div>'
st.markdown(chips_html, unsafe_allow_html=True)

if not any_ok:
    st.markdown("""
    <div class="error-box">
    ❌ <strong>No models found.</strong><br><br>
    Run the notebook to generate these files in the same folder as app.py:<br>
    &nbsp;&nbsp;• <code>model_norm.pkl</code>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Section 15)<br>
    &nbsp;&nbsp;• <code>scaler.pkl</code>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Section 15)<br>
    &nbsp;&nbsp;• <code>feature_names_norm.json</code>&nbsp;(Section 15)<br>
    &nbsp;&nbsp;• <code>model_raw.pkl</code>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Section 15)<br>
    &nbsp;&nbsp;• <code>feature_names_raw.json</code>&nbsp;&nbsp;(Section 15)<br>
    &nbsp;&nbsp;• <code>linear_regression_model.pkl</code>&nbsp;(Section 17)<br>
    &nbsp;&nbsp;• <code>feature_names_linear.json</code>&nbsp;&nbsp;&nbsp;(Section 17)
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE SELECTOR
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-title">⚙️ Pipeline Selection</div>', unsafe_allow_html=True)

available_options = []
if norm_ok: available_options.append("🔵 Normalized RF  (Temperature, CPI, etc. scaled to [0,1])")
if raw_ok:  available_options.append("🟠 Unnormalized RF  (raw feature values — no scaling)")
if lr_ok:   available_options.append("🟢 Linear Regression  (normalized · coefficients-based)")

if not available_options:
    st.error("No working pipeline found.")
    st.stop()

pipeline_choice = st.radio(
    "Select prediction pipeline:",
    available_options,
    help="Normalized RF & Linear Regression use MinMax-scaled economic features. "
         "Unnormalized RF uses raw values — tree splits are scale-invariant. "
         "Linear Regression uses feature coefficients instead of feature importance.",
    horizontal=True,
)

if '🔵' in pipeline_choice:
    pipeline_key = 'norm'
elif '🟠' in pipeline_choice:
    pipeline_key = 'raw'
else:
    pipeline_key = 'lr'
pipeline = pipelines[pipeline_key]

if pipeline['error']:
    st.markdown(f'<div class="error-box">❌ {pipeline["error"]}</div>', unsafe_allow_html=True)
    st.stop()

# Pipeline info card
is_norm  = (pipeline_key == 'norm')
is_lr    = (pipeline_key == 'lr')
is_norm_or_lr = is_norm or is_lr

if is_lr:
    badge_color  = "#2e7d32"
    badge_label  = "Linear Regression"
    scaler_status = "✅ scaler.pkl loaded — will call scaler.transform() before predict"
elif is_norm:
    badge_color  = "#1565c0"
    badge_label  = "Normalized RF"
    scaler_status = "✅ scaler.pkl loaded — will call scaler.transform() before predict"
else:
    badge_color  = "#e65100"
    badge_label  = "Unnormalized RF"
    scaler_status = "ℹ️  No scaler — raw values sent directly to model"

st.markdown(f"""
<div class="badge" style="border-color:{badge_color};">
    <strong style="color:{'#90caf9' if is_norm else ('#a5d6a7' if is_lr else '#ffcc80')};">{badge_label} Pipeline</strong>
    &nbsp;·&nbsp; {scaler_status}
    &nbsp;·&nbsp; Model file: <code>{pipeline['model_file']}</code>
    &nbsp;·&nbsp; {len(pipeline['features'])} features
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

left, right = st.columns([1.1, 0.9], gap="large")

with left:
    # ── Store & Department ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🏪 Store Identity</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        store = st.number_input(
            "Store Number", min_value=1, max_value=45, value=1, step=1,
            help="Stores 1–45. Type and Size are auto-resolved from stores.csv metadata."
        )
    with col2:
        dept = st.number_input(
            "Department", min_value=1, max_value=99, value=5, step=1,
            help="Department 1–99. This is one of the strongest predictors."
        )

    # Auto-resolve Type and Size — exactly as training did via stores.csv join
    store_type, store_size = STORE_META[store]

    # Validate store number
    if store not in STORE_META:
        st.markdown('<div class="error-box">⚠️ Invalid store number. Choose 1–45.</div>',
                    unsafe_allow_html=True)
        st.stop()

    st.markdown(f"""
    <div class="badge">
        🔍 Store #{store} &nbsp;→&nbsp;
        <strong>Type {store_type}</strong> &nbsp;·&nbsp;
        <strong>{store_size:,} sq ft</strong>
        &nbsp;&nbsp;<span style="color:#3a5a8a;font-size:0.78rem;">auto-resolved · not user inputs</span>
    </div>
    """, unsafe_allow_html=True)

    is_holiday_choice = st.selectbox(
        "Holiday Week?",
        options=["No  (0)", "Yes  (1)"],
        index=0,
        help="Major US holidays: Super Bowl, Labour Day, Thanksgiving, Christmas."
    )
    is_holiday_int = 1 if "Yes" in is_holiday_choice else 0

    # ── Economic Indicators ───────────────────────────────────────────────────
    st.markdown('<div class="section-title">📈 Economic Indicators</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="warn-box" style="font-size:0.8rem; padding:0.8rem 1rem; margin-bottom:0.8rem;">
        ⚠️ <strong>Low-impact inputs for this model:</strong>
        Temperature, Fuel Price, CPI, and Unemployment have <em>weak individual effects</em>
        on predictions. Store identity (which store, which department) dominates.
        These inputs are still included for correctness — see the Insight panel on the right.
    </div>
    """, unsafe_allow_html=True)

    col5, col6 = st.columns(2)
    with col5:
        temperature = st.number_input(
            "Temperature (°F)", min_value=-30.0, max_value=120.0,
            value=65.2, step=0.1, format="%.1f",
            help="Weekly average temperature at the store location."
        )
    with col6:
        fuel_price = st.number_input(
            "Fuel Price ($/gal)", min_value=0.0, max_value=10.0,
            value=3.25, step=0.01, format="%.2f",
            help="Regional average fuel price for the week."
        )

    col7, col8 = st.columns(2)
    with col7:
        cpi = st.number_input(
            "CPI", min_value=100.0, max_value=300.0,
            value=211.0, step=0.1, format="%.1f",
            help="Consumer Price Index — reflects purchasing power."
        )
    with col8:
        unemployment = st.number_input(
            "Unemployment (%)", min_value=0.0, max_value=20.0,
            value=7.2, step=0.1, format="%.1f",
            help="Regional unemployment rate for the week."
        )

    # ── Promotions ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🏷️ Promotional Activity</div>', unsafe_allow_html=True)
    markdown_val = st.number_input(
        "MarkDown ($)", min_value=0.0, max_value=200000.0,
        value=3000.0, step=1000.0, format="%.2f",
        help=(
            "Sum of MarkDown1–5: total promotional discount spend this week. "
            "Training data range: $0 – ~$160,510. "
            "Only active from November 2011 onwards — ~50% of training rows have MarkDown=0. "
            "⚠️ Values below $10,000 have minimal model effect (< 6% of scaled range). "
            "Try $50,000–$160,000 to observe a meaningful prediction shift."
        )
    )


# ── RIGHT COLUMN: Summary + Predict + Insights ───────────────────────────────
with right:
    st.markdown('<div class="section-title">🔮 Prediction</div>', unsafe_allow_html=True)

    # Summary card
    st.markdown(f"""
    <div class="card">
        <div style="font-size:0.72rem;color:#5c9ee8;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.8rem;">
            Input Summary — {badge_label} Pipeline
        </div>
        <table style="width:100%;font-size:0.83rem;border-collapse:collapse;">
            <tr><td style="color:#8ba8cc;padding:0.2rem 0;">Store / Dept</td>
                <td style="color:#e8eaf0;text-align:right;font-weight:500;">#{store} / Dept {dept}</td></tr>
            <tr><td style="color:#8ba8cc;padding:0.2rem 0;">Store Type</td>
                <td style="color:#e8eaf0;text-align:right;font-weight:500;">
                    Type {store_type} <span style="color:#3a5a8a;font-size:0.74rem;">(auto)</span></td></tr>
            <tr><td style="color:#8ba8cc;padding:0.2rem 0;">Size</td>
                <td style="color:#e8eaf0;text-align:right;font-weight:500;">
                    {store_size:,} sq ft <span style="color:#3a5a8a;font-size:0.74rem;">(auto)</span></td></tr>
            <tr><td style="color:#8ba8cc;padding:0.2rem 0;">Holiday Week</td>
                <td style="color:#e8eaf0;text-align:right;font-weight:500;">{is_holiday_choice}</td></tr>
            <tr><td style="color:#8ba8cc;padding:0.2rem 0;">Temperature</td>
                <td style="color:#e8eaf0;text-align:right;font-weight:500;">{temperature}°F</td></tr>
            <tr><td style="color:#8ba8cc;padding:0.2rem 0;">Fuel Price</td>
                <td style="color:#e8eaf0;text-align:right;font-weight:500;">${fuel_price:.2f}</td></tr>
            <tr><td style="color:#8ba8cc;padding:0.2rem 0;">CPI</td>
                <td style="color:#e8eaf0;text-align:right;font-weight:500;">{cpi:.1f}</td></tr>
            <tr><td style="color:#8ba8cc;padding:0.2rem 0;">Unemployment</td>
                <td style="color:#e8eaf0;text-align:right;font-weight:500;">{unemployment:.1f}%</td></tr>
            <tr><td style="color:#8ba8cc;padding:0.2rem 0;">MarkDown</td>
                <td style="color:#e8eaf0;text-align:right;font-weight:500;">${markdown_val:,.0f}</td></tr>
            <tr><td style="color:#8ba8cc;padding:0.2rem 0;">Pipeline</td>
                <td style="color:{'#90caf9' if is_norm else ('#a5d6a7' if is_lr else '#ffcc80')};text-align:right;font-weight:600;">{badge_label}</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # Predict button
    predict_clicked = st.button("🔮  Predict Weekly Sales")

    if predict_clicked:
        prediction, debug, err = make_prediction(
            pipeline_key, pipeline,
            store, dept, store_type,
            is_holiday_int, store_size,
            temperature, fuel_price, cpi, unemployment, markdown_val
        )

        if err:
            st.markdown(f'<div class="error-box">❌ Prediction failed:<br>{err}</div>',
                        unsafe_allow_html=True)
        elif prediction is not None:
            holiday_flag = " 🎄" if is_holiday_int == 1 else ""
            if is_lr:
                box_class  = "result-norm"
                pipe_label = "Linear Regression"
            elif is_norm:
                box_class  = "result-norm"
                pipe_label = "Normalized RF"
            else:
                box_class  = "result-raw"
                pipe_label = "Unnormalized RF"

            st.markdown(f"""
            <div class="result-box {box_class}">
                <div class="result-label">Predicted Weekly Sales · {pipe_label}</div>
                <div class="result-value">${prediction:,.2f}</div>
                <div class="result-sub">
                    Store #{store} · Dept {dept} · Type {store_type}{holiday_flag}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Debug panel (expander)
            with st.expander("🔬 Debug: what was sent to the model"):
                # ── MarkDown scaling info ─────────────────────────────────────
                md_raw = markdown_val
                if is_norm_or_lr and pipeline['scaler'] is not None:
                    scaler_obj   = pipeline['scaler']
                    md_col_idx   = SCALER_COLS.index('MarkDown')
                    md_train_min = float(scaler_obj.data_min_[md_col_idx])
                    md_train_max = float(scaler_obj.data_max_[md_col_idx])
                    md_scaled    = float(np.clip(
                        (md_raw - md_train_min) / (md_train_max - md_train_min), 0.0, 1.0))
                    md_coverage  = md_raw / md_train_max * 100 if md_train_max > 0 else 0.0
                    md_debug_str = (
                        f"{md_scaled:.5f}  ← [0,1] scaled  "
                        f"(raw ${md_raw:,.0f} = {md_coverage:.1f}% of training max ${md_train_max:,.0f})"
                    )
                else:
                    md_train_max = None
                    md_coverage  = None
                    md_debug_str = f"{md_raw:,.2f}  ← raw value (no scaling applied)"

                st.markdown(f"""
<div class="debug-row">features sent    : {debug.get('n_features_sent','?')}</div>
<div class="debug-row">features expected : {debug.get('n_features_expected','?')}</div>
<div class="debug-row">scaler applied   : {debug.get('scaler_applied','?')}</div>
<div class="debug-row">Temperature sent : {debug.get('Temperature_sent','?'):.5f}{'  ← [0,1] scaled' if is_norm_or_lr else '  ← raw °F'}</div>
<div class="debug-row">CPI sent         : {debug.get('CPI_sent','?'):.5f}{'  ← [0,1] scaled' if is_norm_or_lr else '  ← raw value'}</div>
<div class="debug-row">MarkDown sent    : {md_debug_str}</div>
                """, unsafe_allow_html=True)

                # Warn if markdown coverage is very low (normalized / LR pipeline)
                if is_norm_or_lr and md_train_max and md_coverage is not None and md_coverage < 5:
                    st.warning(
                        f"⚠️ MarkDown **${md_raw:,.0f}** covers only **{md_coverage:.1f}%** of the "
                        f"training range (max ≈ ${md_train_max:,.0f}). After scaling this becomes "
                        f"**{md_scaled:.4f}** — a very small change for the model. "
                        f"Try values above **${md_train_max * 0.25:,.0f}** to see a noticeable shift."
                    )

                st.caption(
                    "Normalized RF & Linear Regression: Temperature, CPI, Unemployment, Fuel Price, and MarkDown "
                    "are scaled to [0,1] before prediction. Unnormalized RF: original values sent directly. "
                    "MarkDown only meaningfully affects predictions at higher values (>$20,000) due to "
                    "dataset sparsity — ~50% of training rows have MarkDown=0 (pre-Nov-2011 data)."
                )
    else:
        st.markdown("""
        <div style="text-align:center;padding:2.5rem 1rem;color:#2d4a6e;">
            <div style="font-size:3rem;margin-bottom:0.6rem;">🛒</div>
            <div style="font-family:'Syne',sans-serif;font-size:0.95rem;font-weight:600;color:#3a5a8a;">
                Fill in the inputs and click<br>Predict Weekly Sales
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  INSIGHTS PANEL
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="section-title">💡 Why Some Inputs Have Low Effect on Predictions</div>',
            unsafe_allow_html=True)

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
    <div class="card">
        <div style="font-size:0.78rem;color:#5c9ee8;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.8rem;">
            🏪 Store &amp; Dept dominate (high importance)
        </div>
        <p style="font-size:0.85rem;color:#c8d8ee;line-height:1.7;margin:0;">
            The model learned that a Walmart electronics department in a Type A store
            with 200,000+ sq ft will consistently sell far more than a small Type C store
            regardless of what temperature it is outside.<br><br>
            <strong>Store</strong> and <strong>Dept</strong> (one-hot encoded) account for
            the majority of feature importance because they capture the permanent structural
            baseline of each store/department combination.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown("""
    <div class="card">
        <div style="font-size:0.78rem;color:#ffd54f;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.8rem;">
            🌡️ Temperature, CPI, Fuel Price, Unemployment (low importance)
        </div>
        <p style="font-size:0.85rem;color:#c8d8ee;line-height:1.7;margin:0;">
            These macro-economic indicators have <em>real-world effects</em> on shopping
            behaviour, but at weekly store-level granularity their signal is drowned by
            store identity. The Random Forest down-weights them automatically — they still
            participate in splits, but rarely at the top of the tree where large impurity
            reductions happen.<br><br>
            <strong>Practical takeaway:</strong> Changing Temperature from 60°F to 80°F
            will move the prediction by very little. Changing Store from a Type C to a
            Type A can swing it by tens of thousands.
        </p>
    </div>
    """, unsafe_allow_html=True)

# MarkDown insight card — full-width
st.markdown("""
<div class="card" style="border-color:#1a3a5a;">
    <div style="font-size:0.78rem;color:#7aabdf;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.8rem;">
        🏷️ MarkDown — Why It Has Small Effect &amp; How to See It  <span style="color:#2d5a8a;font-size:0.7rem;">(v6 fix explained)</span>
    </div>
    <div style="display:flex;gap:1.5rem;flex-wrap:wrap;">
        <div style="flex:1;min-width:220px;">
            <p style="font-size:0.85rem;color:#c8d8ee;line-height:1.7;margin:0;">
                <strong style="color:#90caf9;">Why changes feel invisible at low values:</strong><br>
                The MinMaxScaler compresses all MarkDown values into [0, 1] using the training max
                of ~$160,510. A slider value of $3,000 becomes <code>0.0187</code> — only 1.87% of the
                scaled range. The model can barely distinguish that from zero.<br><br>
                <strong style="color:#90caf9;">To see a real shift:</strong> set MarkDown above
                <strong>$20,000</strong> (12% of range) or closer to <strong>$80,000–$160,000</strong>
                for a strong effect. Check the 🔬 Debug panel after predicting to see the exact scaled value sent.
            </p>
        </div>
        <div style="flex:1;min-width:220px;">
            <p style="font-size:0.85rem;color:#c8d8ee;line-height:1.7;margin:0;">
                <strong style="color:#90caf9;">Why MarkDown has low feature importance naturally:</strong><br>
                ~50% of training rows have MarkDown = 0 (all data before Nov 2011 had no markdowns).
                The Random Forest learns that zero is the dominant state — non-zero markdowns are a
                marginal signal on top of Store/Dept identity.<br><br>
                This is a <strong>data reality</strong>, not a code bug. The model is correct.
                MarkDown does have a real effect at high values — it just cannot compete with knowing
                which store and which department you are predicting.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <strong>🌲 Why Normalization Makes No Difference for Random Forest</strong><br><br>
    Random Forest splits data by finding the best <em>threshold</em> for one feature at a time.
    Thresholds use rank-order comparisons: "is Temperature &gt; X?" — the answer is identical
    whether X is 65.2°F (raw) or 0.43 (scaled), because the same rows end up in the left and right
    branches.<br><br>
    Distance-based models (KNN, SVM, linear regression) <strong>do</strong> need normalization
    because a raw Size of 200,000 would numerically overpower a raw Temperature of 65.
    Tree models are immune to this — that is why the Normalized and Unnormalized RF
    produce near-identical accuracy metrics.
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="info-box">
    <strong>📐 Linear Regression vs Random Forest — When to Use Each</strong><br><br>
    <strong style="color:#a5d6a7;">Linear Regression</strong> assumes a <em>linear relationship</em> between each feature and the target.
    Its predictions are fully explainable via feature coefficients: a positive coefficient on
    <code>Dept_38</code> means that department adds a fixed dollar amount to the prediction
    regardless of other feature values. However, it cannot capture interaction effects
    (e.g. "holiday discounts only matter in large stores") and tends to underfit complex retail data.
    <br><br>
    <strong style="color:#90caf9;">Random Forest</strong> captures non-linear interactions and is generally more accurate on tabular
    retail data — hence its higher R² on this dataset. It requires no scaling (tree splits are
    rank-order, not distance-based) and automatically handles feature interactions.
    The trade-off: its predictions are less directly interpretable than LR coefficients.
    <br><br>
    <strong>Practical check:</strong> compare the R² and RMSE values from notebook Section 17 —
    the gap between LR and RF quantifies exactly how much non-linearity is present in this dataset.
</div>
""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;font-size:0.74rem;color:#2d4a6e;padding:1.5rem 0 0.5rem;">
    Walmart Sales Predictor v7 · Triple Pipeline · Random Forest + Linear Regression · scikit-learn
</div>
""", unsafe_allow_html=True)
