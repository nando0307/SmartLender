"""SmartLend: The Algorithm Arena — Main Dashboard"""
import streamlit as st
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

st.set_page_config(
    page_title="SmartLend Arena",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.components.theme import inject_css, hero, metric_card, divider

inject_css()

# Hero Section
hero(
    "SmartLend: The Algorithm Arena",
    "Compare 8 classical ML algorithms across 5 prediction tasks on real Lending Club data. "
    "Enter a loan application and watch algorithms compete in real-time."
)

# Tech tags
st.markdown("""
<div style="margin-top: -16px; margin-bottom: 32px;">
    <span class="tech-tag">scikit-learn</span>
    <span class="tech-tag">XGBoost</span>
    <span class="tech-tag">LightGBM</span>
    <span class="tech-tag">CatBoost</span>
    <span class="tech-tag">SHAP</span>
    <span class="tech-tag">Optuna</span>
    <span class="tech-tag">MLflow</span>
    <span class="tech-tag">Streamlit</span>
</div>
""", unsafe_allow_html=True)

# Summary stats
results_dir = os.path.join(project_root, 'results', 'comparison_tables')
models_dir = os.path.join(project_root, 'models')

n_models = 0
n_modules = 0
for d in os.listdir(models_dir):
    mod_path = os.path.join(models_dir, d)
    if os.path.isdir(mod_path) and not d.startswith('.'):
        count = len([f for f in os.listdir(mod_path) if f.endswith('.joblib')])
        if count > 0:
            n_modules += 1
            n_models += count

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(metric_card("Algorithms", "8", "blue"), unsafe_allow_html=True)
with col2:
    st.markdown(metric_card("Modules", str(n_modules), "emerald"), unsafe_allow_html=True)
with col3:
    st.markdown(metric_card("Trained Models", str(n_models), "gold"), unsafe_allow_html=True)
with col4:
    st.markdown(metric_card("Explanations", "SHAP", "blue"), unsafe_allow_html=True)

divider()

# Navigation Cards
st.markdown('<div class="section-header">Explore the Arena</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-icon">🔮</div>
        <div class="nav-title">Predict</div>
        <div class="nav-desc">Enter a loan application and see all 8 models predict simultaneously with risk scores.</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-icon">🏟️</div>
        <div class="nav-title">Arena</div>
        <div class="nav-desc">Compare algorithm performance across classification, regression, and temporal tasks.</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-icon">🔍</div>
        <div class="nav-title">Explainer</div>
        <div class="nav-desc">Understand why each model makes its predictions with interactive SHAP analysis.</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-icon">🎯</div>
        <div class="nav-title">Segments</div>
        <div class="nav-desc">Explore borrower clusters discovered by unsupervised learning with PCA visualization.</div>
    </div>
    """, unsafe_allow_html=True)

divider()

# Verdict preview
verdict_path = os.path.join(project_root, 'results', 'verdict.md')
if os.path.exists(verdict_path):
    st.markdown('<div class="section-header">Algorithm Arena Verdict</div>', unsafe_allow_html=True)
    with open(verdict_path) as f:
        verdict = f.read()
    with st.expander("📄 View Full Verdict", expanded=False):
        st.markdown(verdict)

# Footer
st.markdown("""
<div style="text-align: center; padding: 40px 0 20px 0; color: #64748b; font-size: 0.8rem; font-family: 'Inter', sans-serif;">
    SmartLend • Built by <strong>Nando (Do Le)</strong> • USF Computer Science, AI/ML
</div>
""", unsafe_allow_html=True)
