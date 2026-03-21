"""SmartLend: The Algorithm Arena — Main Dashboard"""
import streamlit as st

st.set_page_config(
    page_title="SmartLend Arena",
    page_icon="🏟️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("SmartLend: The Algorithm Arena")
st.markdown(
    "Compare 8 ML algorithms on real lending data. "
    "Enter a loan application or explore algorithm performance across 5 prediction tasks."
)

st.markdown("---")
st.markdown("### Select a page from the sidebar to get started:")
st.markdown("""
- **Predict:** Enter a loan application and see all 8 models predict simultaneously
- **Arena:** Compare algorithm performance across all modules
- **Explainer:** See SHAP explanations for why each model made its decision
- **Segments:** Explore customer segments discovered by clustering
""")
