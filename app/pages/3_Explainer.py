"""3_Explainer.py — SHAP explanations page."""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import shap
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from app.components.model_loader import load_module_models, get_available_modules

st.set_page_config(page_title="Explainer — SmartLend", page_icon="🔍", layout="wide")
st.title("🔍 SHAP Explainer")
st.markdown("Understand **why** each model makes its predictions using SHAP values.")

# Module and model selection
available = get_available_modules()
if not available:
    st.warning("⚠️ No trained models found. Run the notebooks first.")
    st.stop()

selected_module = st.selectbox("Select Module", available)
models = load_module_models(selected_module)

if not models:
    st.warning(f"⚠️ No models found for {selected_module}.")
    st.stop()

selected_algorithm = st.selectbox("Select Algorithm", list(models.keys()))
model = models[selected_algorithm]

# Load sample test data
test_path = os.path.join(project_root, 'data', 'processed', 'test.parquet')
if not os.path.exists(test_path):
    st.warning("⚠️ Test data not found at `data/processed/test.parquet`. Run notebook 00 first.")
    st.stop()

test_df = pd.read_parquet(test_path)

# Determine feature columns
from src.config import ALL_FEATURES
feature_cols = [c for c in ALL_FEATURES if c in test_df.columns]
X_test = test_df[feature_cols]

st.info(f"Using {len(X_test)} test samples. SHAP computation may take a moment...")

# Compute SHAP values
max_samples = st.slider("Max samples for SHAP", min_value=50, max_value=500, value=100, step=50)

with st.spinner("Computing SHAP values..."):
    try:
        is_catboost = 'catboost' in selected_algorithm.lower()

        if is_catboost:
            X_sample = X_test.head(max_samples).copy()
            for col in X_sample.select_dtypes(include=['object', 'category']).columns:
                X_sample[col] = X_sample[col].fillna('Missing').astype(str)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            feature_names = list(X_sample.columns)
            X_display = X_sample
        elif hasattr(model, 'named_steps'):
            # Pipeline model — extract preprocessor and inner model
            preprocessor = model.named_steps['preprocessor']
            inner_model = model.named_steps['model']
            X_sample = X_test.head(max_samples)
            X_processed = preprocessor.transform(X_sample)
            feature_names = list(preprocessor.get_feature_names_out())

            try:
                explainer = shap.TreeExplainer(inner_model)
                shap_values = explainer.shap_values(X_processed)
            except Exception:
                background = X_processed[:50]
                explainer = shap.KernelExplainer(inner_model.predict_proba, background)
                shap_values = explainer.shap_values(X_processed)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]

            X_display = pd.DataFrame(X_processed, columns=feature_names)
        else:
            st.error("Unsupported model format.")
            st.stop()

        # Handle binary classification SHAP output
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]  # Class 1 (default)

        st.success("✅ SHAP values computed successfully!")

        # Global importance
        st.subheader("🌍 Global Feature Importance")
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_display, feature_names=feature_names, max_display=15, show=False)
        st.pyplot(fig)
        plt.close()

        # Single prediction explanation
        st.subheader("🎯 Explain a Single Prediction")
        sample_idx = st.number_input("Sample Index", min_value=0, max_value=max_samples - 1, value=0)

        fig2, ax2 = plt.subplots(figsize=(12, 5))
        expected = explainer.expected_value
        if not np.isscalar(expected):
            expected = expected[1] if len(expected) > 1 else expected[0]

        if isinstance(X_display, pd.DataFrame):
            data_point = X_display.iloc[sample_idx].values
        else:
            data_point = X_display[sample_idx]

        explanation = shap.Explanation(
            values=shap_values[sample_idx],
            base_values=expected,
            data=data_point,
            feature_names=feature_names,
        )
        shap.plots.waterfall(explanation, show=False)
        st.pyplot(fig2)
        plt.close()

        # Show raw feature values
        st.subheader("📋 Raw Feature Values")
        if isinstance(X_display, pd.DataFrame):
            st.dataframe(X_display.iloc[[sample_idx]], use_container_width=True)
        else:
            st.dataframe(
                pd.DataFrame([X_display[sample_idx]], columns=feature_names),
                use_container_width=True,
            )

    except Exception as e:
        st.error(f"❌ Error computing SHAP: {e}")
        st.exception(e)
