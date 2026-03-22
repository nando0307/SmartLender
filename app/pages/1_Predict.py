"""1_Predict.py — Loan prediction with premium fintech UI."""
import streamlit as st
import pandas as pd
import sys, os, time

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

st.set_page_config(page_title="Predict — SmartLend", page_icon="🔮", layout="wide")

from app.components.theme import inject_css, hero, metric_card, section_header, badge, divider
from app.components.model_loader import load_module_models
from app.components.input_form import build_loan_form
from app.components.charts import risk_bar_chart

inject_css()

hero("🔮 Loan Prediction", "Enter a loan application and see all algorithms predict simultaneously.")

input_df = build_loan_form()

if input_df is not None:
    divider()

    models = load_module_models('module_a')

    if not models:
        st.warning("⚠️ No trained models found in `models/module_a/`. Run notebook 01 first.")
    else:
        section_header(f"Predictions from {len(models)} Algorithms")

        predictions = []
        for name, model in models.items():
            try:
                start = time.time()
                is_catboost = 'catboost' in name.lower()

                if is_catboost:
                    pred_df = input_df.copy()
                    for col in pred_df.select_dtypes(include=['object', 'category']).columns:
                        pred_df[col] = pred_df[col].fillna('Missing').astype(str)
                    pred = model.predict(pred_df)[0]
                    prob = model.predict_proba(pred_df)[0][1] if hasattr(model, 'predict_proba') else None
                else:
                    pred = model.predict(input_df)[0]
                    prob = model.predict_proba(input_df)[0][1] if hasattr(model, 'predict_proba') else None

                elapsed = time.time() - start
                risk_pct = (prob * 100) if prob is not None else (pred * 100)
                decision = 'Deny' if pred == 1 else 'Approve'

                predictions.append({
                    'algorithm': name,
                    'decision': decision,
                    'risk_pct': risk_pct,
                    'confidence': abs(risk_pct - 50) * 2,
                    'time_ms': elapsed * 1000,
                })
            except Exception as e:
                st.warning(f"⚠️ {name} failed: {e}")

        if predictions:
            pred_df = pd.DataFrame(predictions)

            approve_count = (pred_df['decision'] == 'Approve').sum()
            deny_count = (pred_df['decision'] == 'Deny').sum()
            avg_risk = pred_df['risk_pct'].mean()
            consensus = "APPROVE" if approve_count > deny_count else "DENY"

            # Metric cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(metric_card("Approve", str(approve_count), "emerald"), unsafe_allow_html=True)
            with col2:
                st.markdown(metric_card("Deny", str(deny_count), "red"), unsafe_allow_html=True)
            with col3:
                st.markdown(metric_card("Avg Risk", f"{avg_risk:.1f}%", "gold"), unsafe_allow_html=True)
            with col4:
                variant = "emerald" if consensus == "APPROVE" else "red"
                st.markdown(metric_card("Consensus", consensus, variant), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Results table
            display_df = pred_df[['algorithm', 'decision', 'risk_pct', 'confidence', 'time_ms']].copy()
            display_df.columns = ['Algorithm', 'Decision', 'Risk %', 'Confidence %', 'Time (ms)']
            st.dataframe(
                display_df.style.format({'Risk %': '{:.1f}', 'Confidence %': '{:.1f}', 'Time (ms)': '{:.1f}'}),
                use_container_width=True,
                hide_index=True,
            )

            # Risk chart
            st.plotly_chart(risk_bar_chart(pred_df), use_container_width=True)
