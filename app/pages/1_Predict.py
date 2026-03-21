"""1_Predict.py — Single loan prediction page."""
import streamlit as st
import pandas as pd
import sys
import os
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from app.components.model_loader import load_module_models
from app.components.input_form import build_loan_form
from app.components.charts import risk_bar_chart

st.set_page_config(page_title="Predict — SmartLend", page_icon="🔮", layout="wide")
st.title("🔮 Loan Prediction")
st.markdown("Enter a loan application and see all 8 models predict simultaneously.")

# Build the input form
input_df = build_loan_form()

if input_df is not None:
    st.markdown("---")

    # Load models
    models = load_module_models('module_a')

    if not models:
        st.warning("⚠️ No trained models found in `models/module_a/`. Run notebook 01 first.")
    else:
        st.subheader(f"📊 Results from {len(models)} Algorithms")

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

            # Summary cards
            approve_count = (pred_df['decision'] == 'Approve').sum()
            deny_count = (pred_df['decision'] == 'Deny').sum()
            avg_risk = pred_df['risk_pct'].mean()

            col1, col2, col3 = st.columns(3)
            col1.metric("✅ Approve", approve_count)
            col2.metric("❌ Deny", deny_count)
            col3.metric("📈 Avg Risk", f"{avg_risk:.1f}%")

            # Results table
            st.dataframe(
                pred_df[['algorithm', 'decision', 'risk_pct', 'confidence', 'time_ms']]
                .style.applymap(
                    lambda v: 'color: green' if v == 'Approve' else ('color: red' if v == 'Deny' else ''),
                    subset=['decision']
                )
                .format({'risk_pct': '{:.1f}%', 'confidence': '{:.1f}%', 'time_ms': '{:.1f}ms'}),
                use_container_width=True,
            )

            # Risk bar chart
            st.plotly_chart(risk_bar_chart(pred_df), use_container_width=True)

    # Batch upload
    with st.expander("📁 Batch Upload"):
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Uploaded {len(batch_df)} loans")

            if models:
                batch_results = []
                for name, model in models.items():
                    try:
                        start = time.time()
                        is_catboost = 'catboost' in name.lower()
                        if is_catboost:
                            pred_batch = batch_df.copy()
                            for col in pred_batch.select_dtypes(include=['object', 'category']).columns:
                                pred_batch[col] = pred_batch[col].fillna('Missing').astype(str)
                            preds = model.predict(pred_batch)
                        else:
                            preds = model.predict(batch_df)
                        elapsed = time.time() - start
                        defaults = preds.sum()
                        batch_results.append({
                            'algorithm': name,
                            'defaults': int(defaults),
                            'default_rate': f"{defaults/len(preds)*100:.1f}%",
                            'time_s': f"{elapsed:.2f}s",
                        })
                    except Exception as e:
                        st.warning(f"⚠️ {name} failed on batch: {e}")

                if batch_results:
                    st.dataframe(pd.DataFrame(batch_results), use_container_width=True)
