"""2_Arena.py — Algorithm comparison dashboard."""
import streamlit as st
import pandas as pd
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from app.components.charts import comparison_bar_chart, timing_chart

st.set_page_config(page_title="Arena — SmartLend", page_icon="🏟️", layout="wide")
st.title("🏟️ Algorithm Arena")
st.markdown("Compare algorithm performance across all prediction modules.")

# Module selection
results_dir = os.path.join(project_root, 'results', 'comparison_tables')

module_options = {
    'module_a': 'Module A — Default Classification',
    'module_b': 'Module B — Interest Rate Regression',
    'module_d': 'Module D — Loss Amount Regression',
    'module_e': 'Module E — Temporal Default Prediction',
}

available_modules = {}
for key, label in module_options.items():
    filepath = os.path.join(results_dir, f'{key}.csv')
    if os.path.exists(filepath):
        available_modules[key] = label

if not available_modules:
    st.warning("⚠️ No comparison results found. Run the notebooks (01-05) first to generate results.")
    st.stop()

selected_module = st.selectbox(
    "Select Module",
    list(available_modules.keys()),
    format_func=lambda x: available_modules[x],
)

# Load comparison data
filepath = os.path.join(results_dir, f'{selected_module}.csv')
df = pd.read_csv(filepath)

# Determine primary metric
if selected_module in ['module_a', 'module_e']:
    primary_metric = 'auc_roc'
    sort_ascending = False
elif selected_module == 'module_d':
    primary_metric = 'rmse'
    sort_ascending = True
else:
    primary_metric = 'r2'
    sort_ascending = False

df_sorted = df.sort_values(primary_metric, ascending=sort_ascending)

# Winner card
winner = df_sorted.iloc[0] if not sort_ascending else df_sorted.iloc[0]
st.success(f"🏆 **Winner: {winner['algorithm']}** — {primary_metric}: {winner[primary_metric]:.4f}")

# Comparison table
st.subheader("📊 Full Comparison Table")
st.dataframe(
    df_sorted.style.highlight_max(
        subset=[c for c in df_sorted.columns if c not in ['algorithm', 'train_time_s', 'infer_time_s', 'log_loss', 'rmse', 'mae', 'mape']],
        color='#2ecc71', axis=0
    ).highlight_min(
        subset=[c for c in ['log_loss', 'rmse', 'mae', 'mape'] if c in df_sorted.columns],
        color='#2ecc71', axis=0
    ).format({c: '{:.4f}' for c in df_sorted.columns if c != 'algorithm'}),
    use_container_width=True,
)

# Charts
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(
        comparison_bar_chart(df_sorted, primary_metric, f'{primary_metric.upper()} by Algorithm'),
        use_container_width=True,
    )
with col2:
    st.plotly_chart(timing_chart(df_sorted), use_container_width=True)

# Additional metrics
if selected_module in ['module_a', 'module_e']:
    metric_options = ['auc_roc', 'accuracy', 'f1', 'precision', 'recall']
else:
    metric_options = ['r2', 'rmse', 'mae', 'mape']

available_metrics = [m for m in metric_options if m in df.columns]
if len(available_metrics) > 1:
    st.subheader("📈 Explore Other Metrics")
    selected_metric = st.selectbox("Select Metric", available_metrics)
    st.plotly_chart(
        comparison_bar_chart(df_sorted, selected_metric),
        use_container_width=True,
    )
