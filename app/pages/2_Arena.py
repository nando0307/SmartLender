"""2_Arena.py — Algorithm comparison with fintech design."""
import streamlit as st
import pandas as pd
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

st.set_page_config(page_title="Arena — SmartLend", page_icon="🏟️", layout="wide")

from app.components.theme import inject_css, hero, section_header, winner_banner, metric_card, divider
from app.components.charts import comparison_bar_chart, timing_chart

inject_css()

hero("🏟️ The Algorithm Arena", "Head-to-head comparison of all algorithms across every prediction module.")

results_dir = os.path.join(project_root, 'results', 'comparison_tables')

module_options = {
    'module_a': ('Module A', 'Default Classification', 'auc_roc', False),
    'module_b': ('Module B', 'Interest Rate Regression', 'r2', False),
    'module_d': ('Module D', 'Loss Amount Regression', 'rmse', True),
    'module_e': ('Module E', 'Temporal Default Prediction', 'auc_roc', False),
}

available = {}
for key, (short, desc, metric, asc) in module_options.items():
    if os.path.exists(os.path.join(results_dir, f'{key}.csv')):
        available[key] = (short, desc, metric, asc)

if not available:
    st.warning("⚠️ No results found. Run the notebooks first.")
    st.stop()

# Module selector
selected = st.selectbox(
    "Select Module",
    list(available.keys()),
    format_func=lambda x: f"{available[x][0]} — {available[x][1]}",
)

short_name, desc, primary_metric, sort_asc = available[selected]
df = pd.read_csv(os.path.join(results_dir, f'{selected}.csv'))
df_sorted = df.sort_values(primary_metric, ascending=sort_asc)

divider()

# Winner
winner = df_sorted.iloc[0]
winner_banner(winner['algorithm'], primary_metric.upper().replace('_', '-'), f"{winner[primary_metric]:.4f}")

# Top 3 metric cards
section_header("Top 3 Performers")
top3 = df_sorted.head(3)
cols = st.columns(3)
medals = ['🥇', '🥈', '🥉']
for i, (_, row) in enumerate(top3.iterrows()):
    with cols[i]:
        st.markdown(metric_card(
            f"{medals[i]} {row['algorithm']}",
            f"{row[primary_metric]:.4f}",
            ["gold", "blue", "emerald"][i]
        ), unsafe_allow_html=True)

divider()

# Full comparison table
section_header("Full Comparison Table")
st.dataframe(
    df_sorted.style.format({c: '{:.4f}' for c in df_sorted.columns if c != 'algorithm'}),
    use_container_width=True,
    hide_index=True,
)

# Charts side by side
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(
        comparison_bar_chart(df_sorted, primary_metric, f'{primary_metric.upper().replace("_", " ")} by Algorithm'),
        use_container_width=True,
    )
with col2:
    st.plotly_chart(timing_chart(df_sorted), use_container_width=True)

# Explore other metrics
if selected in ['module_a', 'module_e']:
    metric_options = ['auc_roc', 'accuracy', 'f1', 'precision', 'recall']
else:
    metric_options = ['r2', 'rmse', 'mae', 'mape']

available_metrics = [m for m in metric_options if m in df.columns]
if len(available_metrics) > 1:
    divider()
    section_header("Explore Other Metrics")
    selected_metric = st.selectbox("Select Metric", available_metrics)
    st.plotly_chart(comparison_bar_chart(df_sorted, selected_metric), use_container_width=True)
