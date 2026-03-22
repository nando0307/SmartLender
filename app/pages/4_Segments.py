"""4_Segments.py — Clustering visualization with fintech design."""
import streamlit as st
import pandas as pd
import numpy as np
import os, sys
import joblib

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

st.set_page_config(page_title="Segments — SmartLend", page_icon="🎯", layout="wide")

from app.components.theme import inject_css, hero, section_header, metric_card, divider
from app.components.charts import cluster_scatter

inject_css()

hero("🎯 Customer Segments", "Explore borrower segments discovered by K-Means clustering.")

cluster_path = os.path.join(project_root, 'data', 'processed', 'cluster_data.parquet')
profiles_path = os.path.join(project_root, 'results', 'comparison_tables', 'module_c_profiles.csv')
scaler_path = os.path.join(project_root, 'models', 'module_c', 'scaler.joblib')

if not os.path.exists(cluster_path):
    st.warning("⚠️ Cluster data not found. Run notebook 03 first.")
    st.stop()

cluster_df = pd.read_parquet(cluster_path)
cluster_df['cluster'] = cluster_df['cluster'].astype(str)

divider()

# Cluster distribution metrics
dist = cluster_df['cluster'].value_counts().sort_index()
n_clusters = len(dist)

cols = st.columns(min(n_clusters, 6))
cluster_colors = ['blue', 'emerald', 'gold', 'red', 'blue', 'emerald']
for i, (cluster_id, count) in enumerate(dist.items()):
    with cols[i % len(cols)]:
        pct = count / len(cluster_df) * 100
        st.markdown(metric_card(
            f"Cluster {cluster_id}",
            f"{count:,}",
            cluster_colors[i % len(cluster_colors)]
        ), unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center; color:#64748b; font-size:0.8rem;'>{pct:.1f}%</div>",
                    unsafe_allow_html=True)

divider()

# Scatter plot
section_header("PCA Projection")
st.plotly_chart(cluster_scatter(cluster_df, 'pca_1', 'pca_2', 'cluster'), use_container_width=True)

# Profiles
if os.path.exists(profiles_path):
    divider()
    section_header("Cluster Profiles")
    st.markdown("<p style='color:#94a3b8;'>Mean feature values per cluster segment:</p>", unsafe_allow_html=True)
    profiles = pd.read_csv(profiles_path)
    st.dataframe(
        profiles.style.format('{:.2f}', subset=profiles.columns[1:]),
        use_container_width=True, hide_index=True,
    )

# Re-cluster
divider()
section_header("Experiment: Adjust Clusters")

if os.path.exists(scaler_path):
    new_k = st.slider("Number of Clusters", min_value=2, max_value=10, value=5)

    if st.button("🔄 Re-Cluster", type="primary"):
        with st.spinner("Re-clustering..."):
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA

            full_path = os.path.join(project_root, 'data', 'processed', 'full_cleaned.parquet')
            if os.path.exists(full_path):
                full_df = pd.read_parquet(full_path)
                from src.config import NUMERICAL_FEATURES
                num_cols = [c for c in NUMERICAL_FEATURES if c in full_df.columns]
                X = full_df[num_cols].dropna()

                scaler = joblib.load(scaler_path)
                X_scaled = scaler.transform(X)

                km = KMeans(n_clusters=new_k, random_state=42, n_init=10)
                labels = km.fit_predict(X_scaled)

                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)

                new_df = pd.DataFrame({
                    'pca_1': X_pca[:, 0], 'pca_2': X_pca[:, 1],
                    'cluster': labels.astype(str),
                })

                st.plotly_chart(cluster_scatter(new_df, 'pca_1', 'pca_2', 'cluster'), use_container_width=True)

                from sklearn.metrics import silhouette_score
                sil = silhouette_score(X_scaled, labels, sample_size=10000)
                st.markdown(metric_card("Silhouette Score", f"{sil:.3f}", "emerald"), unsafe_allow_html=True)
            else:
                st.warning("⚠️ Full cleaned data not found.")
else:
    st.info("Run notebook 03 to enable re-clustering.")
