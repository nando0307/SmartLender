"""4_Segments.py — Clustering visualization page."""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import joblib

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from app.components.charts import cluster_scatter

st.set_page_config(page_title="Segments — SmartLend", page_icon="🎯", layout="wide")
st.title("🎯 Customer Segments")
st.markdown("Explore borrower segments discovered by K-Means clustering.")

# Load pre-computed cluster data
cluster_path = os.path.join(project_root, 'data', 'processed', 'cluster_data.parquet')
profiles_path = os.path.join(project_root, 'results', 'comparison_tables', 'module_c_profiles.csv')
kmeans_path = os.path.join(project_root, 'models', 'module_c', 'kmeans.joblib')
scaler_path = os.path.join(project_root, 'models', 'module_c', 'scaler.joblib')

if not os.path.exists(cluster_path):
    st.warning("⚠️ Cluster data not found. Run notebook 03 (Module C) first.")
    st.stop()

cluster_df = pd.read_parquet(cluster_path)
cluster_df['cluster'] = cluster_df['cluster'].astype(str)

# Interactive scatter plot
st.subheader("📍 Cluster Visualization (PCA)")
st.plotly_chart(
    cluster_scatter(cluster_df, 'pca_1', 'pca_2', 'cluster'),
    use_container_width=True,
)

# Cluster distribution
st.subheader("📊 Cluster Distribution")
dist = cluster_df['cluster'].value_counts().sort_index()
col1, col2 = st.columns([1, 2])
with col1:
    st.dataframe(
        pd.DataFrame({'Cluster': dist.index, 'Count': dist.values, 'Pct': (dist.values / len(cluster_df) * 100).round(1)}),
        use_container_width=True,
    )

# Cluster profiles
if os.path.exists(profiles_path):
    st.subheader("👤 Cluster Profiles")
    st.markdown("Mean feature values per cluster:")
    profiles = pd.read_csv(profiles_path)
    st.dataframe(profiles.style.format('{:.2f}', subset=profiles.columns[1:]), use_container_width=True)

# Re-cluster with different k
st.markdown("---")
st.subheader("🔄 Experiment: Adjust Clusters")

if os.path.exists(scaler_path):
    new_k = st.slider("Number of Clusters", min_value=2, max_value=10, value=5)

    if st.button("Re-Cluster"):
        with st.spinner("Re-clustering..."):
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA

            # Load full numerical data
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

                new_cluster_df = pd.DataFrame({
                    'pca_1': X_pca[:, 0],
                    'pca_2': X_pca[:, 1],
                    'cluster': labels.astype(str),
                })

                st.plotly_chart(
                    cluster_scatter(new_cluster_df, 'pca_1', 'pca_2', 'cluster'),
                    use_container_width=True,
                )

                # Show new profiles
                profiled = X.copy()
                profiled['cluster'] = labels
                new_profiles = profiled.groupby('cluster').mean().round(2)
                st.dataframe(new_profiles, use_container_width=True)

                from sklearn.metrics import silhouette_score
                sil = silhouette_score(X_scaled, labels, sample_size=10000)
                st.metric("Silhouette Score", f"{sil:.3f}")
            else:
                st.warning("⚠️ Full cleaned data not found. Run notebook 00 first.")
else:
    st.info("Run notebook 03 to enable re-clustering.")
