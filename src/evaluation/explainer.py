"""SHAP explainability wrappers."""
import shap
import matplotlib.pyplot as plt
import numpy as np


def get_shap_values(model, X, feature_names=None, max_samples=500):
    """
    Compute SHAP values for a model.

    Handles both tree-based models (TreeExplainer - fast) and
    linear/other models (KernelExplainer - slow, uses sampling).

    Returns:
        explainer, shap_values
        (The sampled X is stored as explainer.X_sample_ for optional use.)
    """
    X_sample = X[:max_samples] if len(X) > max_samples else X

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample, check_additivity=False)
    except Exception:
        # Fallback for non-tree models (e.g., Logistic Regression pipeline)
        explainer = shap.KernelExplainer(model.predict_proba, X_sample[:100])
        shap_values = explainer.shap_values(X_sample)

    # Normalize output: for binary classification, TreeExplainer may return
    # a list of [class_0_shap, class_1_shap] — take class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Newer SHAP versions may return an Explanation object
    if hasattr(shap_values, 'values'):
        shap_values = shap_values.values

    # Store sampled X on the explainer so callers can use it if needed
    explainer.X_sample_ = X_sample

    return explainer, shap_values


def plot_global_importance(shap_values, X, feature_names, title="Feature importance", save_path=None):
    """SHAP summary plot - global feature importance."""
    # Ensure row counts match (shap_values may be computed on a sample)
    n = shap_values.shape[0]
    X_plot = X[:n] if hasattr(X, '__len__') and len(X) > n else X

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_plot, feature_names=feature_names, max_display=15, show=False)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_single_prediction(shap_values, explainer, X, idx, feature_names, save_path=None):
    """SHAP waterfall plot - explain ONE prediction."""
    plt.figure(figsize=(12, 5))
    explanation = shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[1],
        data=X[idx] if hasattr(X, '__getitem__') else X.iloc[idx],
        feature_names=feature_names,
    )
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
