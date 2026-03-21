"""Evaluation metrics for classification and regression."""
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, log_loss,
    mean_squared_error, mean_absolute_error, r2_score,
)
import numpy as np


def classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Standard classification metrics."""
    scores = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
    }
    if y_prob is not None:
        scores['auc_roc'] = roc_auc_score(y_true, y_prob)
        scores['log_loss'] = log_loss(y_true, y_prob)
    return scores


def regression_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Standard regression metrics."""
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
    }
