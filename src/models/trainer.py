"""Universal training and evaluation pipeline."""
import os
import time
import joblib
import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from src.config import MODELS_DIR


def train_and_evaluate(
    name: str,
    model,
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metrics_fn: callable,
    cat_feature_indices: list = None,
    module_name: str = 'module_a',
    save: bool = True,
) -> dict:
    """
    Train a single model, evaluate it, log to MLflow, optionally save.

    IMPORTANT: CatBoost handles categoricals natively. For CatBoost:
    - Do NOT wrap in a preprocessing pipeline
    - DO pass cat_features to fit()
    - Fill NaN in categoricals with "Missing" string before passing to CatBoost

    For all other models:
    - Wrap in Pipeline with the preprocessor
    - Preprocessor handles imputation + encoding

    Args:
        name: Algorithm name (e.g., "XGBoost")
        model: sklearn-compatible model instance
        preprocessor: fitted ColumnTransformer (ignored for CatBoost)
        X_train, y_train: Training data
        X_test, y_test: Test data
        metrics_fn: Function(y_true, y_pred, y_prob=None) -> dict of metric scores
        cat_feature_indices: Column indices of categoricals (for CatBoost only)
        module_name: Subfolder name for saving (e.g., "module_a")
        save: Whether to save the trained model to disk

    Returns:
        dict with algorithm name, all metrics, train/infer times, and model object
    """
    is_catboost = 'CatBoost' in name

    if is_catboost:
        # CatBoost: fill categorical NaN with string, pass raw data
        X_train_cb = X_train.copy()
        X_test_cb = X_test.copy()
        for col in X_train_cb.select_dtypes(include=['object', 'category']).columns:
            X_train_cb[col] = X_train_cb[col].fillna('Missing').astype(str)
            X_test_cb[col] = X_test_cb[col].fillna('Missing').astype(str)

        # Train
        start = time.time()
        model.fit(X_train_cb, y_train, cat_features=cat_feature_indices)
        train_time = time.time() - start

        # Predict
        start = time.time()
        y_pred = model.predict(X_test_cb)
        y_prob = None
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test_cb)[:, 1]
        infer_time = time.time() - start

        pipeline_or_model = model  # Save raw CatBoost model
    else:
        # All other models: wrap in preprocessing pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model),
        ])

        # Train
        start = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start

        # Predict
        start = time.time()
        y_pred = pipeline.predict(X_test)
        y_prob = None
        if hasattr(pipeline, 'predict_proba'):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        infer_time = time.time() - start

        pipeline_or_model = pipeline

    # Evaluate
    if y_prob is not None:
        scores = metrics_fn(y_test, y_pred, y_prob)
    else:
        scores = metrics_fn(y_test, y_pred)

    scores['train_time_s'] = round(train_time, 3)
    scores['infer_time_s'] = round(infer_time, 4)

    # Log to MLflow
    with mlflow.start_run(run_name=f"{module_name}/{name}", nested=True):
        mlflow.log_params({'algorithm': name, 'module': module_name})
        mlflow.log_metrics(scores)

    # Save model
    if save:
        save_dir = os.path.join(MODELS_DIR, module_name)
        os.makedirs(save_dir, exist_ok=True)
        save_name = name.lower().replace(' ', '_') + '.joblib'
        joblib.dump(pipeline_or_model, os.path.join(save_dir, save_name))

    return {
        'algorithm': name,
        **scores,
        'model': pipeline_or_model,
    }


def run_arena(
    models: dict,
    preprocessor,
    X_train, y_train, X_test, y_test,
    metrics_fn,
    cat_feature_indices=None,
    module_name='module_a',
) -> pd.DataFrame:
    """
    Train ALL models in the registry and return a comparison DataFrame.

    This is the main function for each module notebook.
    """
    results = []

    with mlflow.start_run(run_name=module_name):
        for name, model in models.items():
            print(f"Training {name}...")
            result = train_and_evaluate(
                name=name,
                model=model,
                preprocessor=preprocessor,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                metrics_fn=metrics_fn,
                cat_feature_indices=cat_feature_indices,
                module_name=module_name,
            )
            results.append(result)
            print(f"  Done: {', '.join(f'{k}={v:.4f}' for k, v in result.items() if isinstance(v, float))}")

    # Build comparison DataFrame (exclude the model object column)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'model'} for r in results])
    return df, results
