"""Hyperparameter tuning with Optuna."""
import numpy as np
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import roc_auc_score
from src.config import N_OPTUNA_TRIALS, RANDOM_STATE

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_xgboost_classifier(X_train, y_train, n_trials=N_OPTUNA_TRIALS):
    """Tune XGBoost classifier with Optuna. Returns best params dict."""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': RANDOM_STATE,
            'eval_metric': 'logloss',
            'verbosity': 0,
        }
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params, study.best_value


def tune_lightgbm_classifier(X_train, y_train, n_trials=N_OPTUNA_TRIALS):
    """Tune LightGBM classifier with Optuna."""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': RANDOM_STATE,
            'verbose': -1,
        }
        model = LGBMClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params, study.best_value


def tune_catboost_classifier(X_train, y_train, cat_indices, n_trials=N_OPTUNA_TRIALS):
    """Tune CatBoost classifier with Optuna. Handles categorical features natively.

    Uses manual cross-validation instead of cross_val_score because sklearn's
    clone() cannot handle CatBoost's cat_features parameter.
    """
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 50, 500),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
            'random_seed': RANDOM_STATE,
            'verbose': 0,
        }

        # Manual 3-fold CV to avoid sklearn clone() incompatibility with cat_features
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = []

        X_arr = X_train.values if hasattr(X_train, 'values') else X_train
        y_arr = y_train.values if hasattr(y_train, 'values') else y_train

        for train_idx, val_idx in skf.split(X_arr, y_arr):
            X_fold_train, X_fold_val = X_arr[train_idx], X_arr[val_idx]
            y_fold_train, y_fold_val = y_arr[train_idx], y_arr[val_idx]

            model = CatBoostClassifier(**params, cat_features=cat_indices)
            model.fit(X_fold_train, y_fold_train)
            y_prob = model.predict_proba(X_fold_val)[:, 1]
            scores.append(roc_auc_score(y_fold_val, y_prob))

        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params, study.best_value
