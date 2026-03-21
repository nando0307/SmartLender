"""
Model registry: all 8 algorithms in one place.

Design principle: adding a new algorithm = adding one line.
All models follow the scikit-learn API: fit(), predict(), predict_proba(), score().
"""
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from src.config import RANDOM_STATE


def get_classifiers() -> dict:
    """Return dict of name -> classifier instance with default hyperparameters."""
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10, random_state=RANDOM_STATE
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=200, random_state=RANDOM_STATE, algorithm='SAMME'
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            eval_metric='logloss', verbosity=0, enable_categorical=False
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=200, random_state=RANDOM_STATE, verbose=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=200, random_seed=RANDOM_STATE, verbose=0
        ),
    }


def get_regressors() -> dict:
    """Return dict of name -> regressor instance with default hyperparameters."""
    return {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(
            max_depth=10, random_state=RANDOM_STATE
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200, random_state=RANDOM_STATE
        ),
        'AdaBoost': AdaBoostRegressor(
            n_estimators=200, random_state=RANDOM_STATE
        ),
        'XGBoost': XGBRegressor(
            n_estimators=200, random_state=RANDOM_STATE, verbosity=0
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=200, random_state=RANDOM_STATE, verbose=-1
        ),
        'CatBoost': CatBoostRegressor(
            iterations=200, random_seed=RANDOM_STATE, verbose=0
        ),
    }
