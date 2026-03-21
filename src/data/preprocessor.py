"""Preprocessing pipelines using ColumnTransformer."""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES


def build_preprocessor(X_train) -> ColumnTransformer:
    """
    Build a ColumnTransformer that handles both numerical and categorical features.

    Numerical: median imputation -> standard scaling
    Categorical: most-frequent imputation -> one-hot encoding

    Only includes columns that exist in X_train.
    """
    num_cols = [c for c in NUMERICAL_FEATURES if c in X_train.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols),
        ],
        remainder='drop',
    )

    return preprocessor


def get_catboost_cat_indices(X_train) -> list:
    """
    Return column indices for categorical features.
    CatBoost uses these instead of a preprocessor.
    """
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]
    return [X_train.columns.get_loc(c) for c in cat_cols]
