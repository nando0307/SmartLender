"""Load and clean the raw Lending Club CSV."""
import os
from typing import Optional

import pandas as pd
from src.config import *


def load_raw_data(filepath: Optional[str] = None, sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load raw CSV from Kaggle.

    Args:
        filepath: Path to CSV. Defaults to data/raw/accepted_2007_to_2018Q4.csv
        sample_size: If set, sample N rows for faster iteration.

    Returns:
        Cleaned DataFrame with only relevant columns and terminal loan statuses.
    """
    if filepath is None:
        filepath = os.path.join(DATA_RAW, 'accepted_2007_to_2018Q4.csv')

    # Handle Kaggle extraction quirk: CSV may be nested inside a same-named directory
    if os.path.isdir(filepath):
        # Look for the CSV file inside the directory
        for f in os.listdir(filepath):
            if f.endswith('.csv'):
                filepath = os.path.join(filepath, f)
                break

    # Also check lowercase variant from Kaggle extraction
    if not os.path.isfile(filepath):
        alt = os.path.join(DATA_RAW, 'accepted_2007_to_2018q4.csv', 'accepted_2007_to_2018Q4.csv')
        if os.path.isfile(alt):
            filepath = alt

    # Load with low_memory=False to avoid mixed type warnings
    df = pd.read_csv(filepath, low_memory=False)

    # Keep only terminal loan statuses
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()

    # Create binary target
    df['loan_status_binary'] = (df['loan_status'] == 'Charged Off').astype(int)

    # Clean int_rate
    if df['int_rate'].dtype == object:
        df['int_rate'] = df['int_rate'].str.replace('%', '').astype(float)

    # Clean emp_length
    emp_map = {
        '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
        '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
        '8 years': 8, '9 years': 9, '10+ years': 10
    }
    df['emp_length_num'] = df['emp_length'].map(emp_map)

    # Parse dates
    df['issue_d'] = pd.to_datetime(df['issue_d'], format='mixed', errors='coerce')

    # Engineer loss amount for Module D
    if 'total_rec_prncp' in df.columns:
        df['charged_off_amount'] = df['loan_amnt'] - df['total_rec_prncp']
        df['charged_off_amount'] = df['charged_off_amount'].clip(lower=0)

    # Keep only needed columns
    keep_cols = (
        ALL_FEATURES +
        [TARGET_CLASSIFICATION, TARGET_INTEREST_RATE, 'charged_off_amount',
         'issue_d', 'loan_status', 'emp_length_num']
    )
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # Drop rows where all numerical features are NaN
    df = df.dropna(subset=NUMERICAL_FEATURES, how='all')

    # Optional sampling
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=RANDOM_STATE)

    return df.reset_index(drop=True)


def split_data(df: pd.DataFrame, target: str, temporal: bool = False):
    """
    Split data into train/test.

    Args:
        df: Cleaned DataFrame
        target: Name of target column
        temporal: If True, split by date (Module E). If False, stratified random split.

    Returns:
        X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split

    feature_cols = [c for c in ALL_FEATURES if c in df.columns]

    if temporal:
        # Time-based split: train on pre-2017, test on 2017+
        train_mask = df['issue_d'] < TEMPORAL_SPLIT_DATE
        test_mask = df['issue_d'] >= TEMPORAL_SPLIT_DATE

        X_train = df.loc[train_mask, feature_cols]
        X_test = df.loc[test_mask, feature_cols]
        y_train = df.loc[train_mask, target]
        y_test = df.loc[test_mask, target]
    else:
        X = df[feature_cols]
        y = df[target]

        stratify = y if y.nunique() <= 10 else None  # Only stratify for classification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
        )

    return X_train, X_test, y_train, y_test
