"""Feature engineering functions for SmartLend."""
import pandas as pd
import numpy as np


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features for Module E.

    Features engineered:
    - months_since_earliest_cr_line: proxy for credit history length
    - balance_to_loan_ratio: current balance / original loan amount
    """
    df = df.copy()

    # Months since earliest credit line (if earliest_cr_line column exists)
    if 'earliest_cr_line' in df.columns and 'issue_d' in df.columns:
        df['earliest_cr_line'] = pd.to_datetime(
            df['earliest_cr_line'], format='mixed', errors='coerce'
        )
        df['months_since_earliest_cr_line'] = (
            (df['issue_d'] - df['earliest_cr_line']).dt.days / 30.44
        ).round(0)
    else:
        # Fallback: use total_acc as a proxy for credit history length
        df['months_since_earliest_cr_line'] = df.get('total_acc', 0) * 12

    # Balance-to-loan ratio
    if 'revol_bal' in df.columns and 'loan_amnt' in df.columns:
        df['balance_to_loan_ratio'] = (
            df['revol_bal'] / (df['loan_amnt'] + 1e-8)
        ).clip(0, 100)

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction features that may improve model performance.
    """
    df = df.copy()

    # Income-to-loan ratio (ability to repay)
    if 'annual_inc' in df.columns and 'loan_amnt' in df.columns:
        df['income_to_loan'] = df['annual_inc'] / (df['loan_amnt'] + 1e-8)

    # Installment-to-income ratio (monthly burden)
    if 'installment' in df.columns and 'annual_inc' in df.columns:
        df['installment_to_income'] = (
            df['installment'] * 12 / (df['annual_inc'] + 1e-8)
        )

    # Credit utilization risk (high revol_util + high dti = very risky)
    if 'revol_util' in df.columns and 'dti' in df.columns:
        df['credit_risk_score'] = df['revol_util'] * df['dti'] / 100

    return df
