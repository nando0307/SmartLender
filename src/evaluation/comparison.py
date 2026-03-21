"""Build comparison DataFrames across modules and algorithms."""
import os
import pandas as pd
from src.config import RESULTS_DIR


def save_comparison(df: pd.DataFrame, module_name: str):
    """Save a comparison DataFrame to CSV."""
    save_dir = os.path.join(RESULTS_DIR, 'comparison_tables')
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f'{module_name}.csv')
    df.to_csv(filepath, index=False)
    print(f"Saved comparison table to {filepath}")
    return filepath


def load_comparison(module_name: str) -> pd.DataFrame:
    """Load a comparison CSV for a given module."""
    filepath = os.path.join(RESULTS_DIR, 'comparison_tables', f'{module_name}.csv')
    return pd.read_csv(filepath)


def build_master_comparison() -> pd.DataFrame:
    """
    Load all module comparison CSVs and build a master table.
    Each row = (algorithm, module, metric, value).
    """
    modules = ['module_a', 'module_b', 'module_d', 'module_e']
    all_dfs = []

    for module in modules:
        try:
            df = load_comparison(module)
            df['module'] = module
            all_dfs.append(df)
        except FileNotFoundError:
            print(f"Warning: {module}.csv not found, skipping.")

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


def get_winners(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each module, find the winning algorithm based on the primary metric.

    Primary metrics:
    - module_a: auc_roc (classification)
    - module_b: r2 (regression)
    - module_d: rmse (lower is better)
    - module_e: auc_roc (classification)
    """
    metric_map = {
        'module_a': ('auc_roc', 'max'),
        'module_b': ('r2', 'max'),
        'module_d': ('rmse', 'min'),
        'module_e': ('auc_roc', 'max'),
    }

    winners = []
    for module, (metric, direction) in metric_map.items():
        module_df = master_df[master_df['module'] == module]
        if module_df.empty or metric not in module_df.columns:
            continue

        if direction == 'max':
            best_idx = module_df[metric].idxmax()
        else:
            best_idx = module_df[metric].idxmin()

        best_row = module_df.loc[best_idx]
        winners.append({
            'module': module,
            'winner': best_row['algorithm'],
            'metric': metric,
            'value': best_row[metric],
        })

    return pd.DataFrame(winners)
