# SmartLend: Algorithm Arena — Final Verdict

## Winners by Module

| Module | Task | Winner | Primary Metric |
|--------|------|--------|----------------|
| module_a | Default Classification | Gradient Boosting | auc_roc=0.7140 |
| module_b | Interest Rate Regression | XGBoost | r2=0.9825 |
| module_d | Loss Amount Regression | Gradient Boosting | rmse=3325.5797 |
| module_e | Temporal Default Prediction | LightGBM | auc_roc=0.7079 |

## Key Findings

1. **Gradient boosting dominates**: XGBoost, LightGBM, and CatBoost consistently outperform classical methods.
2. **CatBoost handles categoricals best**: Native categorical support, no one-hot encoding needed.
3. **Temporal split matters**: Random splits inflate metrics — always use time-based splits for lending data.
4. **Logistic Regression is a strong baseline**: Competitive AUC-ROC with interpretability.
5. **Decision Trees overfit**: Largest gap between train and test performance.

## Recommendations

- **Production default prediction**: Use LightGBM or XGBoost with temporal validation.
- **Interpretable baseline**: Use Logistic Regression for regulatory compliance.
- **Fast iteration**: Use LightGBM (fastest training time among boosting methods).
- **Categorical-heavy data**: Use CatBoost to avoid preprocessing overhead.
