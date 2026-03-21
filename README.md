# SmartLend: The Algorithm Arena 🏟️

A full-stack machine learning application that compares **8 classical ML algorithms** across **5 different prediction tasks** on real Lending Club data (~2.2M loans).

## Algorithms Compared

| # | Algorithm | Type |
|---|-----------|------|
| 1 | Logistic Regression / Linear Regression | Linear |
| 2 | Decision Tree | Tree |
| 3 | Random Forest | Ensemble (Bagging) |
| 4 | Gradient Boosting | Ensemble (Boosting) |
| 5 | AdaBoost | Ensemble (Boosting) |
| 6 | XGBoost | Ensemble (Boosting) |
| 7 | LightGBM | Ensemble (Boosting) |
| 8 | CatBoost | Ensemble (Boosting) |

## Prediction Modules

| Module | Task | Target |
|--------|------|--------|
| A | Loan Default Classification | Will this loan default? (binary) |
| B | Interest Rate Regression | What interest rate should this loan get? |
| C | Customer Segmentation | What borrower segments exist? (clustering) |
| D | Loss Amount Regression | How much will we lose on a default? |
| E | Temporal Default Prediction | Does the model degrade over time? |

## Tech Stack

- **ML**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Tuning**: Optuna
- **Explainability**: SHAP
- **Tracking**: MLflow
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Dashboard**: Streamlit (multi-page app)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data (requires Kaggle API credentials)
python data/download.py

# 3. Run notebooks in order (00 → 06)
jupyter notebook notebooks/

# 4. Launch dashboard
streamlit run app/streamlit_app.py
```

## Project Structure

```
smartlend/
├── src/              # Reusable ML pipeline code
│   ├── data/         # Loading, preprocessing, feature engineering
│   ├── models/       # Registry, training, hyperparameter tuning
│   ├── evaluation/   # Metrics, comparison tables, SHAP explainability
│   └── utils/        # Timing utilities
├── notebooks/        # 7 Jupyter notebooks (exploration + 5 modules + comparison)
├── app/              # Streamlit multi-page dashboard
├── data/             # Raw and processed datasets
├── models/           # Saved .joblib model files
└── results/          # Comparison tables, figures, verdict
```

## Dataset

**Lending Club Loan Data** — All loans issued 2007–2018 (~2.2M rows, 151 columns).

Source: [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

## Author

**Nando (Do Le)** — University of South Florida, Computer Science, AI/ML Concentration
