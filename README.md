# 🛒 Walmart Weekly Sales Forecasting — ML Project

A machine learning project that predicts Walmart's weekly sales per store/department using historical sales, store metadata, and economic indicators (CPI, unemployment, fuel price, temperature, markdown promotions, and holidays). Four modeling pipelines are trained and compared, and the best models are served through an interactive Streamlit app.

## Overview

The project walks through a full ML workflow on the classic [Walmart Recruiting — Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting) dataset:

1. Merge and clean sales, store, and macroeconomic feature data
2. Engineer features (aggregated markdowns, holiday flags, IQR outlier capping)
3. Perform EDA on a normalized copy of the data (kept separate from the raw training data to avoid leakage/double-normalization)
4. Train and compare **four** regression approaches:
   | Approach | Model | Normalization |
   |---|---|---|
   | 1 | Random Forest | MinMax-scaled economic features |
   | 2 | Random Forest | Raw (unscaled) features |
   | 3 | Linear Regression | MinMax-scaled economic features |
   | 4 | XGBoost | Raw (unscaled) features |
5. Evaluate with MAE, RMSE, and R², and visualize predictions/residuals/feature importance
6. Serve predictions through a Streamlit web app where the user can pick which trained pipeline to use

## Project structure

```
.
├── Codes/
│   ├── wm_sales_v8_with_xgboost.ipynb   # Final notebook — data prep, EDA, all 4 pipelines
│   ├── xgboost_cells_to_append.py       # Standalone XGBoost training script
│   ├── *.pkl / *.json                   # Small trained models + feature-order metadata
│   ├── *.png                            # Model comparison / diagnostic plots
│   └── walmart_ml_app/
│       └── app.py                       # Streamlit app — loads trained models and serves predictions
├── Dataset/
│   ├── train.csv, test.csv              # Weekly sales by store/department
│   ├── stores.csv                       # Store type & size
│   └── features.xlsx                    # CPI, unemployment, fuel price, markdowns, temperature
├── archive.zip                          # Supplementary retail inventory dataset
├── ML_Presentation.pptx
├── Poster-ml.pptx
├── walmart_ml_report_v2 (1).docx
├── walmart_ml_report_v2 (1).pdf
└── requirements.txt
```

## ⚠️ Note on trained model files

The large trained Random Forest models (`model.pkl`, `model_raw.pkl`, `model_norm.pkl`, ~2.9 GB each) are **not included in this repository** — they exceed GitHub's 100 MB file size limit. The smaller models (Linear Regression, XGBoost, and the scaler) are included and ready to use.

To regenerate the missing Random Forest models, run `Codes/wm_sales_v8_with_xgboost.ipynb` end to end — it will re-save `model.pkl`, `model_raw.pkl`, and `model_norm.pkl` into `Codes/`. Copy them into `Codes/walmart_ml_app/` to use them in the Streamlit app.

## Getting started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the notebook

Open `Codes/wm_sales_v8_with_xgboost.ipynb` in Jupyter to reproduce data cleaning, training, and evaluation.

### 3. Run the Streamlit app

```bash
cd Codes/walmart_ml_app
streamlit run app.py
```

The app lets you pick between the Normalized RF, Unnormalized RF, Linear Regression, and XGBoost pipelines and get a live weekly sales prediction from store, department, date, and economic inputs.

## Dataset

- **train.csv / test.csv** — weekly sales by Store, Dept, and Date, with an `IsHoliday` flag
- **stores.csv** — store `Type` and `Size`
- **features.xlsx** — Temperature, Fuel_Price, CPI, Unemployment, and 5 MarkDown promotional columns per store/week

Source: Kaggle's Walmart Recruiting — Store Sales Forecasting competition dataset.

## Docs

- [`walmart_ml_report_v2 (1).pdf`](./walmart_ml_report_v2%20(1).pdf) — full write-up (opens inline on GitHub)
- [`walmart_ml_report_v2 (1).docx`](./walmart_ml_report_v2%20(1).docx) — same report, editable Word version
- [`Poster-ml.pptx`](./Poster-ml.pptx) — project poster
- [`ML_Presentation.pptx`](./ML_Presentation.pptx) — slide deck
