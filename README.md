# San Diego Permit Delay Predictor

Predicting and explaining San Diego building permit approval delays using gradient boosting and SHAP, built on the City of San Diego's open permit dataset.

**[Live App](#) · [Analysis Notebook](eda.ipynb) · [Decision Log](DECISIONS.md)**

## Overview

The City of San Diego's Development Services Department processes tens of thousands of permit approvals a year, everything from homeowner renovations to large residential developments. Some permits move in days, others sit for months. This project predicts whether a given permit is likely to be delayed and, more importantly, explains *why*, using SHAP to break down which factors are driving each individual prediction.

## Key Findings

- **Permit type is by far the strongest predictor of delay**, outweighing both timing and location.
- Permits requiring full construction plan review (Combination Building Permit, Building Permit, Electrical/Plumbing/Mechanical Pmt) are significantly more likely to be delayed.
- Routine, no-plan-required permits (Traffic Control Permit, No-Plan Residential Combo) are strongly associated with on-time approval.
- This pattern was first spotted in early exploratory analysis and later confirmed independently by the model's SHAP feature importances.
- Submission month has a modest effect (December and February show the highest delay rates), while location shows a smaller, continuous effect.

## Data

- Source: [City of San Diego Open Data Portal](https://data.sandiego.gov/datasets/development-permits/) — "Approvals for development projects"
- 2024 data used for training (~55K approvals), 2025 data used as a held-out test set (~55K approvals), split by year to simulate real-world deployment and avoid data leakage
- Target variable: whether a permit's approval took longer than the 75th percentile (66 days) of the training distribution

## Methodology

1. **Cleaning:** removed ~2 rows with negative approval times (data entry errors) and ~2,100 rows missing location data
2. **Feature engineering:**
   - Approval type: grouped 94 raw categories into the top 15 + "Other," then one-hot encoded
   - Submission month: one-hot encoded (kept over quarter, which averaged away a real December/February pattern)
   - Location: kept as raw latitude/longitude, since gradient boosting can split on continuous coordinates directly
3. **Modeling:** `GradientBoostingClassifier` (scikit-learn), trained on 2024, evaluated on 2025
4. **Explainability:** SHAP `TreeExplainer` for both global feature importance and per-prediction breakdowns

Full reasoning behind each decision is logged in [DECISIONS.md](DECISIONS.md).

## Results

| Metric | Score |
|---|---|
| Accuracy | 85.1% |
| Precision | 65.7% |
| Recall | 74.9% |
| ROC-AUC | 0.916 |

(Naive baseline of always predicting "not delayed" would score 77.7% accuracy — the model provides real lift beyond class imbalance.)

## App

An interactive Streamlit dashboard lets users input a hypothetical permit (type, month, location) and see:
- A predicted delay probability
- A SHAP waterfall chart explaining exactly which factors drove that specific prediction

To run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

Python, Pandas, scikit-learn, SHAP, Streamlit, Matplotlib

## Project Structure

```
sd-permit-delays/
├── eda.ipynb              # Full analysis: cleaning, feature engineering, modeling, SHAP
├── app.py                 # Streamlit dashboard
├── DECISIONS.md            # Log of key project decisions and reasoning
├── model.pkl               # Trained model
├── features_to_keep.pkl    # Feature column reference
├── top_15_types.pkl        # Approval type grouping reference
├── cutoff.pkl               # Delay threshold (66 days)
└── data/                   # Raw CSVs (not tracked in git, see .gitignore)
```