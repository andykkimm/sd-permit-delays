"""Training pipeline for the SD permit delay model.

Reproducible end-to-end: cleans the raw CSVs, trains and compares the
baseline (v1) and enriched (v2) binary models, and saves all artifacts
and figures.

Run:  python train.py
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from features import engineer_features

RANDOM_STATE = 42


def load_and_clean(path):
    """Returns (full, modeling) frames: `full` keeps rows missing lat/long
    (used for the top-15 type computation, matching the original notebook);
    `modeling` drops them per the original cleaning policy."""
    df = pd.read_csv(path, low_memory=False)
    df["days_to_approval"] = (
        pd.to_datetime(df["APPROVAL_ISSUE_DATE"])
        - pd.to_datetime(df["APPROVAL_CREATE_DATE"])
    ).dt.days
    df = df[df["days_to_approval"] >= 0]  # drop data-entry errors
    modeling = df.dropna(subset=["GIS_LATITUDE", "GIS_LONGITUDE"])
    return df.reset_index(drop=True), modeling.reset_index(drop=True)


def evaluate(name, model, X, y):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    m = {
        "model": name,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }
    print(
        f"{name:28s} acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
        f"rec={m['recall']:.3f} auc={m['roc_auc']:.3f}"
    )
    return m


def main():
    import os

    os.makedirs("figures", exist_ok=True)

    print("Loading data...")
    train_full, train_df = load_and_clean("data/approvals_issued_2024_datasd.csv")
    _, test_df = load_and_clean("data/approvals_issued_2025_datasd.csv")

    # 75th percentile computed before the lat/long drop, matching the
    # original notebook's 66-day definition
    cutoff = float(train_full["days_to_approval"].quantile(0.75))
    y_train = (train_df["days_to_approval"] > cutoff).astype(int)
    y_test = (test_df["days_to_approval"] > cutoff).astype(int)
    print(
        f"train n={len(train_df)}, test n={len(test_df)}, cutoff={cutoff:.0f} days, "
        f"delay rate train={y_train.mean():.3f} test={y_test.mean():.3f}"
    )

    # Top-15 approval types, computed from training data only (before the
    # lat/long drop, matching the original notebook / saved artifact)
    top_types = train_full["APPROVAL_TYPE"].value_counts().head(15).index.tolist()

    # Permit-holder volume map, computed from training data only
    holder_freq = train_df["APPROVAL_PERMIT_HOLDER"].value_counts().to_dict()

    X_train = engineer_features(train_df, top_types, holder_freq)
    X_test = engineer_features(test_df, top_types, holder_freq)

    # v1 feature set = type + month + lat/long (the original 30 columns)
    v1_cols = [
        c
        for c in X_train.columns
        if c.startswith(("type_", "month_")) or c.startswith("GIS_")
    ]
    v2_cols = list(X_train.columns)
    print(f"v1 features: {len(v1_cols)}, v2 features: {len(v2_cols)}")

    metrics = {}

    # --- Phase 1: baseline reproduction vs enriched model ---
    print("\nTraining v1 (baseline reproduction)...")
    m1 = GradientBoostingClassifier(random_state=RANDOM_STATE)
    m1.fit(X_train[v1_cols], y_train)
    metrics["v1"] = evaluate("v1 (type+month+latlong)", m1, X_test[v1_cols], y_test)

    print("Training v2 (enriched features)...")
    m2 = GradientBoostingClassifier(random_state=RANDOM_STATE)
    m2.fit(X_train[v2_cols], y_train)
    metrics["v2"] = evaluate("v2 (enriched)", m2, X_test[v2_cols], y_test)

    # --- SHAP global importance for v2 ---
    print("\nComputing SHAP summary (5,000-row test sample)...")
    import shap

    sample = X_test[v2_cols].sample(5000, random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(m2)
    shap_values = explainer.shap_values(sample)
    from features import friendly_name

    shap.summary_plot(
        shap_values,
        sample,
        feature_names=[friendly_name(c) for c in v2_cols],
        show=False,
        max_display=15,
    )
    plt.gcf().set_size_inches(9, 6)
    plt.tight_layout()
    plt.savefig("figures/shap_summary_v2.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("saved figures/shap_summary_v2.png")

    # --- persist artifacts ---
    joblib.dump(m2, "model_v2.pkl")
    joblib.dump(v2_cols, "features_v2.pkl")
    joblib.dump(top_types, "top_15_types.pkl")
    joblib.dump(cutoff, "cutoff.pkl")
    joblib.dump(holder_freq, "holder_freq.pkl")
    with open("figures/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nArtifacts saved. Done.")


if __name__ == "__main__":
    main()
