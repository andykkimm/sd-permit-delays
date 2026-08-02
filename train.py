"""Training pipeline for the SD permit delay models.

Reproducible end-to-end: cleans the raw CSVs, trains and compares the
baseline (v1) and enriched (v2) binary models, calibrates probabilities,
trains quantile regressors for user-chosen day thresholds, and saves all
artifacts + figures.

Run:  python train.py
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from features import engineer_features

RANDOM_STATE = 42
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
# Day thresholds for the "your own deadline" feature: one classifier per
# threshold, interpolated in the app for arbitrary user deadlines.
THRESHOLD_GRID = [15, 30, 45, 66, 90, 120, 180]


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
        "brier": brier_score_loss(y, proba),
    }
    print(
        f"{name:28s} acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
        f"rec={m['recall']:.3f} auc={m['roc_auc']:.3f} brier={m['brier']:.4f}"
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

    # --- Phase 2: calibration ---
    # Method selection must not touch the 2025 test set: hold out the last
    # two months of 2024 (temporal, mimicking deployment), pick the method
    # with the best Brier score there, then refit on all of 2024.
    print("\nSelecting calibration method on Nov-Dec 2024 holdout...")
    create = pd.to_datetime(train_df["APPROVAL_CREATE_DATE"])
    holdout_mask = (create >= "2024-11-01").values
    Xt, yt = X_train[v2_cols][~holdout_mask], y_train[~holdout_mask]
    Xh, yh = X_train[v2_cols][holdout_mask], y_train[holdout_mask]

    candidates = {}
    base = GradientBoostingClassifier(random_state=RANDOM_STATE).fit(Xt, yt)
    candidates["none"] = base
    for method in ["isotonic", "sigmoid"]:
        c = CalibratedClassifierCV(
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            method=method, cv=5,
        )
        c.fit(Xt, yt)
        candidates[method] = c
    briers = {
        name: brier_score_loss(yh, m.predict_proba(Xh)[:, 1])
        for name, m in candidates.items()
    }
    for name, b in briers.items():
        print(f"  {name:10s} holdout brier={b:.4f}")
    best_method = min(briers, key=briers.get)
    print(f"  selected: {best_method}")
    metrics["calibration_selection"] = {"holdout_brier": briers, "chosen": best_method}

    if best_method == "none":
        cal = m2  # already fit on all of 2024
    else:
        cal = CalibratedClassifierCV(
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            method=best_method, cv=5,
        )
        cal.fit(X_train[v2_cols], y_train)
    metrics["v2_calibrated"] = evaluate(
        f"v2 deployed ({best_method})", cal, X_test[v2_cols], y_test
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    for model, label, style in [
        (m2, "v2 uncalibrated", "o-"),
        (cal, f"v2 deployed ({best_method})", "s-"),
    ]:
        proba = model.predict_proba(X_test[v2_cols])[:, 1]
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10)
        ax.plot(mean_pred, frac_pos, style, label=label)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed delay rate")
    ax.set_title("Calibration on 2025 test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/calibration.png", dpi=150)
    plt.close(fig)
    print("saved figures/calibration.png")

    # --- Phase 3: user-chosen day thresholds ---
    # Primary approach: a grid of binary classifiers, one per threshold.
    # days_to_approval is zero-inflated (~38% of permits approved almost
    # instantly), which broke quantile regression's lower tail (kept below
    # as a documented negative result). Binary classifiers at each grid
    # point sidestep the point mass entirely.
    print("\nTraining threshold-grid classifiers...")
    threshold_models = {}
    threshold_metrics = {}
    for t in THRESHOLD_GRID:
        yt_train = (train_df["days_to_approval"] > t).astype(int)
        yt_test = (test_df["days_to_approval"] > t).astype(int)
        tm = GradientBoostingClassifier(random_state=RANDOM_STATE)
        tm.fit(X_train[v2_cols], yt_train)
        proba = tm.predict_proba(X_test[v2_cols])[:, 1]
        threshold_models[t] = tm
        threshold_metrics[t] = {
            "base_rate": float(yt_test.mean()),
            "roc_auc": float(roc_auc_score(yt_test, proba)),
            "brier": float(brier_score_loss(yt_test, proba)),
        }
        print(
            f"  t={t:3d}d  base_rate={yt_test.mean():.3f} "
            f"auc={threshold_metrics[t]['roc_auc']:.3f} "
            f"brier={threshold_metrics[t]['brier']:.4f}"
        )
    metrics["threshold_grid"] = threshold_metrics

    # Negative-result comparison: quantile regression on the raw durations
    print("\nTraining quantile regressors (negative-result comparison)...")
    quantile_models = {}
    coverage = {}
    for q in QUANTILES:
        qm = GradientBoostingRegressor(
            loss="quantile", alpha=q, random_state=RANDOM_STATE
        )
        qm.fit(X_train[v2_cols], train_df["days_to_approval"])
        quantile_models[q] = qm
        pred = qm.predict(X_test[v2_cols])
        coverage[q] = float((test_df["days_to_approval"].values <= pred).mean())
        print(f"  q={q:.2f}  2025 coverage={coverage[q]:.3f} (target {q:.2f})")
    metrics["quantile_coverage"] = coverage

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(QUANTILES, [coverage[q] for q in QUANTILES], "o-", label="observed")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="ideal")
    ax.set_xlabel("Target quantile")
    ax.set_ylabel("Observed coverage on 2025")
    ax.set_title("Quantile model coverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/quantile_coverage.png", dpi=150)
    plt.close(fig)
    print("saved figures/quantile_coverage.png")

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
    joblib.dump(m2, "model_v2_raw.pkl")  # uncalibrated, for SHAP explanations
    joblib.dump(cal, "model_v2_cal.pkl")  # calibrated, for probabilities
    joblib.dump(v2_cols, "features_v2.pkl")
    joblib.dump(top_types, "top_15_types.pkl")
    joblib.dump(cutoff, "cutoff.pkl")
    joblib.dump(holder_freq, "holder_freq.pkl")
    joblib.dump(threshold_models, "threshold_models.pkl")
    joblib.dump(quantile_models, "quantile_models.pkl")
    with open("figures/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nArtifacts saved. Done.")


if __name__ == "__main__":
    main()
