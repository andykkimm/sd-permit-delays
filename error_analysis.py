"""Error analysis for the v2 binary delay model.

Answers "where is the model wrong?" on the 2025 test set:
- precision/recall per permit type
- error rate by submission month
- geographic distribution of false negatives vs correct predictions
- a delay-rate map of San Diego for the README

Run after train.py:  python error_analysis.py
Writes reports/error_analysis.md and figures/*.png
"""

import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import engineer_features
from train import load_and_clean

def main():
    os.makedirs("reports", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    model = joblib.load("model_v2_cal.pkl")
    feature_cols = joblib.load("features_v2.pkl")
    top_types = joblib.load("top_15_types.pkl")
    holder_freq = joblib.load("holder_freq.pkl")
    cutoff = joblib.load("cutoff.pkl")

    _, test_df = load_and_clean("data/approvals_issued_2025_datasd.csv")
    X = engineer_features(test_df, top_types, holder_freq)[feature_cols]
    y = (test_df["days_to_approval"] > cutoff).astype(int).values
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    test_df = test_df.assign(y=y, pred=pred, proba=proba)
    test_df["type_grouped"] = test_df["APPROVAL_TYPE"].where(
        test_df["APPROVAL_TYPE"].isin(list(top_types)), "Other"
    )

    lines = ["# Where the model is wrong (2025 test set)\n"]

    # --- per permit type ---
    rows = []
    for t, g in test_df.groupby("type_grouped"):
        tp = ((g.y == 1) & (g.pred == 1)).sum()
        fp = ((g.y == 0) & (g.pred == 1)).sum()
        fn = ((g.y == 1) & (g.pred == 0)).sum()
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        rows.append(
            dict(
                type=t,
                n=len(g),
                delay_rate=g.y.mean(),
                accuracy=(g.y == g.pred).mean(),
                precision=prec,
                recall=rec,
            )
        )
    per_type = pd.DataFrame(rows).sort_values("n", ascending=False)
    lines.append("## Per permit type\n")
    lines.append(per_type.round(3).to_markdown(index=False))
    lines.append("")

    # --- by month ---
    month = pd.to_datetime(test_df["APPROVAL_CREATE_DATE"]).dt.month
    by_month = test_df.groupby(month).apply(
        lambda g: pd.Series(
            {
                "n": len(g),
                "delay_rate": g.y.mean(),
                "error_rate": (g.y != g.pred).mean(),
                "fn_rate": ((g.y == 1) & (g.pred == 0)).mean(),
            }
        ),
        include_groups=False,
    )
    lines.append("## By submission month\n")
    lines.append(by_month.round(3).to_markdown())
    lines.append("")

    # --- false negatives: how bad were the misses? ---
    fn_days = test_df.loc[(test_df.y == 1) & (test_df.pred == 0), "days_to_approval"]
    del_days = test_df.loc[test_df.y == 1, "days_to_approval"]
    lines.append("## False negatives: how bad were the misses?\n")
    lines.append(
        f"- {len(fn_days)} delayed permits were predicted on-time "
        f"({len(fn_days)/max(1,(test_df.y==1).sum()):.1%} of all delayed permits)\n"
        f"- median actual delay among false negatives: {fn_days.median():.0f} days "
        f"(vs {del_days.median():.0f} days for all delayed permits)\n"
        f"- worst miss: {fn_days.max():.0f} days\n"
    )

    # --- geography: FN locations vs all delayed ---
    fig, ax = plt.subplots(figsize=(7, 7))
    ok = test_df[test_df.y == test_df.pred]
    fn = test_df[(test_df.y == 1) & (test_df.pred == 0)]
    ax.scatter(ok.GIS_LONGITUDE, ok.GIS_LATITUDE, s=2, alpha=0.15,
               c="#9ecae1", label="correct predictions")
    ax.scatter(fn.GIS_LONGITUDE, fn.GIS_LATITUDE, s=6, alpha=0.6,
               c="#d62728", label="false negatives (missed delays)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Where the model misses delays (2025)")
    ax.legend(markerscale=4)
    fig.tight_layout()
    fig.savefig("figures/fn_map.png", dpi=150)
    plt.close(fig)

    # --- README visual: observed delay rate across the city ---
    fig, ax = plt.subplots(figsize=(7, 7))
    hb = ax.hexbin(
        test_df.GIS_LONGITUDE, test_df.GIS_LATITUDE, C=test_df.y,
        gridsize=40, reduce_C_function=np.mean, mincnt=15, cmap="RdYlBu_r",
    )
    fig.colorbar(hb, ax=ax, label="observed delay rate")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("San Diego permit delay rate by area (2025, cells with 15+ permits)")
    fig.tight_layout()
    fig.savefig("figures/delay_map.png", dpi=150)
    plt.close(fig)

    with open("reports/error_analysis.md", "w") as f:
        f.write("\n".join(lines))
    print("wrote reports/error_analysis.md, figures/fn_map.png, figures/delay_map.png")

    # console summary
    print(per_type.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
