"""Shared feature engineering for the SD permit delay models.

Used by both train.py and app.py so the training-time and serving-time
feature construction can never drift apart.

All features must be knowable at submission time. Documented assumptions:
- APPROVAL_PROCESSING_CODE (Standard/Express/Expedite/Not Required) is an
  intake classification chosen when the application is filed.
- APPROVAL_SCOPE is the applicant-written description at submission.
- APPROVAL_PERMIT_HOLDER is the applicant/contractor named on the application.
Post-decision columns (close/expire dates, status) are never used.
"""

import numpy as np
import pandas as pd

PROCESSING_CODES = ["Standard", "Express", "Expedite", "Not Required"]

# Representative permits-per-year values for the app's contractor-volume
# dropdown, matching the buckets observed in training data.
HOLDER_VOLUME_CHOICES = {
    "First-time / DIY applicant": 1,
    "Occasional (2-10 permits/yr)": 5,
    "Active contractor (11-100 permits/yr)": 50,
    "High-volume contractor (100+ permits/yr)": 300,
    "Not sure": 0,
}


def engineer_features(df, top_types, holder_freq=None):
    """Build the model feature matrix from a raw approvals dataframe.

    Parameters
    ----------
    df : raw dataframe with the city's column names
    top_types : list of APPROVAL_TYPE values kept as their own categories
    holder_freq : dict mapping APPROVAL_PERMIT_HOLDER -> permit count in the
        training year. Unseen/missing holders get 0.

    Returns a dataframe of model features (no target).
    """
    out = pd.DataFrame(index=df.index)

    # --- location ---
    out["GIS_LATITUDE"] = df["GIS_LATITUDE"]
    out["GIS_LONGITUDE"] = df["GIS_LONGITUDE"]

    # --- permit type: top N + Other ---
    type_grouped = df["APPROVAL_TYPE"].where(
        df["APPROVAL_TYPE"].isin(top_types), "Other"
    )
    for t in list(top_types) + ["Other"]:
        out[f"type_{t}"] = (type_grouped == t).astype(int)

    # --- submission month ---
    month = pd.to_datetime(df["APPROVAL_CREATE_DATE"]).dt.month
    for m in range(1, 13):
        out[f"month_{m}"] = (month == m).astype(int)

    # --- linked to a larger project ---
    out["has_project"] = df["PROJECT_ID"].notna().astype(int)

    # --- processing track (missing/undefined -> all zeros) ---
    for code in PROCESSING_CODES:
        col = "proc_" + code.replace(" ", "_")
        out[col] = (df["APPROVAL_PROCESSING_CODE"] == code).astype(int)

    # --- declared valuation ---
    val = pd.to_numeric(df["APPROVAL_VALUATION"], errors="coerce")
    has_val = val.notna() & (val > 0)
    out["has_valuation"] = has_val.astype(int)
    out["log_valuation"] = np.where(has_val, np.log1p(val.fillna(0)), 0.0)

    # --- scope description length (log chars; 0 = none provided) ---
    scope_len = df["APPROVAL_SCOPE"].fillna("").str.len()
    out["log_scope_len"] = np.log1p(scope_len)

    # --- permit-holder volume (log permits/yr in training data) ---
    if holder_freq is not None:
        counts = df["APPROVAL_PERMIT_HOLDER"].map(holder_freq).fillna(0)
    else:
        counts = pd.Series(0, index=df.index)
    out["log_holder_volume"] = np.log1p(counts)

    return out


def build_input_row(
    features_to_keep,
    top_types,
    approval_type,
    submit_month,
    latitude,
    longitude,
    has_project=False,
    processing_code=None,
    valuation=0.0,
    scope_text="",
    holder_permits_per_year=0,
    scope_len=None,
):
    """Build a single-row feature frame from app inputs.

    Routes through the same column conventions as engineer_features so the
    app can never drift from training. Returns a 1-row DataFrame ordered by
    features_to_keep.
    """
    row = {col: 0 for col in features_to_keep}
    row["GIS_LATITUDE"] = latitude
    row["GIS_LONGITUDE"] = longitude

    type_col = f"type_{approval_type}"
    if type_col in row:
        row[type_col] = 1
    elif "type_Other" in row:
        row["type_Other"] = 1

    month_col = f"month_{submit_month}"
    if month_col in row:
        row[month_col] = 1

    row["has_project"] = int(bool(has_project))

    if processing_code in PROCESSING_CODES:
        row["proc_" + processing_code.replace(" ", "_")] = 1

    if valuation and valuation > 0:
        row["has_valuation"] = 1
        row["log_valuation"] = float(np.log1p(valuation))

    # scope_len overrides the text length when provided (used for imputing
    # a typical description length when the user leaves the field blank)
    effective_len = scope_len if scope_len is not None else len(scope_text or "")
    row["log_scope_len"] = float(np.log1p(effective_len))
    row["log_holder_volume"] = float(np.log1p(holder_permits_per_year or 0))

    return pd.DataFrame([row])[features_to_keep]


MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
]


def friendly_name(col):
    """Readable labels for feature columns, used in SHAP charts."""
    month_names = MONTH_NAMES
    if col.startswith("type_"):
        return "Permit Type: " + col[len("type_"):]
    if col.startswith("month_"):
        return "Submitted in " + month_names[int(col[len("month_"):]) - 1]
    if col.startswith("proc_"):
        return "Processing: " + col[len("proc_"):].replace("_", " ")
    return {
        "GIS_LATITUDE": "Latitude",
        "GIS_LONGITUDE": "Longitude",
        "has_project": "Part of a larger project",
        "has_valuation": "Declared a project valuation",
        "log_valuation": "Project valuation (log $)",
        "log_scope_len": "Length of scope description",
        "log_holder_volume": "Permit holder's annual volume",
    }.get(col, col)


def plain_phrase(col, value):
    """Sentence-fragment form of a feature, for plain-English explanations.

    Reads naturally in "<phrase> raised the estimate" — the SHAP chart says
    the same thing in log-odds, which most readers can't parse on sight.

    `value` is the feature's value for this permit, and it matters: one-hot
    columns that are OFF still carry a SHAP attribution, so phrasing them
    like they're on ("the Express processing track") would claim something
    about the permit that isn't true. Returns None when a feature has
    nothing worth saying in words.
    """
    on = bool(value)

    if col.startswith("type_"):
        # "not being a Traffic Control Permit" is noise; only speak when on
        return f"the permit type ({col[len('type_'):]})" if on else None
    if col.startswith("month_"):
        m = MONTH_NAMES[int(col[len("month_"):]) - 1]
        return f"submitting in {m}" if on else None
    if col.startswith("proc_"):
        track = col[len("proc_"):].replace("_", " ")
        return f"the {track} processing track" if on else (
            f"not being on the {track} track"
        )
    if col == "has_project":
        return ("being part of a larger development project" if on else
                "being a standalone permit, not tied to a larger project")
    if col == "has_valuation":
        return ("having a declared project valuation" if on else
                "having no declared project valuation")
    return {
        "GIS_LATITUDE": "the project location",
        "GIS_LONGITUDE": "the project location",
        "log_valuation": "the declared project valuation",
        "log_scope_len": "the length of the scope description",
        "log_holder_volume": "the permit holder's annual permit volume",
    }.get(col, col)
