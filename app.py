import json

import folium
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from streamlit_folium import st_folium

from features import (
    HOLDER_VOLUME_CHOICES,
    PROCESSING_CODES,
    build_input_row,
    friendly_name,
)

st.set_page_config(page_title="SD Permit Delay Predictor", layout="wide")


# --- Load saved models and reference data (cached across reruns) ---
@st.cache_resource
def load_artifacts():
    return {
        "raw": joblib.load("model_v2_raw.pkl"),        # uncalibrated, for SHAP
        "cal": joblib.load("model_v2_cal.pkl"),        # deployed probabilities
        "thresholds": joblib.load("threshold_models.pkl"),
        "features": joblib.load("features_v2.pkl"),
        "top_types": list(joblib.load("top_15_types.pkl")),
        "cutoff": float(joblib.load("cutoff.pkl")),
    }


A = load_artifacts()
MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
MONTH_NAME_TO_NUM = {name: i + 1 for i, name in enumerate(MONTH_NAMES)}
SD_CENTER = (32.7157, -117.1611)

st.title("San Diego Permit Delay Predictor")
st.write(
    "Estimates how long a building permit approval is likely to take — and the "
    "chance it blows past **your** deadline — based on permit type, timing, "
    "location, and project details."
)

input_col, output_col = st.columns(2, gap="large")

# --- Inputs (left column) ---
with input_col:
    st.header("Permit Details")

    approval_type = st.selectbox("Approval Type", A["top_types"] + ["Other"])
    submit_month_name = st.selectbox("Submission Month", MONTH_NAMES)
    submit_month = MONTH_NAME_TO_NUM[submit_month_name]

    st.markdown("**Location** — click the map to place your permit")
    # CartoDB Voyager: clean Mapbox-style look, free, no API token needed.
    # To use actual Mapbox tiles instead, sign up for a token and pass
    # tiles=f"https://api.mapbox.com/styles/v1/mapbox/streets-v12/tiles/{{z}}/{{x}}/{{y}}?access_token={token}"
    # with attr="Mapbox", reading the token from st.secrets.
    m = folium.Map(location=SD_CENTER, zoom_start=10, tiles="CartoDB Voyager")
    if "clicked_location" in st.session_state:
        folium.Marker(st.session_state["clicked_location"]).add_to(m)
    map_state = st_folium(m, height=320, use_container_width=True, key="sd_map")
    if map_state and map_state.get("last_clicked"):
        st.session_state["clicked_location"] = (
            map_state["last_clicked"]["lat"],
            map_state["last_clicked"]["lng"],
        )
    latitude, longitude = st.session_state.get("clicked_location", SD_CENTER)
    st.caption(f"Selected location: {latitude:.4f}, {longitude:.4f}")

    with st.expander("More project details (optional, improves the estimate)"):
        has_project = st.checkbox(
            "Part of a larger development project", value=False
        )
        processing_code = st.selectbox(
            "Processing track",
            ["Not sure"] + PROCESSING_CODES,
            help="The review track assigned when the application is filed.",
        )
        valuation = st.number_input(
            "Declared project valuation ($, 0 if none)",
            min_value=0.0, value=0.0, step=1000.0,
        )
        scope_text = st.text_area(
            "Scope of work description (as you'd write it on the application)",
            value="",
        )
        holder_choice = st.selectbox(
            "Who's pulling the permit?", list(HOLDER_VOLUME_CHOICES)
        )

    deadline = st.number_input(
        "**Your deadline (days)** — what counts as 'too long' for you?",
        min_value=1, max_value=365, value=int(A["cutoff"]),
        help="The model estimates the chance your approval takes longer than this.",
    )

    sensitivity = st.slider(
        "Prediction sensitivity (lower = flag more permits as at-risk)",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
    )
    st.caption(
        "Sensitivity doesn't change the estimates — it changes how cautious the "
        "DELAYED/ON-TIME label is. Cautious planners might flag anything over "
        "30% risk; others only want a warning above 70%."
    )

    predict_clicked = st.button("Predict", type="primary")

# --- Prediction helpers ---
THRESHOLD_GRID = sorted(A["thresholds"].keys())


def survival_curve(X):
    """P(days > t) at each grid threshold, forced monotone non-increasing."""
    probs = np.array(
        [A["thresholds"][t].predict_proba(X)[0][1] for t in THRESHOLD_GRID]
    )
    return np.minimum.accumulate(probs)


def prob_exceeds(curve, t):
    """P(days > t) for an arbitrary deadline, interpolating the grid.

    Returns (prob, extrapolated); extrapolated is True when t lies outside
    the trained 15-180 day grid, where the estimate is an endpoint value.
    """
    p = float(np.interp(t, THRESHOLD_GRID, curve))
    return p, t < THRESHOLD_GRID[0] or t > THRESHOLD_GRID[-1]


def curve_median(curve):
    """Deadline at which risk crosses 50% — a 'typical wait' estimate.

    Returns (days, note) where note flags out-of-grid answers.
    """
    if curve[0] < 0.5:  # even 15 days is more likely than not enough
        return THRESHOLD_GRID[0], "under"
    if curve[-1] > 0.5:  # majority of similar permits blow past 180 days
        return THRESHOLD_GRID[-1], "over"
    # interpolate on the decreasing curve
    return float(np.interp(0.5, curve[::-1], THRESHOLD_GRID[::-1])), "within"


# --- Outputs (right column) ---
# Computed on Predict, stashed in session_state so results survive the
# reruns Streamlit fires on every map click / widget change.
with output_col:
    st.header("Prediction")
    if predict_clicked:
        X_input = build_input_row(
            A["features"], A["top_types"], approval_type, submit_month,
            latitude, longitude,
            has_project=has_project,
            processing_code=None if processing_code == "Not sure" else processing_code,
            valuation=valuation,
            scope_text=scope_text,
            holder_permits_per_year=HOLDER_VOLUME_CHOICES[holder_choice],
        )

        st.session_state["result"] = {
            "X_input": X_input,
            "deadline": deadline,
            "sensitivity": sensitivity,
        }

    if "result" not in st.session_state:
        st.info("Fill in the permit details on the left and click **Predict**.")
    else:
        R = st.session_state["result"]
        X_input, deadline, sensitivity = R["X_input"], R["deadline"], R["sensitivity"]

        # -- your-deadline estimate from the threshold-grid models --
        curve = survival_curve(X_input)
        p_deadline, extrapolated = prob_exceeds(curve, deadline)
        med, med_note = curve_median(curve)

        st.subheader(f"Chance of exceeding your {deadline}-day deadline")
        st.metric("Estimated risk", f"{p_deadline:.0%}")
        if extrapolated:
            st.caption(
                f"Deadlines outside {THRESHOLD_GRID[0]}-{THRESHOLD_GRID[-1]} "
                "days fall beyond the modeled range; this is the nearest "
                "in-range estimate."
            )
        if med_note == "under":
            st.write(
                f"Permits like this are usually quick — better than even odds "
                f"of approval within {THRESHOLD_GRID[0]} days."
            )
        elif med_note == "over":
            st.write(
                f"Permits like this usually take a while — better than even "
                f"odds of exceeding {THRESHOLD_GRID[-1]} days."
            )
        else:
            st.write(
                f"Typical approval for a permit like this: about **{med:.0f} "
                f"days** (the 50/50 point)."
            )

        # -- the standard 66-day framing, calibrated --
        probability = A["cal"].predict_proba(X_input)[0][1]
        prediction = int(probability >= sensitivity)
        st.subheader(f"Standard {int(A['cutoff'])}-day definition")
        if prediction == 1:
            st.error(f"Likely DELAYED — {probability:.1%} estimated probability")
        else:
            st.success(f"Likely ON TIME — {probability:.1%} estimated delay probability")
        st.caption(
            f"'Delayed' here means taking longer than {int(A['cutoff'])} days, the "
            "75th percentile of 2024 approvals — about 25% of permits. These "
            "probabilities are calibration-checked on held-out 2025 data: "
            "among permits given ~30%, roughly 30% actually ran late "
            "(expected calibration error ≈ 2%)."
        )

        # -- SHAP explanation --
        st.subheader("Why this prediction?")
        st.markdown("🔴 **Pushed toward DELAYED** &nbsp;&nbsp; 🔵 **Pushed toward ON TIME**")

        explainer = shap.TreeExplainer(A["raw"])
        shap_values = explainer.shap_values(X_input)[0]

        top_n = 10
        order = np.argsort(np.abs(shap_values))[::-1][:top_n]
        top_values = shap_values[order]
        top_labels = [friendly_name(A["features"][i]) for i in order]
        colors = ['#d62728' if v > 0 else '#1f77b4' for v in top_values]

        fig, ax = plt.subplots(figsize=(5, 4.5))
        y_pos = np.arange(len(top_labels))
        ax.barh(y_pos, top_values, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_labels, fontsize=8)
        ax.invert_yaxis()  # strongest feature at the top
        ax.set_xlabel("Impact on prediction (SHAP value)")
        ax.axvline(0, color='black', linewidth=0.8)
        fig.tight_layout()
        st.pyplot(fig)

# --- How the models work ---
with st.expander("How this works: two models, one goal"):
    st.markdown(
        """
**This app started with one model and grew a second because of user feedback.**

The original model answers a fixed question — *"will this take longer than 66
days?"* — as a binary classifier. A tester asked to set their **own** threshold
("what about 30 days? 90?"), which a binary model fundamentally can't do: it's
trained on one specific label.

The fix was reframing: a **grid of classifiers** trained at 15, 30, 45, 66,
90, 120, and 180 days now traces out the full risk curve. Interpolating
between them answers "what's the chance this exceeds *t* days?" for any
deadline — the feature that couldn't exist before.

(We first tried quantile regression for this, but approval times are
zero-inflated — roughly 38% of permits are approved almost instantly — and
the lower-quantile models collapsed to predicting zero for everything. The
threshold grid sidesteps that failure mode; the full story is in the
project's decision log.)
        """
    )
    try:
        with open("figures/metrics.json") as f:
            metrics = json.load(f)
        rows = []
        for key, label in [("v1", "v1 — type, month, location"),
                           ("v2", "v2 — + project/valuation/scope/contractor (deployed)")]:
            if key in metrics:
                m = metrics[key]
                rows.append({
                    "Model": label,
                    "Accuracy": f"{m['accuracy']:.1%}",
                    "Precision": f"{m['precision']:.1%}",
                    "Recall": f"{m['recall']:.1%}",
                    "ROC-AUC": f"{m['roc_auc']:.3f}",
                })
        st.table(pd.DataFrame(rows))
        st.caption(
            "Binary-model comparison on the held-out 2025 test set. The "
            "quantile models are validated separately by coverage: their "
            "predicted percentiles match observed 2025 approval times."
        )
    except FileNotFoundError:
        pass
