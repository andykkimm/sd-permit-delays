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


# --- Load saved model and reference data (cached across reruns) ---
@st.cache_resource
def load_artifacts():
    return {
        "model": joblib.load("model_v2.pkl"),
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
    "Predicts whether a building permit approval is likely to be delayed "
    f"(defined as taking more than {int(A['cutoff'])} days), "
    "based on permit type, timing, location, and project details."
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
            "sensitivity": sensitivity,
        }

    if "result" not in st.session_state:
        st.info("Fill in the permit details on the left and click **Predict**.")
    else:
        R = st.session_state["result"]
        X_input, sensitivity = R["X_input"], R["sensitivity"]

        probability = A["model"].predict_proba(X_input)[0][1]
        prediction = int(probability >= sensitivity)
        if prediction == 1:
            st.error(f"Likely DELAYED — {probability:.1%} estimated probability")
        else:
            st.success(f"Likely ON TIME — {probability:.1%} estimated delay probability")
        st.caption(
            f"'Delayed' means taking longer than {int(A['cutoff'])} days, the "
            "75th percentile of 2024 approvals — about 25% of permits."
        )

        # -- SHAP explanation --
        st.subheader("Why this prediction?")
        st.markdown("🔴 **Pushed toward DELAYED** &nbsp;&nbsp; 🔵 **Pushed toward ON TIME**")

        explainer = shap.TreeExplainer(A["model"])
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

# --- About the model ---
with st.expander("About the model"):
    st.markdown(
        """
A gradient boosting classifier trained on 2024 San Diego permit approvals
and evaluated on a fully held-out 2025 test year. The second version added
project-context features (linked development project, processing track,
declared valuation, scope description, contractor volume) mined from
columns the first version ignored:
        """
    )
    try:
        with open("figures/metrics.json") as f:
            metrics = json.load(f)
        rows = []
        for key, label in [("v1", "v1 — type, month, location"),
                           ("v2", "v2 — + project context (deployed)")]:
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
        st.caption("Metrics on the held-out 2025 test set.")
    except FileNotFoundError:
        pass
