import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# --- Load saved model and reference data ---
model = joblib.load('model.pkl')
features_to_keep = joblib.load('features_to_keep.pkl')
top_15_types = joblib.load('top_15_types.pkl')
cutoff = joblib.load('cutoff.pkl')

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

st.title("San Diego Permit Delay Predictor")
st.write(
    "Predicts whether a building permit approval is likely to be delayed "
    f"(defined as taking more than {int(cutoff)} days), based on permit type, "
    "submission month, and location."
)

# --- Inputs ---
st.header("Permit Details")
st.write("Enter a hypothetical permit's details below to see a live prediction.")

approval_type = st.selectbox("Approval Type", list(top_15_types) + ["Other"])

# Month as a slider instead of a dropdown, showing the month name instead of the raw number
submit_month = st.slider(
    "Submission Month", min_value=1, max_value=12, value=1,
    format_func=lambda m: MONTH_NAMES[m - 1]
)

latitude = st.number_input("Latitude", value=32.7157, format="%.4f")
longitude = st.number_input("Longitude", value=-117.1611, format="%.4f")

# Sensitivity slider: lets the user adjust how confident the model needs to be
# before calling something "delayed," instead of a fixed 50% cutoff.
# Note: this does NOT change what "delayed" means (still >66 days) -- it only
# changes how cautious the app is about flagging risk.
sensitivity = st.slider(
    "Prediction sensitivity (lower = flag more permits as at-risk)",
    min_value=0.1, max_value=0.9, value=0.5, step=0.05
)
st.caption(
    "This doesn't change what counts as a delay, it changes how cautious the "
    "app is about warning you. Someone extra cautious could say 'show me "
    "delayed for anything over 30% risk' — basically, 'I'd rather get some "
    "false alarms than miss a real delay.' Someone else could say 'only tell "
    "me delayed if you're over 70% sure' — meaning 'I don't want to be "
    "bothered unless it's a strong signal.' Same model, same 66-day "
    "definition underneath, just a different comfort level for how the "
    "results get labeled to you."
)

# --- Build a single-row feature vector matching training format ---
input_row = {col: 0 for col in features_to_keep}
input_row['GIS_LATITUDE'] = latitude
input_row['GIS_LONGITUDE'] = longitude

type_col = f"type_{approval_type}"
if type_col in input_row:
    input_row[type_col] = 1

month_col = f"month_{submit_month}"
if month_col in input_row:
    input_row[month_col] = 1

X_input = pd.DataFrame([input_row])[features_to_keep]

def friendly_name(col):
    """Convert raw column names into readable labels for the chart."""
    if col.startswith("type_"):
        return "Permit Type: " + col.replace("type_", "")
    if col.startswith("month_"):
        month_num = int(col.replace("month_", ""))
        return "Submitted in " + MONTH_NAMES[month_num - 1]
    if col == "GIS_LATITUDE":
        return "Latitude"
    if col == "GIS_LONGITUDE":
        return "Longitude"
    return col

friendly_feature_names = [friendly_name(col) for col in X_input.columns]

# Reference: overall delay rate across the training data (2024), for benchmarking
OVERALL_DELAY_RATE = 0.249  # 75th percentile cutoff by construction, ~25% of permits

# --- Predict ---
if st.button("Predict"):
    probability = model.predict_proba(X_input)[0][1]
    prediction = int(probability >= sensitivity)

    st.subheader("Prediction")
    if prediction == 1:
        st.error(f"Likely DELAYED — {probability:.1%} estimated probability")
    else:
        st.success(f"Likely ON TIME — {probability:.1%} estimated delay probability")

    if probability > OVERALL_DELAY_RATE:
        st.caption(
            f"For reference, the average delay rate across all San Diego permits "
            f"in 2024 was about {OVERALL_DELAY_RATE:.0%}. This permit's estimated "
            f"risk is above that average."
        )
    else:
        st.caption(
            f"For reference, the average delay rate across all San Diego permits "
            f"in 2024 was about {OVERALL_DELAY_RATE:.0%}. This permit's estimated "
            f"risk is at or below that average."
        )

    # --- SHAP explanation for this specific prediction ---
    st.subheader("Why this prediction?")
    st.write(
        "The chart below shows the features that most influenced this specific "
        "prediction, ranked from strongest to weakest."
    )

    # Compact legend, icons instead of a paragraph
    legend_col1, legend_col2 = st.columns(2)
    with legend_col1:
        st.markdown("🔴 **Pushed toward DELAYED**")
    with legend_col2:
        st.markdown("🔵 **Pushed toward ON TIME**")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_input)[0]

    # Take the top 10 features by absolute impact, sorted for the chart
    top_n = 10
    order = np.argsort(np.abs(shap_values))[::-1][:top_n]
    top_values = shap_values[order]
    top_labels = [friendly_feature_names[i] for i in order]
    colors = ['#d62728' if v > 0 else '#1f77b4' for v in top_values]

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(top_labels))
    ax.barh(y_pos, top_values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_labels)
    ax.invert_yaxis()  # strongest feature at the top
    ax.set_xlabel("Impact on prediction (SHAP value)")
    ax.axvline(0, color='black', linewidth=0.8)
    st.pyplot(fig)