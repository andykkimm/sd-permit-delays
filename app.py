import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# --- Load saved model and reference data ---
model = joblib.load('model.pkl')
features_to_keep = joblib.load('features_to_keep.pkl')
top_15_types = joblib.load('top_15_types.pkl')
cutoff = joblib.load('cutoff.pkl')

st.title("San Diego Permit Delay Predictor")
st.write(
    "Predicts whether a building permit approval is likely to be delayed "
    f"(defined as taking more than {int(cutoff)} days), based on permit type, "
    "submission month, and location."
)

# --- Inputs (now in the main body, not the sidebar) ---
st.header("Permit Details")
st.write("Enter a hypothetical permit's details below to see a live prediction.")

approval_type = st.selectbox("Approval Type", list(top_15_types) + ["Other"])
submit_month = st.selectbox("Submission Month", list(range(1, 13)))
latitude = st.number_input("Latitude", value=32.7157, format="%.4f")
longitude = st.number_input("Longitude", value=-117.1611, format="%.4f")

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

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

def friendly_name(col):
    """Convert raw column names into readable labels for the SHAP chart."""
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
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0][1]

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
        "The chart below shows what pushed this specific prediction toward "
        "**delayed** (red bars, pointing right) or **on time** (blue bars, "
        "pointing left). Bars are ordered by how much impact they had, "
        "with the biggest driver at the top. The starting point on the left "
        "is the average outcome across all permits; the ending point on the "
        "right is this permit's final predicted score."
    )
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_input)

    fig, ax = plt.subplots()
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_input.iloc[0],
            feature_names=friendly_feature_names
        ),
        show=False
    )
    st.pyplot(fig)