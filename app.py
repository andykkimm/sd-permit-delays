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

# --- Predict ---
if st.button("Predict"):
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0][1]

    st.subheader("Prediction")
    if prediction == 1:
        st.error(f"Likely DELAYED — {probability:.1%} estimated probability")
    else:
        st.success(f"Likely ON TIME — {probability:.1%} estimated delay probability")

    # --- SHAP explanation for this specific prediction ---
    st.subheader("Why this prediction?")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_input)

    fig, ax = plt.subplots()
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_input.iloc[0],
            feature_names=X_input.columns.tolist()
        ),
        show=False
    )
    st.pyplot(fig)