import streamlit as st 
import pandas as pd
import requests
import matplotlib.pyplot as plt

API_DEFAULT = "http://localhost:8000/predict"
FEATURES = ["age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal"]

st.set_page_config(page_title="Heart Disease Demo", layout="centered")
st.title("❤️ Heart Disease Prediction with SHAP Explanations")

st.sidebar.header("Settings")
api_url = st.sidebar.text_input("API endpoint", API_DEFAULT)
model_choice = st.sidebar.selectbox("Model", options=["rf", "lr", "nn"], index=0)

# -----------------
# Patient input form
# -----------------
st.header("Patient Input")
with st.form("patient_form"):
    data = {}
    for feat in FEATURES:
        if feat in ["sex", "fbs", "exang"]:
            # binary categorical features
            data[feat] = st.selectbox(feat, options=[0,1], index=0)
        else:
            data[feat] = st.number_input(feat, value=0.0)
    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {"inputs": [data]}
    try:
        resp = requests.post(api_url, params={"model": model_choice}, json=payload, timeout=20)
        resp.raise_for_status()
        data_out = resp.json()
        
        result = data_out["results"][0]
        pred = result["prediction"]
        prob = result["probability"]

        st.subheader("Prediction Result")
        st.success(f"Prediction: **{'Heart Disease' if pred==1 else 'No Heart Disease'}** "
                   f"(probability: {prob:.3f})")

        # -----------------
        # SHAP explanation
        # -----------------
        if "shap_values" in result and result["shap_values"] is not None:
            shap_values = result["shap_values"]
            shap_series = pd.Series(shap_values)

            # Bar chart
            st.subheader("Feature Contributions (SHAP values)")
            st.bar_chart(shap_series)

            # Theoretical explanation
            top_pos = shap_series.sort_values(ascending=False).head(3)
            top_neg = shap_series.sort_values().head(3)

            explanation = []
            if len(top_pos) > 0:
                explanation.append("Factors pushing the prediction **towards Heart Disease**:")
                for f, v in top_pos.items():
                    explanation.append(f"• {f} (+{v:.3f})")
            if len(top_neg) > 0:
                explanation.append("Factors pushing the prediction **away from Heart Disease**:")
                for f, v in top_neg.items():
                    explanation.append(f"• {f} ({v:.3f})")

            st.subheader("Interpretation")
            st.markdown("\n".join(explanation))

        else:
            st.warning("⚠️ No SHAP explanation available for this prediction.")

    except Exception as e:
        st.error(f"Request failed: {e}")
