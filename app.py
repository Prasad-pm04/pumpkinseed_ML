import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

st.set_page_config(page_title="Pumpkin Seed Classifier", page_icon="🎃", layout="centered")

st.title("🎃 Pumpkin Seed Classifier")
st.write("Upload features or enter them manually to predict the seed class with your trained model.")

MODEL_PATH = Path("Pumpkin_seed_model.pkl")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Model file not found: {MODEL_PATH}. Please place your 'Pumpkin_seed_model.pkl' in the same folder as this app.")
        return None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

FEATURES = ["Area", "Perimeter", "Major_Axis_Length", "Minor_Axis_Length", "Convex_Area", "Equiv_Diameter", "Eccentricity", "Solidity", "Extent", "Roundness", "Aspect_Ration", "Compactness"]

def predict(df: pd.DataFrame):
    # Keep only the expected features in the right order
    X = df[[*FEATURES]].copy()
    # Convert to numeric and handle issues
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    if X.isna().any().any():
        st.warning("Some values could not be parsed to numbers. They are NaN now and will stop prediction.")
        raise ValueError("NaN values in input features")
    # Predict
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        preds = (proba[:, 1] >= 0.5).astype(int) if proba.shape[1] == 2 else model.predict(X)
        return preds, proba
    else:
        preds = model.predict(X)
        proba = None
        return preds, proba

with st.sidebar:
    st.header("About")
    st.markdown(
        "This app loads **Pumpkin_seed_model.pkl** and predicts the **Class** (e.g., 0/1).\n"
        "Columns expected:\n"
        f"- " + "\n- ".join(FEATURES)
    )
    st.markdown("---")
    st.markdown("**Tip:** Use the *Sample CSV* below to test quickly.")

tab1, tab2 = st.tabs(["📥 CSV / Batch", "✍️ Manual Entry"])

with tab1:
    st.subheader("Batch prediction via CSV")
    st.write("Upload a CSV with these columns (order can vary):")
    st.code(",".join(FEATURES), language="text")
    sample = pd.DataFrame([{
        "Area": 56276, "Perimeter": 888.242, "Major_Axis_Length": 326.1485, "Minor_Axis_Length": 220.2388,
        "Convex_Area": 56831, "Equiv_Diameter": 267.6805, "Eccentricity": 0.7376, "Solidity": 0.9902,
        "Extent": 0.7453, "Roundness": 0.8963, "Aspect_Ration": 1.4809, "Compactness": 0.8207
    }])
    st.download_button("Download sample_input.csv", sample.to_csv(index=False).encode("utf-8"), "sample_input.csv", "text/csv")

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None and model is not None:
        try:
            df = pd.read_csv(file)
            missing = [c for c in FEATURES if c not in df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                preds, proba = predict(df)
                out = df.copy()
                out["Predicted_Class"] = preds
                if proba is not None and proba.ndim == 2 and proba.shape[1] >= 2:
                    out["Probability_Class_1"] = proba[:, 1]
                st.success("Predictions complete.")
                st.dataframe(out.head(50))
                st.download_button("Download predictions CSV", out.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")
        except Exception as e:
            st.exception(e)

with tab2:
    st.subheader("Manual single prediction")
    st.write("Enter feature values:")
    cols = st.columns(3)
    values = {}
    defaults = {
        "Area": 56276, "Perimeter": 888.242, "Major_Axis_Length": 326.1485, "Minor_Axis_Length": 220.2388,
        "Convex_Area": 56831, "Equiv_Diameter": 267.6805, "Eccentricity": 0.7376, "Solidity": 0.9902,
        "Extent": 0.7453, "Roundness": 0.8963, "Aspect_Ration": 1.4809, "Compactness": 0.8207
    }
    for i, feat in enumerate(FEATURES):
        with cols[i % 3]:
            # Use number_input with float default
            values[feat] = st.number_input(feat, value=float(defaults.get(feat, 0.0)))

    if st.button("Predict", type="primary") and model is not None:
        row = pd.DataFrame([values], columns=FEATURES)
        try:
            preds, proba = predict(row)
            st.success(f"Predicted Class: {int(preds[0])}")
            if proba is not None and proba.ndim == 2 and proba.shape[1] >= 2:
                st.write(f"Probability of Class 1: {float(proba[0,1]):.4f}")
        except Exception as e:
            st.exception(e)

st.markdown("---")
st.caption("Built with Streamlit. Place your **Pumpkin_seed_model.pkl** next to this script and run: `streamlit run app.py`.")
