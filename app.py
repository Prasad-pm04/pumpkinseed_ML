import streamlit as st
import pandas as pd
import numpy as np
import pickle
from io import BytesIO

st.set_page_config(page_title="Pumpkin Seed Classifier", layout="centered")
st.title("🎃 Pumpkin Seed Classifier BY TrainwithPrasadM — Web App")
st.write("Load your trained model (Pickle) or use the model file available at `/mnt/data/Pumpkin_seed_model.pkl` and predict seed classes from measured features.")

# Feature list (must match the order used for training)
FEATURES = [
    "Area",
    "Perimeter",
    "Major_Axis_Length",
    "Minor_Axis_Length",
    "Convex_Area",
    "Equiv_Diameter",
    "Eccentricity",
    "Solidity",
    "Extent",
    "Roundness",
    "Aspect_Ration",
    "Compactness",
]

# Try to load model from known path first
MODEL_PATH = "/mnt/data/Pumpkin_seed_model.pkl"
model = None
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        st.sidebar.success(f"Loaded model from {MODEL_PATH}")
except Exception as e:
    st.sidebar.info("No model found at /mnt/data/Pumpkin_seed_model.pkl")

# Allow user to upload a model if not found or if they prefer their own
uploaded_model = st.sidebar.file_uploader("Upload a Pickle (.pkl) model (optional)", type=["pkl"])
if uploaded_model is not None:
    try:
        model = pickle.load(uploaded_model)
        st.sidebar.success("Model uploaded and loaded successfully")
    except Exception as e:
        st.sidebar.error("Uploaded file could not be loaded as a pickle model. Make sure it's a valid sklearn-compatible pickle.")

if model is None:
    st.warning("No model is loaded. Please upload a .pkl model or place it at /mnt/data/Pumpkin_seed_model.pkl on the server.")

st.markdown("---")

st.header("Single sample prediction")
cols = st.columns(2)
# default values taken from user's sample training row (first row)
defaults = {
    "Area": 56276.0,
    "Perimeter": 888.242,
    "Major_Axis_Length": 326.1485,
    "Minor_Axis_Length": 220.2388,
    "Convex_Area": 56831.0,
    "Equiv_Diameter": 267.6805,
    "Eccentricity": 0.7376,
    "Solidity": 0.9902,
    "Extent": 0.7453,
    "Roundness": 0.8963,
    "Aspect_Ration": 1.4809,
    "Compactness": 0.8207,
}

inputs = {}
for i, feat in enumerate(FEATURES):
    c = cols[i % 2]
    val = c.number_input(feat, value=float(defaults.get(feat, 0.0)), format="%.6f")
    inputs[feat] = val

if st.button("Predict single sample"):
    if model is None:
        st.error("No model loaded. Upload or place model at /mnt/data/Pumpkin_seed_model.pkl")
    else:
        X = pd.DataFrame([inputs], columns=FEATURES)
        try:
            pred = model.predict(X)
            st.success(f"Predicted class: {pred[0]}")
            # show probabilities if available
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                proba_df = pd.DataFrame(proba, columns=[f"class_{i}" for i in range(proba.shape[1])])
                st.table(proba_df)
        except Exception as e:
            st.error(f"Model prediction failed: {e}")

st.markdown("---")

st.header("Batch prediction from CSV")
st.write("Upload a CSV with the exact feature columns (order not important) and get predictions for each row.")
uploaded_csv = st.file_uploader("Upload CSV for batch prediction", type=["csv"] , key="csv")

if uploaded_csv is not None:
    try:
        df = pd.read_csv(uploaded_csv)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            st.error(f"CSV is missing these required columns: {missing}")
        else:
            if st.button("Run batch prediction"):
                if model is None:
                    st.error("No model loaded. Upload or place model at /mnt/data/Pumpkin_seed_model.pkl")
                else:
                    X = df[FEATURES]
                    try:
                        preds = model.predict(X)
                        out = df.copy()
                        out["predicted_class"] = preds
                        # add probabilities if available
                        if hasattr(model, "predict_proba"):
                            proba = model.predict_proba(X)
                            for i in range(proba.shape[1]):
                                out[f"prob_class_{i}"] = proba[:, i]
                        st.success("Batch prediction completed")
                        st.dataframe(out.head())

                        # provide download
                        towrite = BytesIO()
                        out.to_csv(towrite, index=False)
                        towrite.seek(0)
                        st.download_button("Download predictions as CSV", data=towrite, file_name="pumpkin_predictions.csv", mime="text/csv")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

st.markdown("---")

st.header("Notes & Tips")
st.markdown(
    """
- Make sure the model was trained with the **exact same feature names** listed in the app. If your training pipeline used a different column order or feature names, either adapt the app or retrain.
- If the model expects scaled features (StandardScaler, MinMax), the model pickle should already include any preprocessing (Pipeline). Prefer saving a `Pipeline` that includes preprocessing + estimator.
- To deploy on a server (Render/Heroku/etc.):
  1. Include this file and `Pumpkin_seed_model.pkl` in the project repository (or upload the model at startup).
  2. Add a `requirements.txt` listing `streamlit`, `scikit-learn`, `pandas`, `numpy` and other libs used.
  3. Run with `streamlit run pumpkin_seed_streamlit_app.py --server.port $PORT` (platforms usually give `$PORT`).

"""
)

st.write("If you'd like, I can also generate a simple `requirements.txt`, a `Dockerfile`, or a Flask alternative for deployment.")
