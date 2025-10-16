from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
import joblib
import tensorflow as tf
import pandas as pd
import os
import shap
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Model directory
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Feature ordering
FEATURES = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal"
]

app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease using Logistic Regression, Random Forest, and Neural Network with SHAP explanations (safe)."
)

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Patient(BaseModel):
    age: float = Field(..., example=63)
    sex: int = Field(..., example=1)
    cp: int = Field(..., example=3)
    trestbps: float = Field(..., example=145)
    chol: float = Field(..., example=233)
    fbs: int = Field(..., example=1)
    restecg: int = Field(..., example=0)
    thalach: float = Field(..., example=150)
    exang: int = Field(..., example=0)
    oldpeak: float = Field(..., example=2.3)
    slope: int = Field(..., example=0)
    ca: int = Field(..., example=0)
    thal: int = Field(..., example=1)

class PredictRequest(BaseModel):
    inputs: List[Patient]

@app.on_event("startup")
def load_models():
    """Load models on startup"""
    def _load(path):
        if not os.path.exists(path):
            return None
        try:
            return joblib.load(path)
        except Exception:
            return None

    app.state.random_forest = _load(os.path.join(MODELS_DIR, "random_forest.joblib"))
    app.state.logistic = _load(os.path.join(MODELS_DIR, "logistic_pipeline.joblib"))
    app.state.scaler = _load(os.path.join(MODELS_DIR, "scaler.joblib"))

    nn_path = os.path.join(MODELS_DIR, "neural_network.keras")
    if os.path.exists(nn_path):
        try:
            app.state.nn = tf.keras.models.load_model(nn_path)
        except Exception:
            app.state.nn = None
    else:
        app.state.nn = None

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": {
        "random_forest": bool(app.state.random_forest),
        "logistic": bool(app.state.logistic),
        "neural_network": bool(app.state.nn),
        "scaler": bool(app.state.scaler)
    }}

def _df_from_inputs(inputs: List[Patient]) -> pd.DataFrame:
    rows = []
    for p in inputs:
        d = p.dict()
        row = [d.get(f) for f in FEATURES]
        rows.append(row)
    return pd.DataFrame(rows, columns=FEATURES)

@app.post("/predict")
def predict(req: PredictRequest, model: Optional[Literal['rf','lr','nn']] = Query('rf')):
    X = _df_from_inputs(req.inputs)

    if X.isnull().any().any():
        raise HTTPException(status_code=422, detail="Input contains nulls or missing values.")

    try:
        model = model.lower()
        shap_values = None

        # ---------------- Random Forest ----------------
        if model == 'rf':
            clf = app.state.random_forest
            if clf is None:
                raise HTTPException(status_code=503, detail="Random Forest model not available on server.")

            # predict_proba works whether clf is an estimator or a pipeline
            probs = clf.predict_proba(X)[:, 1]
            preds = (probs > 0.5).astype(int)

            # For SHAP: if a pipeline was saved, extract the final estimator and
            # also obtain the transformed inputs that the estimator expects.
            try:
                estimator_for_shap = clf
                X_for_shap = X
                if hasattr(clf, 'named_steps'):
                    # pipeline-like object
                    steps = list(clf.named_steps.items())
                    estimator_for_shap = steps[-1][1]
                    # if there's a scaler in the pipeline, apply it to X for SHAP
                    if 'scaler' in clf.named_steps:
                        try:
                            X_for_shap = clf.named_steps['scaler'].transform(X)
                        except Exception:
                            X_for_shap = X

                explainer = shap.TreeExplainer(estimator_for_shap)
                sv = explainer.shap_values(X_for_shap)
                shap_values = sv[1] if isinstance(sv, list) else sv
            except Exception as e:
                print(f"⚠️ SHAP (RF) failed: {e}")
                shap_values = None

        # ---------------- Logistic Regression ----------------
        elif model == 'lr':
            clf = app.state.logistic
            probs = clf.predict_proba(X)[:, 1]
            preds = (probs > 0.5).astype(int)

            try:
                scaler = clf.named_steps['scaler']
                X_scaled = scaler.transform(X)
                explainer = shap.LinearExplainer(clf.named_steps['lr'], X_scaled, feature_perturbation="interventional")
                sv = explainer.shap_values(X_scaled)
                shap_values = sv[0] if isinstance(sv, list) else sv
            except Exception as e:
                print(f"⚠️ SHAP (LR) failed: {e}")
                shap_values = None

        # ---------------- Neural Network ----------------
        elif model == 'nn':
            scaler = app.state.scaler
            clf = app.state.nn
            if clf is None or scaler is None:
                raise HTTPException(status_code=503, detail="Neural network or scaler not available on server.")

            # scale inputs and run prediction
            try:
                X_scaled = scaler.transform(X)
            except Exception as e:
                print(f"⚠️ Scaler transform failed: {e}")
                raise HTTPException(status_code=500, detail=f"Scaler transform failed: {e}")

            try:
                probs = clf.predict(X_scaled).ravel()
                preds = (probs > 0.5).astype(int)
            except Exception as e:
                print(f"⚠️ NN prediction failed: {e}")
                raise HTTPException(status_code=500, detail=f"Neural network prediction failed: {e}")

            # SHAP for NN can be fragile depending on TF/shap versions; try and fall back
            try:
                if X_scaled.shape[0] > 1:
                    background = X_scaled[np.random.choice(X_scaled.shape[0], min(50, X_scaled.shape[0]), replace=False)]
                else:
                    background = np.tile(X_scaled, (10, 1))

                # prefer DeepExplainer when supported; fallback to no explanation
                try:
                    explainer = shap.DeepExplainer(clf, background)
                    sv = explainer.shap_values(X_scaled)
                    shap_values = sv[0] if isinstance(sv, list) else sv
                except Exception as e:
                    print(f"⚠️ SHAP (NN) DeepExplainer failed: {e}")
                    shap_values = None
            except Exception as e:
                print(f"⚠️ SHAP (NN) setup failed: {e}")
                shap_values = None

        else:
            raise HTTPException(status_code=400, detail="Unknown model (choose 'rf', 'lr', 'nn').")

        # ---------------- Build Response ----------------
        results = []
        for i in range(len(preds)):
            result = {
                "prediction": int(preds[i]),
                "probability": float(probs[i]),
                "shap_values": dict(zip(FEATURES, shap_values[i])) if shap_values is not None else None
            }
            results.append(result)

        return {"model": model, "n": len(results), "results": results}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        # print traceback to server logs for debugging
        print(tb)
        # return error detail to client for debugging (development only)
        raise HTTPException(status_code=500, detail={"error": str(e), "trace": tb})
