# Heart Disease Prediction

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![status](https://img.shields.io/badge/status-demo-yellow)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

Heart Disease Prediction — a small demo that exposes machine-learning models via a FastAPI REST API and provides a Streamlit UI for interactive predictions and SHAP-based explanations. The project includes a logistic regression pipeline, a random forest, and a TensorFlow neural network, plus example client/tests.

This repo is intended as a learning/demo project for model serving, interpretability (SHAP), and building a lightweight ML web demo.

## Features

- FastAPI backend (`predict.py`) exposing a `/predict` endpoint and `/health` status endpoint
- Streamlit UI (`streamlit_app.py`) for interactive prediction input and SHAP visualization
- Three models included under `models/` (if present): Random Forest, Logistic Regression pipeline, Neural Network
- SHAP explanations where supported
- Example client/test script: `test_api.py`

## Quick start (PowerShell)

1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Start the FastAPI server (terminal 1):

```powershell
uvicorn predict:app --reload --host 127.0.0.1 --port 8000
```

3. Start the Streamlit UI (terminal 2):

```powershell
. .\.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

4. (Optional) Run the example API client:

```powershell
python test_api.py
```

## API

- GET /health — returns server status and which models are loaded
- POST /predict?model=<rf|lr|nn> — accepts JSON payload of the form:

```json
{
  "inputs": [
    {
      "age": 63,
      "sex": 1,
      "cp": 3,
      "trestbps": 145,
      "chol": 233,
      "fbs": 1,
      "restecg": 0,
      "thalach": 150,
      "exang": 0,
      "oldpeak": 2.3,
      "slope": 0,
      "ca": 0,
      "thal": 1
    }
  ]
}
```

Response includes predictions, probabilities, and an optional `shap_values` map when available.

### Example: curl (health)

```powershell
curl http://127.0.0.1:8000/health
```

### Example: Python requests (predict)

```python
import requests
payload = {"inputs":[{...}]}
resp = requests.post('http://127.0.0.1:8000/predict?model=rf', json=payload)
print(resp.status_code, resp.json())
```

## Models

Place model files (if available) in the `models/` directory:

- `random_forest.joblib` — RandomForestClassifier or a Pipeline
- `logistic_pipeline.joblib` — sklearn Pipeline containing scaler + logistic regression
- `neural_network.keras` — TensorFlow Keras model
- `scaler.joblib` — scaler for neural network inputs

The app attempts to load these on startup. Use the `/health` endpoint to verify which models were successfully loaded.

## Troubleshooting

- Connection refused (Streamlit shows HTTPConnectionPool... Failed to establish a new connection)
  - Ensure the FastAPI server is running (see Quick start). Start uvicorn on `127.0.0.1:8000`.

- `500 Internal Server Error` for `rf` or `nn` predictions
  - SHAP explainers can be fragile depending on how the model was saved and package versions.
  - Run the debugging helpers to inspect models and reproduce the error outside of FastAPI:

```powershell
python __inspect_rf.py    # prints type/structure of random_forest.joblib
python __debug_rf_nn.py   # attempts to load RF/NN and run SHAP locally, prints tracebacks
```

- If the random forest was saved as a `Pipeline` (contains `named_steps`), SHAP needs the final estimator and the correctly transformed inputs. The API attempts to detect this, but re-saving the final estimator separately is a reliable fix.

- For neural networks, if SHAP fails, the API will still return predictions; SHAP may be disabled when incompatible.

## Development notes

- The repository uses `uvicorn` + `fastapi` for the API and `streamlit` for the UI. SHAP is used for explanations and may require compatible versions of `shap` and `tensorflow`.
- Tests are minimal — `test_api.py` demonstrates a request payload.

## Contributing

Contributions are welcome. Open an issue or PR for bug fixes and improvements. If you change model serialization format, update `predict.py` accordingly to ensure SHAP is given the estimator and transformed inputs it expects.

## License

This project is provided under the MIT license.

---
