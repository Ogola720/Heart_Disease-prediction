import os, joblib, numpy as np, pandas as pd
import shap

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
FEATURES = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]

X = pd.DataFrame([[63,1,3,145,233,1,0,150,0,2.3,0,0,1]], columns=FEATURES)

# RF
try:
    rf = joblib.load(os.path.join(MODELS_DIR, 'random_forest.joblib'))
    print('RF loaded, type:', type(rf))
    probs = rf.predict_proba(X)[:,1]
    print('RF probs ok', probs)
    try:
        expl = shap.TreeExplainer(rf)
        sv = expl.shap_values(X)
        print('RF shap ok', type(sv))
    except Exception as e:
        print('RF shap error:')
        import traceback
        traceback.print_exc()
except Exception as e:
    print('RF load/predict error:')
    import traceback
    traceback.print_exc()

# NN
try:
    import tensorflow as tf
    nn_path = os.path.join(MODELS_DIR, 'neural_network.keras')
    nn = tf.keras.models.load_model(nn_path)
    print('NN loaded, type:', type(nn))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
    Xs = scaler.transform(X)
    probs = nn.predict(Xs).ravel()
    print('NN probs ok', probs)
    try:
        # set a small background
        bg = Xs if Xs.shape[0] > 1 else np.tile(Xs, (10,1))
        expl = shap.DeepExplainer(nn, bg)
        sv = expl.shap_values(Xs)
        print('NN shap ok', type(sv))
    except Exception as e:
        print('NN shap error:')
        import traceback
        traceback.print_exc()
except Exception as e:
    print('NN load/predict error:')
    import traceback
    traceback.print_exc()
