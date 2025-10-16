import requests

url = "http://127.0.0.1:8000/predict?model=rf"
payload = {
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
resp = requests.post(url, json=payload)
print(resp.json())
