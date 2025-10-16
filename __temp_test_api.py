import requests, json
payload = {"inputs":[{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}]}
for model in ['lr','rf','nn']:
    url = f'http://127.0.0.1:8000/predict?model={model}'
    try:
        r = requests.post(url, json=payload, timeout=10)
        print(model, r.status_code)
        try:
            print(json.dumps(r.json(), indent=2))
        except Exception:
            print(r.text)
    except Exception as e:
        print(model, 'request error:', repr(e))      