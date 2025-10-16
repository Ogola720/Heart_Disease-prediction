import joblib, os
m = joblib.load(os.path.join(os.path.dirname(__file__), 'models', 'random_forest.joblib'))
print(type(m))
try:
    print(repr(m))
except Exception:
    print('repr failed')

# if pipeline-like try to list attributes
if hasattr(m, 'named_steps'):
    print('named_steps:', m.named_steps.keys())
if hasattr(m, 'estimators_'):
    print('estimators_ attr exists')
if hasattr(m, 'feature_importances_'):
    print('feature_importances_ exists')
