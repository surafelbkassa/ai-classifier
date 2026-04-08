import joblib
import sys
import json

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

text = sys.argv[1]
X = vectorizer.transform([text])
pred = model.predict(X)[0]

print(json.dumps({"category": pred}))
