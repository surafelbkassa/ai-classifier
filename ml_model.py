import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Training data (expand this!)
emails = [
    ("Free money click here", "spam"),
    ("You won a million dollars", "spam"),
    ("Buy now limited offer", "spam"),
    ("Meeting at 3pm urgent", "work"),
    ("Deadline tomorrow please finish", "work"),
    ("Client called need response ASAP", "work"),
    ("Hey how are you doing", "personal"),
    ("Let's get coffee sometime", "personal"),
    ("Happy birthday friend", "personal"),
]

texts, labels = zip(*emails)

# Vectorize and train
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
model = LogisticRegression()
model.fit(X, labels)

# Save
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("✅ ML model trained with Logistic Regression")
