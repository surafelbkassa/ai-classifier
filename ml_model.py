# ml_model.py
# Run this AFTER load_enron_data.py

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the balanced dataset from the CSV file
print("Loading labeled email data...")
df = pd.read_csv('enron_balanced.csv')

print(f"Loaded {len(df)} emails")
print(f"Label distribution:")
print(df['label'].value_counts())

# Split into features (X) and target (y)
X = df['message'].values
y = df['label'].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {len(X_train)} emails")
print(f"Test set: {len(X_test)} emails")

# Vectorize and train
print("\nTraining model...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression(C=1.0, max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate
X_test_vec = vectorizer.transform(X_test)
accuracy = model.score(X_test_vec, y_test)
print(f"\n✅ Model accuracy on test set: {accuracy:.2%}")

# Save model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("\n✅ Model saved to 'model.pkl'")
print("✅ Vectorizer saved to 'vectorizer.pkl'")

# Test on a few examples
test_emails = [
    "urgent meeting at 3pm",
    "free money click here",
    "let's get coffee this weekend"
]

print("\n--- Testing on sample emails ---")
for email in test_emails:
    X_sample = vectorizer.transform([email])
    pred = model.predict(X_sample)[0]
    proba = model.predict_proba(X_sample)[0]
    confidence = max(proba)
    print(f"'{email}' → {pred} (confidence: {confidence:.1%})")