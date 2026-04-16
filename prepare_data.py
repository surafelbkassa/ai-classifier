# prepare_data.py
# Run this ONCE on VS Code to download and prepare your dataset

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Set your Kaggle token (you'll add this on VS Code)
# os.environ["KAGGLE_API_TOKEN"] = "your_token_here"

import kagglehub
from kagglehub import KaggleDatasetAdapter

print("Loading Enron dataset from Kaggle...")

# Load the dataset
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "wcukierski/enron-email-dataset",
    "enron.csv",
)

print(f"Loaded {len(df)} emails")

# Simple labeling based on keywords
def simple_label(file_path, message):
    file_lower = str(file_path).lower()
    msg_lower = str(message).lower()
    
    if 'spam' in file_lower:
        return 'spam'
    
    work_words = ['meeting', 'deadline', 'report', 'client', 'urgent', 'task']
    personal_words = ['happy', 'birthday', 'coffee', 'dinner', 'weekend', 'thanks']
    
    work_count = sum(1 for w in work_words if w in msg_lower)
    personal_count = sum(1 for w in personal_words if w in msg_lower)
    
    if work_count > personal_count:
        return 'work'
    elif personal_count > work_count:
        return 'personal'
    else:
        return 'work'  # default

print("Adding labels...")
df['label'] = df.apply(lambda row: simple_label(row['file'], row['message']), axis=1)

# Take 500 samples per category for balance
balanced_df = pd.DataFrame()
for label in ['spam', 'work', 'personal']:
    subset = df[df['label'] == label]
    if len(subset) > 500:
        subset = subset.sample(n=500, random_state=42)
    balanced_df = pd.concat([balanced_df, subset])

print(f"Balanced dataset: {len(balanced_df)} emails")
print(balanced_df['label'].value_counts())

# Save to CSV
balanced_df[['message', 'label']].to_csv('emails_1500.csv', index=False)
print("✅ Saved to emails_1500.csv")

# Train model immediately
print("\nTraining model...")
X = balanced_df['message'].values
y = balanced_df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

X_test_vec = vectorizer.transform(X_test)
accuracy = model.score(X_test_vec, y_test)
print(f"✅ Model accuracy: {accuracy:.2%}")

# Save model
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("✅ Model saved to model.pkl and vectorizer.pkl")