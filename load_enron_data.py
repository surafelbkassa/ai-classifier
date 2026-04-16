# load_enron_data.py
# Run this ONCE to download and prepare your dataset
import os
os.environ["KAGGLE_API_TOKEN"] = "KGAT_bf2e13b42cbb8f7bdc1a9f43aa46a483"
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import re

print("Loading Enron dataset from Kaggle...")

# Load the Enron dataset
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "wcukierski/enron-email-dataset",
    "enron.csv",
)

print(f"Loaded {len(df)} emails")
print(f"Columns: {df.columns.tolist()}")

# The Enron dataset has 'file' and 'message' columns
# We need to create labels based on the file path
# Emails in folders with 'spam' are spam, others are 'work' or 'personal'

def create_label(file_path, message):
    """Create label based on file path and message content"""
    file_lower = str(file_path).lower()
    
    # Check if it's spam based on folder name
    if 'spam' in file_lower:
        return 'spam'
    
    # Check for work-related keywords
    work_keywords = ['meeting', 'deadline', 'report', 'project', 'client', 'urgent', 'task', 'manager', 'quarterly', 'server', 'production', 'eod', 'asap']
    personal_keywords = ['happy', 'birthday', 'coffee', 'dinner', 'weekend', 'thanks', 'miss', 'thinking', 'love', 'friend', 'family']
    
    message_lower = str(message).lower()
    
    # Count work vs personal keywords
    work_count = sum(1 for kw in work_keywords if kw in message_lower)
    personal_count = sum(1 for kw in personal_keywords if kw in message_lower)
    
    if work_count > personal_count:
        return 'work'
    elif personal_count > work_count:
        return 'personal'
    else:
        # Default based on folder name
        if 'inbox' in file_lower or 'sent' in file_lower:
            return 'personal'
        else:
            return 'work'

# Apply labeling (this may take a minute on large dataset)
print("Creating labels (this may take a moment)...")
df['label'] = df.apply(lambda row: create_label(row['file'], row['message']), axis=1)

# Show label distribution
print("\nLabel distribution:")
print(df['label'].value_counts())

# Filter to get balanced dataset (500-1000 examples)
# Let's take 300 of each category
balanced_df = pd.DataFrame()
for label in ['spam', 'work', 'personal']:
    label_df = df[df['label'] == label]
    if len(label_df) > 300:
        label_df = label_df.sample(n=300, random_state=42)
    balanced_df = pd.concat([balanced_df, label_df])

print(f"\nBalanced dataset: {len(balanced_df)} emails")
print(balanced_df['label'].value_counts())

# Save to CSV for reuse
balanced_df[['message', 'label']].to_csv('enron_balanced.csv', index=False)
print("\n✅ Saved to 'enron_balanced.csv'")