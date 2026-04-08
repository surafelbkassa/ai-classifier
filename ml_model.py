import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Training data (expand this!)
# emails = [
#     ("Free money click here", "spam"),
#     ("You won a million dollars", "spam"),
#     ("Buy now limited offer", "spam"),
#     ("Meeting at 3pm urgent", "work"),
#     ("Deadline tomorrow please finish", "work"),
#     ("Client called need response ASAP", "work"),
#     ("Hey how are you doing", "personal"),
#     ("Let's get coffee sometime", "personal"),
#     ("Happy birthday friend", "personal"),
emails = [
    # Spam (10 examples)
    ("Free money click here", "spam"),
    ("You won a million dollars", "spam"),
    ("Buy now limited offer", "spam"),
    ("Congratulations you are a winner", "spam"),
    ("Lowest price ever sale", "spam"),
    ("Claim your prize today", "spam"),
    ("100% free gift card", "spam"),
    ("Investment opportunity double your money", "spam"),
    ("You've been selected", "spam"),
    ("Act now limited time", "spam"),
    
    # Work / Urgent (10 examples)
    ("Meeting at 3pm urgent", "work"),
    ("Deadline tomorrow please finish", "work"),
    ("Client called need response ASAP", "work"),
    ("Quarterly report due Friday", "work"),
    ("Server down fix immediately", "work"),
    ("Please review this document by EOD", "work"),
    ("Schedule changed team meeting", "work"),
    ("Urgent: production issue", "work"),
    ("Your task is overdue", "work"),
    ("Manager requested update", "work"),
    
    # Personal (10 examples)
    ("Hey how are you doing", "personal"),
    ("Let's get coffee sometime", "personal"),
    ("Happy birthday friend", "personal"),
    ("See you at the party", "personal"),
    ("Thanks for yesterday", "personal"),
    ("Miss you let's catch up", "personal"),
    ("Dinner tonight?", "personal"),
    ("Weekend plans?", "personal"),
    ("Great talking to you", "personal"),
    ("Thinking of you", "personal"),
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
