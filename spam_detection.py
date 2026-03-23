import pandas as pd
import numpy as np
import re
import string

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib

# ---------------------------
# 1. Load Dataset
# ---------------------------
# Use your dataset file name (example: spam.csv)
df = pd.read_csv("spam.csv", encoding='latin-1')

# Keep only required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ---------------------------
# 2. Text Cleaning Function
# ---------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = text.strip()
    return text

df['message'] = df['message'].apply(clean_text)

# ---------------------------
# 3. Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# ---------------------------
# 4. Create Pipeline
# ---------------------------
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('classifier', MultinomialNB())
])

# ---------------------------
# 5. Train Model
# ---------------------------
model.fit(X_train, y_train)

# ---------------------------
# 6. Predictions
# ---------------------------
y_pred = model.predict(X_test)

# ---------------------------
# 7. Evaluation
# ---------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------------------
# 8. Save Model
# ---------------------------
joblib.dump(model, "spam_model.pkl")
print("\nModel saved as spam_model.pkl")

# ---------------------------
# 9. Predict New Message
# ---------------------------
def predict_spam(message):
    message = clean_text(message)
    result = model.predict([message])[0]
    return "SPAM" if result == 1 else "HAM"

# ---------------------------
# 10. Test Prediction
# ---------------------------
msg = input("\nEnter a message to check: ")
print("Prediction:", predict_spam(msg))
