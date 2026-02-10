import pandas as pd
import numpy as np
import re
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ============================
# Load Dataset
# ============================

# Download from Kaggle/UCI and place in same folder
# File name: spam.csv

df = pd.read_csv("spam.csv", encoding="latin-1")


# ============================
# Clean Dataset
# ============================

# Keep only useful columns
df = df[["v1", "v2"]]
df.columns = ["label", "text"]

# Convert labels to binary
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})


# ============================
# Text Cleaning Function
# ============================

def clean_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove special characters
    text = re.sub(r"[^a-z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["text"] = df["text"].apply(clean_text)


# ============================
# Train-Test Split
# ============================

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================
# TF-IDF Vectorization
# ============================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# ============================
# Model + Fine-Tuning
# ============================

model = LogisticRegression(max_iter=1000)

params = {
    "C": [0.1, 1, 5, 10],
    "solver": ["liblinear", "lbfgs"]
}

grid = GridSearchCV(
    model,
    params,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

print("Training & Fine-Tuning...")

grid.fit(X_train_vec, y_train)

best_model = grid.best_estimator_

print("Best Parameters:", grid.best_params_)


# ============================
# Evaluation
# ============================

y_pred = best_model.predict(X_test_vec)

acc = accuracy_score(y_test, y_pred)

print("\nAccuracy:", acc)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ============================
# Save Model
# ============================

joblib.dump(best_model, "spam_model.pkl")
joblib.dump(vectorizer, "spam_vectorizer.pkl")

print("\nModel Saved Successfully!")
