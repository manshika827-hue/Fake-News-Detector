import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load cleaned dataset
news = pd.read_csv(
    r"C:\Users\Manshika\OneDrive\Desktop\Fake-News-Detector\01_dataset\clean_news.csv"
)
# Remove rows with missing values
news = news.dropna(subset=["clean_text"])

# Convert clean_text to string
news["clean_text"] = news["clean_text"].astype(str)
# Features and Labels
X = news["clean_text"]
y = news["label"]

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train, y_train)

# Prediction
 # Prediction
y_pred = model.predict(X_test)

print("\nFirst 10 Actual labels:")
print(y_test.iloc[:10].tolist())

print("\nFirst 10 Predicted labels:")
print(y_pred[:10].tolist())

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# Save Model
joblib.dump(
    model,
    r"C:\Users\Manshika\OneDrive\Desktop\Fake-News-Detector\03_models\fake_news_model.pkl"
)

joblib.dump(
    vectorizer,
    r"C:\Users\Manshika\OneDrive\Desktop\Fake-News-Detector\03_models\tfidf_vectorizer.pkl"
)

print("\nModel Saved Successfully!")