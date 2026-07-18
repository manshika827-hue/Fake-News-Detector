import joblib
import os
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = []

    for word in text.split():
        if word not in stop_words:
            words.append(stemmer.stem(word))

    return " ".join(words)


# Get project root folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Model paths
model_path = os.path.join(BASE_DIR, "03_models", "fake_news_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "03_models", "tfidf_vectorizer.pkl")


print("Model path:", model_path)
print("Vectorizer path:", vectorizer_path)


# Load model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)


# User input
news = input("Enter news text: ")


# Transform text
cleaned_news = clean_text(news)
news_vector = vectorizer.transform([cleaned_news])


# Predict
prediction = model.predict(news_vector)

print("Raw prediction:", prediction[0])

if prediction[0] == 0:
    print("Prediction: Fake News")
else:
    print("Prediction: Real News")