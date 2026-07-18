import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords")

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
# Project path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


model_path = os.path.join(
    BASE_DIR,
    "03_models",
    "fake_news_model.pkl"
)

vectorizer_path = os.path.join(
    BASE_DIR,
    "03_models",
    "tfidf_vectorizer.pkl"
)


# Load model
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)


# Page title
st.title("📰 Fake News Detection System")

st.write(
    "Enter any news text and the AI model will classify it."
)


# Input box
news = st.text_area(
    "Enter News Article"
)


if st.button("Predict"):
    
    if news.strip() == "":
        st.warning("Please enter news text")

    else:
        cleaned_news = clean_text(news)
        news_vector = vectorizer.transform([cleaned_news])

        prediction = model.predict(news_vector)

        if prediction[0] == 0:
            st.error("🚨 Fake News Detected")
        else:
            st.success("✅ Real News")