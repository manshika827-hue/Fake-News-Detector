import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords
nltk.download("stopwords")

# Load datasets
fake_df = pd.read_csv(
    r"C:\Users\Manshika\OneDrive\Desktop\Fake-News-Detector\01_dataset\Fake.csv",
    on_bad_lines="skip",
    engine="python"
)

true_df = pd.read_csv(
    r"C:\Users\Manshika\OneDrive\Desktop\Fake-News-Detector\01_dataset\True.csv"
)

# Add labels
fake_df["label"] = 0
true_df["label"] = 1

# Merge datasets
news = pd.concat([fake_df, true_df], ignore_index=True)

# Shuffle dataset
news = news.sample(frac=1, random_state=42).reset_index(drop=True)

# Initialize
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# Cleaning function
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

# Apply cleaning
news["clean_text"] = news["text"].apply(clean_text)

# Save cleaned dataset
news.to_csv(
    r"C:\Users\Manshika\OneDrive\Desktop\Fake-News-Detector\01_dataset\clean_news.csv",
    index=False
)

print(news[["text", "clean_text"]].head())

print("\nPreprocessing completed successfully!")