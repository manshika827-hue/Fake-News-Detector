# 📰 Fake News Detection System

A Machine Learning-based Fake News Detection System built using **Python, Scikit-learn, NLP, and Streamlit**. This project classifies news articles as **Fake** or **Real** using Natural Language Processing (NLP) techniques and a Passive Aggressive Classifier.

---

## 📌 Project Overview

Fake news has become a major challenge in today's digital world. This project uses Machine Learning and Natural Language Processing to analyze news articles and determine whether they are genuine or fake.

The model is trained on a large dataset of real and fake news articles using TF-IDF vectorization and the Passive Aggressive Classifier.

---

## 🚀 Features

- Detects Fake and Real News
- Text Preprocessing using NLP
- Stopword Removal
- Porter Stemming
- TF-IDF Feature Extraction
- Passive Aggressive Classifier
- Interactive Streamlit Web Application
- High Prediction Accuracy (~99%)

---

## 🛠️ Technologies Used

- Python 3.x
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Joblib
- Streamlit

---

## 📂 Project Structure

```
Fake-News-Detector/
│
├── 01_dataset/
│   ├── Fake.csv
│   ├── True.csv
│   └── clean_news.csv
│
├── 02_source_code/
│   ├── 02_preprocessing.py
│   ├── 03_train_model.py
│   ├── 04_prediction.py
│   └── 05_streamlit_app.py
│
├── 03_models/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── 04_screenshots/
│
├── 05_report/
│
├── requirements.txt
│
└── README.md
```

---

## 📊 Dataset

The project uses the Kaggle Fake News Dataset.

Dataset contains:

- **Fake.csv**
- **True.csv**

After preprocessing, the cleaned dataset is stored as:

```
clean_news.csv
```

---

## ⚙️ Machine Learning Pipeline

1. Load Dataset
2. Data Cleaning
3. Remove URLs
4. Remove Special Characters
5. Convert to Lowercase
6. Remove Stopwords
7. Apply Porter Stemming
8. TF-IDF Vectorization
9. Train-Test Split
10. Train Passive Aggressive Classifier
11. Save Model & Vectorizer
12. Predict News Category

---

## 🧠 Model Used

**Passive Aggressive Classifier**

Reasons for choosing:

- Fast Training
- High Accuracy
- Suitable for Text Classification
- Performs well on large datasets

---

## 📈 Model Performance

Accuracy:

```
99.4%
```

Classification Metrics:

- Precision: 99%
- Recall: 99%
- F1-Score: 99%

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Fake-News-Detector.git
```

Move into the project

```bash
cd Fake-News-Detector
```

Create Virtual Environment

```bash
python -m venv venv
```

Activate Virtual Environment

Windows

```bash
venv\Scripts\activate
```

Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

### Step 1: Preprocess Dataset

```bash
python 02_source_code/02_preprocessing.py
```

### Step 2: Train Model

```bash
python 02_source_code/03_train_model.py
```

### Step 3: Test Prediction

```bash
python 02_source_code/04_prediction.py
```

### Step 4: Launch Streamlit App

```bash
python -m streamlit run 02_source_code/05_streamlit_app.py
```

---

## 🖥️ Streamlit Application

The web application allows users to:

- Paste a news article
- Click **Predict**
- Instantly receive whether the article is **Fake** or **Real**

---

## 📸 Screenshots

### Home Page

(Add Screenshot)

### Fake News Prediction

(Add Screenshot)

### Real News Prediction

(Add Screenshot)

### Model Accuracy

(Add Screenshot)

---

## 💡 Future Improvements

- BERT-based Fake News Detection
- Confidence Score
- News URL Detection
- File Upload Support
- Multilingual Fake News Detection
- Live News API Integration
- Deployment on Streamlit Cloud

---

## 👩‍💻 Author

**Manshika**

B.Tech Computer Science Student

Interested in:

- Artificial Intelligence
- Machine Learning
- Data Science
- Python Development

---

## 📜 License

This project is developed for educational and learning purposes.