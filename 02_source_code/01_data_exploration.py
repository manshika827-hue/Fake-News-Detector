# Import library
import pandas as pd

# Load datasets
fake_df = pd.read_csv(r"C:\Users\Manshika\OneDrive\Desktop\Fake-News-Detector\01_dataset\Fake.csv", on_bad_lines="skip", engine="python")
true_df = pd.read_csv(r"C:\Users\Manshika\OneDrive\Desktop\Fake-News-Detector\01_dataset\True.csv")

# Display first 5 rows
print("========== FAKE NEWS DATASET ==========")
print(fake_df.head())

print("\n========== TRUE NEWS DATASET ==========")
print(true_df.head())

# Display dataset shape
print("\nFake Dataset Shape:", fake_df.shape)
print("True Dataset Shape:", true_df.shape)

# Display column names
print("\nFake Dataset Columns:")
print(fake_df.columns)

print("\nTrue Dataset Columns:")
print(true_df.columns)

# Check missing values
print("\nMissing Values in Fake Dataset:")
print(fake_df.isnull().sum())

print("\nMissing Values in True Dataset:")
print(true_df.isnull().sum())

# Add labels
fake_df["label"] = 0
true_df["label"] = 1

# Merge datasets
news = pd.concat([fake_df, true_df], ignore_index=True)

# Shuffle dataset
news = news.sample(frac=1, random_state=42)
news.reset_index(drop=True, inplace=True)

# Display merged dataset
print("\nMerged Dataset:")
print(news.head())

# Display merged dataset shape
print("\nMerged Dataset Shape:", news.shape)

# Check label distribution
print("\nLabel Distribution:")
print(news["label"].value_counts())