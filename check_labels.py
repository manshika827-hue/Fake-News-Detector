import pandas as pd

df = pd.read_csv("01_dataset/clean_news.csv")

print(df.columns)

print("\nLabel counts:")
print(df["label"].value_counts())

print("\nFirst rows:")
print(df.head())