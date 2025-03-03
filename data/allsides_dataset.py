import csv
import pandas as pd
import numpy as np
import re
from collections import Counter

def clean_text_func(text):
    return re.sub(r"[^a-zA-Z0-9\s\.,!?;:'\"()-]", "", text)

df = pd.read_csv('data/allsides_balanced_news_headlines-texts.csv')
print(df.head())
# print(len(df))
# print(df.columns)

bias_list = ['left', 'center', 'right']
ds = []
for index, row in df.iterrows():
    title = row["title"]
    text = row["text"]
    bias = row["bias_rating"]
    if bias not in bias_list:
        print(f"invalid bias type {bias}")

    if isinstance(text, str):
        clean_text = clean_text_func(text)
        clean_title = clean_text_func(title)
        ds.append([clean_title, clean_text, bias])
    
print(ds[0])
print(len(ds))

bias_counts = Counter(row[2] for row in ds)

print(f"Left: {bias_counts.get('left', 0)}")
print(f"Right: {bias_counts.get('right', 0)}")
print(f"Center: {bias_counts.get('center', 0)}")


