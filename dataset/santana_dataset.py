import kagglehub
import csv
import pandas as pd
import numpy as np
import re

def clean_text_func(text):
    return re.sub(r"[^a-zA-Z0-9\s\.,!?;:'\"()-]", "", text)

# path = kagglehub.dataset_download("mayobanexsantana/political-bias")
# print("Path to dataset files:", path)

df = pd.read_csv('dataset/mayobanexsantana/political-bias/versions/1/Political_Bias.csv')
print(df.head())
print(len(df))
print(df.columns)

bias_list = ['left', 'lean left', 'center', 'lean right', 'right']
ds = []
for index, row in df.iterrows():
    title = row["Title"]
    text = row["Text"]
    bias = row["Bias"]
    if bias not in bias_list:
        print(f"invalid bias type {bias}")

    if isinstance(text, str):
        clean_text = clean_text_func(text)
        clean_title = clean_text_func(title)
        ds.append([clean_title, clean_text, bias])
    
print(ds[30])
print(len(ds))


