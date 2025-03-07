import csv
import pandas as pd
import numpy as np
import re
from collections import Counter

BIAS_LIST = ['left', 'center', 'right']
BIAS_MAP = {'left': 'left', 'center': 'center', 'right': 'right'}

class Dataset:
    def __init__(self, file_path, title_name="title", text_name="text", bias_name="bias_rating", bias_map=None):
        self.file_path = file_path
        self.title_name = title_name
        self.text_name = text_name
        self.bias_name = bias_name
        self.bias_map = bias_map if bias_map else BIAS_MAP
        self.data = []
        self.load_data()

    def load_data(self):
        df = pd.read_csv(self.file_path)

        for index, row in df.iterrows():
            title = row.get(self.title_name, "")
            text = row.get(self.text_name, "")
            bias = row.get(self.bias_name, "").strip().lower()

            if bias not in self.bias_map:
                print(f"Invalid bias type: {bias}")
                continue

            if isinstance(text, str) and isinstance(title, str):
                clean_text = self.clean_text_func(text)
                clean_title = self.clean_text_func(title)
                self.data.append([clean_title, clean_text, self.bias_map[bias]])

    def clean_text_func(self, text):
        return re.sub(r"[^a-zA-Z0-9\s\.,!?;:'\"()-]", "", text)

    def get_data(self):
        return self.data

    def get_bias_counts(self):
        bias_counts = Counter(row[2] for row in self.data)
        return {
            "Left": bias_counts.get('left', 0),
            "Right": bias_counts.get('right', 0),
            "Center": bias_counts.get('center', 0),
        }
