import pandas as pd
import numpy as np
import re
import torch
from collections import Counter
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset

BIAS_MAP_DEFAULT = {'left': 'left', 'center': 'center', 'right': 'right'}

class BiasDataset:
    def __init__(self, file_path, title_name="title", text_name="text", bias_name="bias_rating", bias_map=None, tokenizer_model="distilbert-base-uncased", max_len=128):
        self.file_path = file_path
        self.title_name = title_name
        self.text_name = text_name
        self.bias_name = bias_name
        self.bias_map = bias_map if bias_map else BIAS_MAP_DEFAULT
        self.max_len = max_len
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

        self.data = []
        self.encoded_data = []
        self.load_data()

    def clean_text(self, text):
        return re.sub(r"[^a-zA-Z0-9\s\.,!?;:'\"()-]", "", str(text))

    def load_data(self):
        df = pd.read_csv(self.file_path)

        for _, row in df.iterrows():
            title = row.get(self.title_name, "")
            text = row.get(self.text_name, "")
            bias = row.get(self.bias_name, "").strip().lower()

            if bias not in self.bias_map:
                print(f"Invalid bias type: {bias}")
                continue

            if isinstance(text, str) and isinstance(title, str):
                clean_title = self.clean_text(title)
                clean_text = self.clean_text(text)
                bias_label = self.bias_map[bias]

                self.data.append([clean_title, clean_text, bias_label])

        self.encode_data()

    def encode_data(self):
        texts = [f"{title} {text}" for title, text, _ in self.data]
        labels = [self.data[i][2] for i in range(len(self.data))]

        label_mapping = {label: idx for idx, label in enumerate(set(labels))}
        self.label_mapping = label_mapping
        numerical_labels = [label_mapping[label] for label in labels]

        encodings = self.tokenizer(texts, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")

        self.encoded_data = list(zip(encodings["input_ids"], encodings["attention_mask"], numerical_labels))

    def get_dataloader(self, batch_size=16, shuffle=True):
        return DataLoader(self.encoded_data, batch_size=batch_size, shuffle=shuffle)

    def save_as_npy(self, filename):
        np.save(filename, np.array(self.data, dtype=object))

    def get_bias_counts(self):
        bias_counts = Counter(row[2] for row in self.data)
        return {label: bias_counts.get(label, 0) for label in self.label_mapping.keys()}
