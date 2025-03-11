import pandas as pd
import numpy as np
import re
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

BIAS_MAP_DEFAULT = {'left': 'left', 'center': 'center', 'right': 'right'}
LABEL_MAPPING = {'left': 0, 'center': 1, 'right': 2}

class DataPreprocessor:
    def __init__(self, tokenizer_model="distilbert-base-uncased", max_len=128, attention_type="NONE"):

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.max_len = max_len
        self.attention_type = attention_type

    def clean_text(self, text):
        """Cleans text by removing unwanted characters."""
        return re.sub(r"[^a-zA-Z0-9\s\.,!?;:'\"()-]", "", str(text))

    def load_data(self, source, title_name="title", text_name="text", bias_name="label", bias_map=None):
        bias_map = bias_map if bias_map else BIAS_MAP_DEFAULT
        data = []

        if isinstance(source, str):
            df = pd.read_csv(source)
            for _, row in df.iterrows():
                title = row.get(title_name, "")
                text = row.get(text_name, "")
                bias = row.get(bias_name, "").strip().lower()
                if bias not in bias_map:
                    continue
                data.append((self.clean_text(title), self.clean_text(text), bias_map[bias]))

        elif isinstance(source, list):
            for title, text, bias in source:
                if bias not in bias_map:
                    continue
                data.append((self.clean_text(title), self.clean_text(text), bias_map[bias]))

        return data

    def encode_data(self, data):
        texts = [f"{title} {text}" for title, text, _ in data]
        labels = [data[i][2] for i in range(len(data))]

        # label_mapping = {label: idx for idx, label in enumerate(set(labels))}
        label_mapping = LABEL_MAPPING
        numerical_labels = [label_mapping[label] for label in labels]

        encodings = self.tokenizer(
            texts, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt"
        )

        attention_bias = torch.ones_like(encodings["input_ids"], dtype=torch.float)

        if (self.attention_type == "TITLE"):
            attention_bias[:, 20:] = 0.5

        encoded_data = list(zip(encodings["input_ids"], encodings["attention_mask"], numerical_labels, attention_bias))
        return encoded_data, label_mapping

class BiasDataset(Dataset):
    def __init__(self, encoded_data):
        self.encoded_data = encoded_data

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        input_ids, attention_mask, label, attention_bias = self.encoded_data[idx]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
            "attention_bias": attention_bias,
        }