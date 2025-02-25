import kagglehub
import csv
import pandas as pd
import numpy as np
import re
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

def clean_text_func(text):
    return re.sub(r"[^a-zA-Z0-9\s\.,!?;:'\"()-]", "", text)

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

train_data, val_data = train_test_split(ds, test_size=0.2, random_state=42)

class PoliticalBiasDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length
        self.label_map = {'left': 0, 'lean left': 1, 'center': 2, 'lean right': 3, 'right': 4}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        title, text, bias = self.data[idx]

        combined_text = title + " " + text
        encoding = self.tokenizer(
            combined_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        label = self.label_map[bias]
        encoding['labels'] = torch.tensor(label, dtype=torch.long)
        return encoding

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

train_dataset = PoliticalBiasDataset(train_data, tokenizer)
val_dataset = PoliticalBiasDataset(val_data, tokenizer)

training_args = TrainingArguments(
    output_dir='./results',            
    num_train_epochs=3,            
    per_device_train_batch_size=8,   
    per_device_eval_batch_size=8,    
    warmup_steps=500,                 
    weight_decay=0.01,               
    logging_dir='./logs',              
    logging_steps=10,
    evaluation_strategy="steps",     
    eval_steps=50,
    save_steps=50,
    load_best_model_at_end=True,    
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)
