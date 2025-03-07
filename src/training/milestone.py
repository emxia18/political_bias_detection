import kagglehub
import csv
import pandas as pd
import numpy as np
import re
import torch
from torch.utils.data import Dataset
from transformers import DistilBertForSequenceClassification
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

tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')
model = BertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=5)

train_dataset = PoliticalBiasDataset(train_data, tokenizer)
val_dataset = PoliticalBiasDataset(val_data, tokenizer)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=10,

    evaluation_strategy="epoch",  
    save_strategy="epoch",  
    save_total_limit=1,           
    load_best_model_at_end=False, 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# trainer.train(resume_from_checkpoint="results/checkpoint-1020")

# checkpoint_path = "results/checkpoint-1020"

# model = BertForSequenceClassification.from_pretrained(checkpoint_path)

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     # train_dataset=train_dataset, 
#     eval_dataset=val_dataset,
# )

eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

predictions = trainer.predict(val_dataset)

logits = predictions.predictions
probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)

predicted_labels = torch.argmax(probabilities, axis=1).numpy()

label_map = {0: 'left', 1: 'lean left', 2: 'center', 3: 'lean right', 4: 'right'}
predicted_biases = [label_map[label] for label in predicted_labels]

true_labels = [label_map[example["labels"].item()] for example in val_dataset]

results_df = pd.DataFrame({
    "Actual Bias": true_labels,
    "Predicted Bias": predicted_biases
})

results_df.to_csv("evaluation_results.csv", index=False)

print(results_df.head())
print("\nEvaluation results saved to 'evaluation_results.csv'")

