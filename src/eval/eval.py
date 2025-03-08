import torch
import pandas as pd
import re
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Define function to clean text
def clean_text_func(text):
    return re.sub(r"[^a-zA-Z0-9\s\.,!?;:'\"()-]", "", text)

# Load the dataset
df = pd.read_csv('dataset/mayobanexsantana/political-bias/versions/1/Political_Bias.csv')

# Define bias categories
bias_list = ['left', 'lean left', 'center', 'lean right', 'right']

# Preprocess dataset
ds = []
for index, row in df.iterrows():
    title = row["Title"]
    text = row["Text"]
    bias = row["Bias"]
    if bias not in bias_list:
        print(f"Invalid bias type {bias}")
    if isinstance(text, str):
        clean_text = clean_text_func(text)
        clean_title = clean_text_func(title)
        ds.append([clean_title, clean_text, bias])

# Split into validation set (assuming training was done separately)
from sklearn.model_selection import train_test_split
_, val_data = train_test_split(ds, test_size=0.2, random_state=42)

# Define dataset class
class PoliticalBiasDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length
        self.label_map = {'left': 0, 'lean left': 1, 'center': 2, 'lean right': 3, 'right': 4}
        self.inverse_label_map = {v: k for k, v in self.label_map.items()}  # Reverse mapping

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
        encoding['labels'] = torch.tensor(self.label_map[bias], dtype=torch.long)
        return encoding

# Load the tokenizer and model
model_path = "results/checkpoint-1020"  # Change this to your latest checkpoint directory
tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')
model = BertForSequenceClassification.from_pretrained(model_path)

# Prepare validation dataset
val_dataset = PoliticalBiasDataset(val_data, tokenizer)

# Set up Trainer for evaluation
training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=val_dataset,
)

# Run evaluation and get predictions
predictions = trainer.predict(val_dataset)

# Extract logits and compute probabilities
logits = predictions.predictions
probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)

# Convert probabilities to predicted class indices
predicted_labels = torch.argmax(probabilities, axis=1).numpy()

# Convert indices to bias labels
label_map = {0: 'left', 1: 'lean left', 2: 'center', 3: 'lean right', 4: 'right'}
predicted_biases = [label_map[label] for label in predicted_labels]

# Extract true labels
true_labels = [label_map[example["labels"].item()] for example in val_dataset]

# Save results to CSV
results_df = pd.DataFrame({
    "Actual Bias": true_labels,
    "Predicted Bias": predicted_biases
})

results_df.to_csv("evaluation_results.csv", index=False)

# Print sample results
print(results_df.head())
print("\nEvaluation results saved to 'evaluation_results.csv'")
