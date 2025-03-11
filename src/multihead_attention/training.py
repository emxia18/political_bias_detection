import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bias_dataset import DataPreprocessor, BiasDataset
from bias_title_trainer import BiasTitleTrainer
from sklearn.model_selection import train_test_split
import pandas as pd
import wandb

wandb.init(project="political-bias-detection")

preprocessor = DataPreprocessor(attention_type="TITLE")

df = pd.read_csv('src/combined_data.csv')
full_data = df.values.tolist()

train_data, val_data = train_test_split(full_data, test_size=0.2, random_state=42)

train_encoded, label_mapping = preprocessor.encode_data(train_data)
val_encoded, _ = preprocessor.encode_data(val_data)

train_dataset = BiasDataset(train_encoded)
val_dataset = BiasDataset(val_encoded)

trainer = BiasTitleTrainer(train_dataset, val_dataset)
trainer.train()

trainer.push_to_huggingface("emxia18/bias-attention-max")