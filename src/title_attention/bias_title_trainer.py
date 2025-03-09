import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import wandb

from collections import Counter
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from captum.attr import IntegratedGradients
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class BiasTitleTrainer:
    def __init__(self, train_dataset, val_dataset, tokenizer_model="distilbert-base-uncased", model_name="distilbert-base-uncased", num_labels=3):
        self.model = BiasedDistilBERT(model_name=model_name, num_labels=num_labels)
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_model)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_labels = num_labels

    def train(self, output_dir='./results', epochs=3, batch_size=8):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=False,
            report_to="wandb",
            run_name=wandb.run.name
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=lambda batch: {
                "input_ids": torch.stack([x["input_ids"] for x in batch]),
                "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
                "labels": torch.tensor([x["labels"] for x in batch], dtype=torch.long),
                "attention_bias": torch.stack([x["attention_bias"] for x in batch]),
            }
        )

        trainer.train()
