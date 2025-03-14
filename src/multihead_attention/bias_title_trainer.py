import torch
import wandb
import numpy as np
from transformers import Trainer, TrainingArguments, EvalPrediction
from biased_bert import BiasedDistilBERT
from transformers import DistilBertTokenizer

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
            run_name=wandb.run.name,
            max_grad_norm=1.0
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
                "attention_bias": torch.stack([x["attention_bias"] for x in batch]) if batch[0].get("attention_bias", None) is not None else None,
            },
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
    
    def compute_metrics(self, eval_pred: EvalPrediction):
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        predictions = np.argmax(logits, axis=-1)
        acc = (predictions == labels).mean()
        return {"accuracy": acc} 

    def push_to_huggingface(self, repo_name, organization=None):
        self.model.push_to_hub(repo_name, organization=organization)
        self.tokenizer.push_to_hub(repo_name, organization=organization)
        print(f"Model and tokenizer pushed to Hugging Face Hub at: https://huggingface.co/{repo_name}")
