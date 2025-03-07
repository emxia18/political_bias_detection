import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

class BiasTrainer:
    def __init__(self, train_dataset, val_dataset, model_name="distilbert-base-uncased", num_labels=3):

        self.model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

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
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
        )

        trainer.train()

    def evaluate(self):

        trainer = Trainer(model=self.model)
        eval_results = trainer.evaluate(self.val_dataset)
        print("Evaluation Results:", eval_results)
