import torch
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

class BiasTrainer:
    def __init__(self, train_dataset, val_dataset, model_name="distilbert-base-uncased", num_labels=3):

        self.model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
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
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
        )

        trainer.train()

        predictions = trainer.predict(self.val_dataset)
        logits = predictions.predictions
        probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)

        predicted_labels = torch.argmax(probabilities, axis=1).tolist()

        if self.num_labels == 3:
            label_map = {0: 'left', 1: 'center', 2: 'right'}
        elif self.num_labels == 5:
            label_map = {0: 'left', 1: 'lean left', 2: 'center', 3: 'lean right', 4: 'right'}
        else:
            raise ValueError("Invalid number of labels. Ensure label_map matches num_labels.")

        predicted_biases = [label_map[label] for label in predicted_labels]

        true_labels = [label_map[example["labels"].item()] for example in self.val_dataset]

        results_df = pd.DataFrame({
            "Actual Bias": true_labels,
            "Predicted Bias": predicted_biases
        })

        results_df.to_csv("evaluation_results.csv", index=False)

        print(results_df.head())
        print("\nEvaluation results saved to 'evaluation_results.csv'")
