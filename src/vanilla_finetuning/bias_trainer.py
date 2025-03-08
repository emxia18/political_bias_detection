import torch
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from captum.attr import IntegratedGradients
import numpy as np
from collections import Counter

class BiasTrainer:
    def __init__(self, train_dataset, val_dataset, tokenizer_model="distilbert-base-uncased", model_name="distilbert-base-uncased", num_labels=3):
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
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
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
        )

        trainer.train()
        # self.analyze_word_importance()

    def analyze_word_importance(self):
        self.model.eval()
        integrated_gradients = IntegratedGradients(self.model)
        word_importance = Counter()
        total_word_count = Counter()

        for sample in self.val_dataset:
            input_ids = sample["input_ids"].unsqueeze(0).to(dtype=torch.long)  # Ensure LongTensor
            attention_mask = sample["attention_mask"].unsqueeze(0).to(dtype=torch.long)  # Ensure LongTensor
            baseline = torch.zeros_like(input_ids).to(dtype=torch.long)  # Ensure baseline is also LongTensor

            attributions, _ = integrated_gradients.attribute(
                inputs=input_ids, 
                baselines=baseline, 
                target=None, 
                additional_forward_args=(attention_mask,),  # Pass attention mask
                return_convergence_delta=True
            )
            
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            attributions = attributions.sum(dim=-1).squeeze(0).detach().numpy()
            attributions = np.abs(attributions)
            attributions /= attributions.sum()

            for token, score in zip(tokens, attributions):
                if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                    word_importance[token] += score
                    total_word_count[token] += 1

        avg_word_importance = {word: word_importance[word] / total_word_count[word] for word in word_importance}
        sorted_importance = sorted(avg_word_importance.items(), key=lambda x: x[1], reverse=True)

        importance_df = pd.DataFrame(sorted_importance, columns=["Word", "Importance"])
        importance_df.to_csv("word_importance.csv", index=False)

        print("\nTop Important Words:")
        print(importance_df.head(20))
        print("\nWord importance scores saved to 'word_importance.csv'")


    def push_to_huggingface(self, repo_name, organization=None):
        self.model.push_to_hub(repo_name, organization=organization)
        self.tokenizer.push_to_hub(repo_name, organization=organization)
        print(f"Model and tokenizer pushed to Hugging Face Hub at: https://huggingface.co/{repo_name}")