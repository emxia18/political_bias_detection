import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from captum.attr import IntegratedGradients
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class BiasEvaluator:
    def __init__(self, model_name, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

    def analyze_word_importance(self, texts):
        integrated_gradients = IntegratedGradients(self._forward_fn)
        word_importance = Counter()
        total_word_count = Counter()

        for count, text in enumerate(texts):
            if count % 100 == 0:
                print(f"Processing text {count}/{len(texts)}")

            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(self.device, dtype=torch.long)
            attention_mask = inputs["attention_mask"].to(self.device, dtype=torch.long)

            embeddings = self.model.get_input_embeddings()(input_ids).detach().to(self.device)
            embeddings.requires_grad_()

            baseline = torch.zeros_like(embeddings).to(self.device)

            attributions = integrated_gradients.attribute(
                inputs=embeddings,
                baselines=baseline,
                target=0,
                additional_forward_args=(attention_mask,),
                return_convergence_delta=False
            )

            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            attributions = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
            attributions = np.abs(attributions)
            attributions /= attributions.sum()

            for token, score in zip(tokens, attributions):
                if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                    word_importance[token] += score
                    total_word_count[token] += 1

        avg_word_importance = {word: word_importance[word] / total_word_count[word] for word in word_importance}
        sorted_importance = sorted(avg_word_importance.items(), key=lambda x: x[1], reverse=True)

        if sorted_importance:
            print("Top word importance:", sorted_importance[0])
        else:
            print("No words found for importance analysis.")

        importance_df = pd.DataFrame(sorted_importance, columns=["Word", "Importance"])
        importance_df.to_csv("word_importance_big.csv", index=False)

        print("\nTop Important Words:")
        print(importance_df.head(20))
        print("\nWord importance scores saved to 'word_importance_big.csv'")

        return importance_df

    def _forward_fn(self, embeddings, attention_mask):
        outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
        return outputs.logits

    def evaluate_model(self, texts, true_labels, label_mapping):
        self.model.eval()
        
        predictions = []
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)

                logits = self.model(input_ids, attention_mask=attention_mask).logits
                probabilities = F.softmax(logits, dim=-1)
                predicted_label = torch.argmax(probabilities, axis=1).cpu().item()
                predictions.append(predicted_label)

        reverse_label_map = {v: k for k, v in label_mapping.items()}
        predicted_biases = [reverse_label_map[label] for label in predictions]
        true_biases = [reverse_label_map[label] for label in true_labels]

        results_df = pd.DataFrame({
            "Actual Bias": true_biases,
            "Predicted Bias": predicted_biases
        })

        results_df.to_csv("evaluation_results.csv", index=False)

        accuracy = accuracy_score(true_labels, predictions)

        print(results_df.head())
        print("\nEvaluation results saved to 'evaluation_results.csv'")
        print(f"\nRaw Accuracy Score: {accuracy:.4f}")

        print("\nClassification Report:")
        print(classification_report(true_labels, predictions, target_names=list(label_mapping.keys())))

        conf_matrix = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=list(label_mapping.keys()), yticklabels=list(label_mapping.keys()))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig("src/connfusion_matrix.png")
        plt.show()

        return accuracy
