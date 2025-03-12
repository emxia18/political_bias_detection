import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from captum.attr import IntegratedGradients
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from IPython.display import display, HTML


class BiasEvaluator:
    def __init__(self, model_name, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

    def analyze_word_importance(self, texts, label_mapping, save_csv):
        integrated_gradients = IntegratedGradients(self._forward_fn)
        
        word_importance = {}
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

            for class_name, class_label in label_mapping.items():
                attributions = integrated_gradients.attribute(
                    inputs=embeddings,
                    baselines=baseline,
                    target=class_label, 
                    additional_forward_args=(attention_mask,),
                    return_convergence_delta=False
                )

                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
                attributions = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
                attributions = np.abs(attributions)
                attributions /= attributions.sum()

                for token, score in zip(tokens, attributions):
                    if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                        if token not in word_importance:
                            word_importance[token] = {label: 0 for label in label_mapping.keys()} 

                        word_importance[token][class_name] += score
                        total_word_count[token] += 1

        for word, scores in word_importance.items():
            total = sum(scores.values())
            if total > 0:
                for class_name in scores:
                    word_importance[word][class_name] /= total 

        importance_df = pd.DataFrame.from_dict(word_importance, orient="index").reset_index()
        importance_df.columns = ["Word"] + list(label_mapping.keys())
        importance_df = importance_df.sort_values(by=list(label_mapping.keys()), ascending=False)

        importance_df.to_csv(save_csv, index=False)

        print("\nTop Important Words Per Class:")
        print(importance_df.head(20))
        print(f"\nWord importance scores saved to '{save_csv}'")

        return importance_df

    def _forward_fn(self, embeddings, attention_mask):
        outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
        return outputs.logits

    def evaluate_model(self, texts, true_labels, label_mapping, eval_results_csv, confusion_matrix_csv):
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

        results_df.to_csv(eval_results_csv, index=False)

        accuracy = accuracy_score(true_labels, predictions)

        print(results_df.head())
        print(f"\nEvaluation results saved to '{eval_results_csv}'")
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
        plt.savefig(confusion_matrix_csv)
        plt.show()

        return accuracy

    def highlight_text(self, sentence, label_mapping, target_label):
        integrated_gradients = IntegratedGradients(self._forward_fn)
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        embeddings = self.model.get_input_embeddings()(input_ids).detach().to(self.device)
        embeddings.requires_grad_()

        baseline = torch.zeros_like(embeddings).to(self.device)

        attributions = integrated_gradients.attribute(
            inputs=embeddings,
            baselines=baseline,
            target=label_mapping[target_label],
            additional_forward_args=(attention_mask,),
            return_convergence_delta=False
        )

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        attributions = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        attributions = np.abs(attributions)
        attributions /= attributions.sum()

        max_attr = max(attributions) if attributions.size > 0 else 1
        highlighted_text = ""
        
        for token, score in zip(tokens, attributions):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            intensity = int((score / max_attr) * 255)
            color = f"rgba(255, 0, 0, {score / max_attr:.2f})"
            highlighted_text += f'<span style="background-color: {color}; padding: 2px; border-radius: 5px;">{token}</span> '

        display(HTML(f"<p style='font-size:16px;'>{highlighted_text}</p>"))
