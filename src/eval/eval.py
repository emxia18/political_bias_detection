import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bias_evaluator import BiasEvaluator
from bias_dataset import DataPreprocessor
from sklearn.model_selection import train_test_split
import pandas as pd

MODEL_NAME = "emxia18/bias-vanilla-4"

evaluator = BiasEvaluator(MODEL_NAME)
preprocessor = DataPreprocessor()

df = pd.read_csv("src/combined_data.csv")
full_data = df.values.tolist()

_, val_data = train_test_split(full_data, test_size=0.1, random_state=42)

processed_data = preprocessor.load_data(val_data)
encoded_data, label_mapping = preprocessor.encode_data(processed_data)

texts = [f"{title} {text}" for title, text, _ in processed_data]
true_labels = [label_mapping[label] for _, _, label in processed_data]

print(f"Validation Samples: {len(texts)}") 

# word_importance = evaluator.analyze_word_importance(texts, label_mapping, 'src/word_importance_vanilla_4.csv')
# accuracy = evaluator.evaluate_model(texts, true_labels, label_mapping, 'src/eval_results_vanilla_4.csv', 'src/confusion_matrix_vanilla_4.png')

sample_sentence = "Climate alarmists falsely claim the world is literally on fire"
target_label = list(label_mapping.keys())[0]  

word_importance = evaluator.get_word_importance(sample_sentence, label_mapping, target_label)

print("\nWord Importance Scores:")
for word, score in word_importance.items():
    print(f"{word}: {score:.4f}")