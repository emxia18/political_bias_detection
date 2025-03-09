from bias_trainer import BiasEvaluator
from bias_dataset import DataPreprocessor
from sklearn.model_selection import train_test_split
import pandas as pd

MODEL_NAME = "emxia18/bias-cleaned-data-vanilla"

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

accuracy = evaluator.evaluate_model(texts, true_labels, label_mapping)
