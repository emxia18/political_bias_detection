import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

# Load evaluation results
results_df = pd.read_csv("evaluation_results.csv")

# Define label mapping
label_map = {'left': 0, 'center': 1, 'right': 2}

# Convert text labels to numerical labels
results_df["Actual Bias"] = results_df["Actual Bias"].map(label_map)
results_df["Predicted Bias"] = results_df["Predicted Bias"].map(label_map)

# Calculate accuracy
accuracy = accuracy_score(results_df["Actual Bias"], results_df["Predicted Bias"])
print(f"Model Accuracy: {accuracy:.4f}")

# Generate confusion matrix
conf_matrix = confusion_matrix(results_df["Actual Bias"], results_df["Predicted Bias"])

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_map.keys(), yticklabels=label_map.keys())
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()
