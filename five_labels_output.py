import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

# Load evaluation results
results_df = pd.read_csv("evaluation_results_five_labels.csv")

# Define label mapping (keeping all five categories)
label_map = {'left': 0, 'lean left': 1, 'center': 2, 'lean right': 3, 'right': 4}

# Convert text labels to numerical labels
results_df["Actual Bias"] = results_df["Actual Bias"].map(label_map)
results_df["Predicted Bias"] = results_df["Predicted Bias"].map(label_map)

# Calculate accuracy
accuracy = accuracy_score(results_df["Actual Bias"], results_df["Predicted Bias"])
print(f"Model Accuracy: {accuracy:.4f}")

# Generate confusion matrix
conf_matrix = confusion_matrix(results_df["Actual Bias"], results_df["Predicted Bias"])

# Plot confusion matrix
plt.figure(figsize=(7, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=label_map.keys(), yticklabels=label_map.keys())
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")

# Save the figure
plt.savefig("confusion_matrix_unmerged.png", dpi=300, bbox_inches='tight')
plt.show()

print("Confusion matrix saved as 'confusion_matrix_unmerged.png'")
