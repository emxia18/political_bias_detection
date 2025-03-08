import os
import pandas as pd
import re

base_path = "data/archive" 

folder_labels = {
    "Center Data": "center",
    "Left Data": "left",
    "Right Data": "right"
}

data = []

def clean_text_func(text):
    text = re.sub(r"[^a-zA-Z0-9\s\.,!?;:'\"()-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# for folder, label in folder_labels.items():
#     folder_path = os.path.join(base_path, folder)
    
#     if os.path.exists(folder_path):
#         text_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")][:500]
        
#         for filename in text_files:
#             file_path = os.path.join(folder_path, filename)
            
#             with open(file_path, "r", encoding="utf-8") as file:
#                 text = file.read().strip()
#                 text = clean_text_func(text)
#                 data.append([text, label])

# df = pd.DataFrame(data, columns=["Text", "Label"])

# csv_output = "data/political_bias_data.csv"
# df.to_csv(csv_output, index=False, encoding="utf-8")

# print(f"CSV file saved as: {csv_output}")

file_path = "data/political_bias_data.csv"
df = pd.read_csv(file_path)

df["Title"] = ""

updated_file_path = "data/political_bias_data_title.csv"
df.to_csv(updated_file_path, index=False)

print(f"Updated CSV saved to: {updated_file_path}")


