import pandas as pd

file_path = "src/final_word_importance.csv"
df1 = pd.read_csv(file_path)

file_path = "src/final_word_importance2.csv"
df2 = pd.read_csv(file_path)

df = pd.concat([df1, df2], ignore_index=True)

df['Word'] = df['Word'].astype(str).fillna('')
df = df[~df['Word'].str.startswith("##")]

top_left = df.nlargest(10, 'left')[['Word', 'left']]
top_center = df.nlargest(10, 'center')[['Word', 'center']]
top_right = df.nlargest(10, 'right')[['Word', 'right']]

print("Top 10 Words - Left:")
print(top_left.to_string(index=False))

print("\nTop 10 Words - Center:")
print(top_center.to_string(index=False))

print("\nTop 10 Words - Right:")
print(top_right.to_string(index=False))

import matplotlib.pyplot as plt

left_words = {
    "perpetrators": 0.978996, "scientific": 0.910941, "cool": 0.901993,
    "educational": 0.900972, "variously": 0.894237, "dysfunction": 0.886804, "delivers": 0.878724
}

center_words = {
    "regaining": 0.923654, "cultural": 0.873603, "favors": 0.792380,
    "coaches": 0.776659, "orchestra": 0.762194, "everywhere": 0.749502, "masterpiece": 0.738212
}

right_words = {
    "alleviate": 0.928535, "scream": 0.918660, "spilling": 0.862873,
    "ignored": 0.859533, "bs": 0.846374, "besides": 0.789424, "horrors": 0.772387
}

left_labels, left_values = zip(*left_words.items())
center_labels, center_values = zip(*center_words.items())
right_labels, right_values = zip(*right_words.items())

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Word Attention by Category", fontsize=16) 

left_color = "#ADD8E6"  # Lighter Blue
center_color = "#98FB98"  # Lighter Green
right_color = "#FFC0CB"  # Lighter Pink

axes[0].bar(left_labels, left_values, color=left_color)
axes[0].set_title("Left", fontsize=14)
axes[0].set_xticks(range(len(left_labels)))
axes[0].set_xticklabels(left_labels, rotation=45, ha="right")

axes[1].bar(center_labels, center_values, color=center_color)
axes[1].set_title("Center", fontsize=14)
axes[1].set_xticks(range(len(center_labels)))
axes[1].set_xticklabels(center_labels, rotation=45, ha="right")

axes[2].bar(right_labels, right_values, color=right_color)
axes[2].set_title("Right", fontsize=14)
axes[2].set_xticks(range(len(right_labels)))
axes[2].set_xticklabels(right_labels, rotation=45, ha="right")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("src/word_attention.png")
plt.show()
