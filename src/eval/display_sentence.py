import matplotlib.pyplot as plt
import matplotlib.font_manager
import numpy as np

def normalize_values(word_dict):
    values = np.array(list(word_dict.values()), dtype=float)
    min_val, max_val = values.min(), values.max()
    return {word: (val - min_val) / (max_val - min_val) if max_val > min_val else 0.5 for word, val in word_dict.items()}

def get_color(value):
    cmap = plt.get_cmap("coolwarm")
    rgba = cmap(value)
    return (rgba[0], rgba[1], rgba[2])

def save_highlighted_words(word_dict, filename="highlighted_words.png"):
    normalized_dict = normalize_values(word_dict)
    
    fig, ax = plt.subplots(figsize=(12, 1.5))
    ax.axis("off")
    
    x_pos = 0.1
    y_pos = 0.5
    word_spacing = 0.03
    
    for word, norm_value in normalized_dict.items():
        color = get_color(norm_value)
        
        t = ax.text(x_pos, y_pos, word, fontsize=20, color=color, ha="left", va="center")
        
        fig.canvas.draw()
        bbox = t.get_window_extent()
        width_inches = bbox.width / fig.dpi
        
        x_pos += width_inches + word_spacing
    
    ax.set_xlim(0, x_pos + 0.1)
    ax.set_ylim(0, 1)
    
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()

word_frequencies = {
    "this": 0.0453,
    "is": 0.0683,
    "a": 0.0093,
    "test": 0.0508,
    "sentence": 0.1118,
    "to": 0.1134,
    "analyze": 0.1943,
    "the": 0.1030,
    "models": 0.0206,
    "attention": 0.0234
}

save_highlighted_words(word_frequencies, "src/highlighted_words.png")