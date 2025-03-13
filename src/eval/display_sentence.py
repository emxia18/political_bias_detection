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
    "climate": 0.0525,
    "alarmists": 0.1512 + 0.0594,
    "falsely": 0.0610,
    "claim": 0.0094,
    "the": 0.0360,
    "world": 0.1461,
    "is": 0.0105,
    "literally": 0.0107,
    "on": 0.0082,
    "fire": 0.0250,
}

save_highlighted_words(word_frequencies, "src/highlighted_words.png")