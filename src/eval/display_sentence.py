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
    fig, ax = plt.subplots(figsize=(14, 2))
    ax.axis("off")
    
    font_props = matplotlib.font_manager.FontProperties(family='sans-serif', size=20)
    
    t = ax.text(0, 0, " ", fontproperties=font_props)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    space_bbox = t.get_window_extent(renderer=renderer)
    t.remove()

    inv = ax.transData.inverted()
    space_bbox_data = inv.transform([[space_bbox.x0, space_bbox.y0], [space_bbox.x1, space_bbox.y1]])
    space_width = space_bbox_data[1][0] - space_bbox_data[0][0]
    
    word_texts = []
    word_widths = []
    for word in normalized_dict.keys():
        t = ax.text(0, 0, word, fontproperties=font_props)
        fig.canvas.draw()
        bbox = t.get_window_extent(renderer=renderer)
        bbox_data = inv.transform([[bbox.x0, bbox.y0], [bbox.x1, bbox.y1]])
        width = bbox_data[1][0] - bbox_data[0][0]
        t.remove()
        word_texts.append(word)
        word_widths.append(width)
    
    spacing_factor = 1.0
    
    x_pos = 0
    texts = []
    for word, width, norm_value in zip(word_texts, word_widths, normalized_dict.values()):
        color = get_color(norm_value)
        t = ax.text(x_pos, 0, word, fontproperties=font_props, color=color, ha="left", va="center")
        texts.append(t)
        x_pos += width + space_width * spacing_factor

    ax.set_xlim(-0.1, x_pos + 0.1)
    ax.set_ylim(-0.5, 0.5)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.show()
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

save_highlighted_words(word_frequencies, "highlighted_words.png")