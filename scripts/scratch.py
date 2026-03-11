import numpy as np
from PIL import Image

# 1. Load the sprite
img = Image.open("chikorita.png").convert("RGBA")
pixels = np.array(img)

# 2. Handle transparency — map transparent pixels to a special "background" color
# Replace fully transparent pixels with a sentinel color
bg_color = (0, 0, 0, 0)
rgb_pixels = []
for row in pixels:
    for pixel in row:
        if pixel[3] < 128:  # transparent
            rgb_pixels.append((0, 0, 0))  # background token
        else:
            rgb_pixels.append(tuple(pixel[:3]))

# 3. Build palette from unique colors
unique_colors = list(set(rgb_pixels))
color_to_id = {color: idx for idx, color in enumerate(unique_colors)}

# 4. Tokenize — each pixel becomes an integer
token_sequence = [color_to_id[c] for c in rgb_pixels]

# Now you have:
h, w = pixels.shape[:2]
token_grid = np.array(token_sequence).reshape(h, w)

print(f"Image size: {w}x{h}")
print(f"Palette size (vocabulary): {len(unique_colors)}")
print(f"Sequence length: {len(token_sequence)}")
print(f"First 20 tokens: {token_sequence[:20]}")
print(f"Palette: {color_to_id}")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Build reverse mapping: id -> color
id_to_color = {idx: color for color, idx in color_to_id.items()}

# Normalize palette colors to [0,1] for matplotlib
cmap_colors = [np.array(id_to_color[i]) / 255.0 for i in range(len(unique_colors))]
cmap = ListedColormap(cmap_colors)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(token_grid, cmap=cmap, interpolation="nearest")
ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
ax.grid(which="minor", color="gray", linewidth=0.5)
ax.tick_params(which="minor", size=0)
plt.title("Token Grid Visualization")
plt.show()
