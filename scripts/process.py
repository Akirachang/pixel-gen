"""Convert raw sprite PNGs into tokenized numpy arrays using a shared palette."""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# Token ID reserved for transparent/background pixels.
BG_TOKEN = 0
BG_COLOR = np.array([0, 0, 0], dtype=np.uint8)
TRANSPARENCY_THRESHOLD = 128


def load_sprite(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a sprite PNG and separate it into RGB values and a transparency mask.

    Args:
        path: Path to the PNG file.

    Returns:
        rgb: (H, W, 3) uint8 array of RGB values.
        mask: (H, W) bool array — True where the pixel is opaque.
    """
    img = Image.open(path).convert("RGBA")
    pixels = np.array(img)
    mask = pixels[:, :, 3] >= TRANSPARENCY_THRESHOLD
    rgb = pixels[:, :, :3]
    return rgb, mask


def build_palette(image_dir: Path) -> np.ndarray:
    """Collect all unique colors across all sprites into a palette.

    The background color is prepended as palette index 0.

    Args:
        image_dir: Directory containing PNG sprites.

    Returns:
        palette: (N_unique + 1, 3) uint8 array. Index 0 is the background color.
    """
    unique_colors: set[tuple[int, ...]] = set()
    for path in sorted(image_dir.glob("*.png")):
        rgb, mask = load_sprite(path)
        opaque_pixels = rgb[mask]  # (N, 3)
        for color in opaque_pixels:
            unique_colors.add(tuple(color))

    sorted_colors = sorted(unique_colors)
    palette = np.vstack([BG_COLOR, np.array(sorted_colors, dtype=np.uint8)])
    print(f"Palette: {len(sorted_colors)} unique colors + 1 background = {palette.shape[0]} total")
    return palette


def tokenize_sprite(
    rgb: np.ndarray, mask: np.ndarray, color_to_id: dict[tuple[int, ...], int]
) -> np.ndarray:
    """Map each pixel to its palette index via exact color lookup.

    Transparent pixels are mapped to BG_TOKEN (0).

    Args:
        rgb: (H, W, 3) uint8 array.
        mask: (H, W) bool array — True where opaque.
        color_to_id: Mapping from (R, G, B) tuple to palette index.

    Returns:
        tokens: (H, W) int array of palette indices.
    """
    h, w = mask.shape
    tokens = np.full((h, w), BG_TOKEN, dtype=np.int32)

    ys, xs = np.where(mask)
    for y, x in zip(ys, xs):
        color = tuple(rgb[y, x])
        tokens[y, x] = color_to_id[color]

    return tokens


def generate_contact_sheet(
    tokens: np.ndarray, palette: np.ndarray, sprite_size: int, out_path: Path
) -> None:
    """Render all tokenized sprites into a single PNG contact sheet.

    Args:
        tokens: (N, H*W) array of palette indices.
        palette: (K, 3) uint8 color palette.
        sprite_size: Width/height of each sprite in pixels.
        out_path: Where to save the contact sheet PNG.
    """
    n = tokens.shape[0]
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    sheet = np.zeros((rows * sprite_size, cols * sprite_size, 3), dtype=np.uint8)

    for i in range(n):
        grid = tokens[i].reshape(sprite_size, sprite_size)
        rgb = palette[grid]  # (H, W, 3)
        r, c = divmod(i, cols)
        y, x = r * sprite_size, c * sprite_size
        sheet[y : y + sprite_size, x : x + sprite_size] = rgb

    Image.fromarray(sheet).save(out_path)
    print(f"Contact sheet ({cols}x{rows}) saved to {out_path}")


def main() -> None:
    """Run the full processing pipeline."""
    parser = argparse.ArgumentParser(description="Process raw sprites into token arrays.")
    parser.add_argument("dataset", help="Name of dataset folder under data/.")
    parser.add_argument(
        "--preview", action="store_true", help="Generate a contact sheet preview."
    )
    args = parser.parse_args()

    # Load dataset-specific config and paths
    ds_cfg = config.load_dataset_config(args.dataset)
    paths = config.dataset_paths(args.dataset)
    sprite_size = ds_cfg["sprite_size"]

    paths["processed"].mkdir(parents=True, exist_ok=True)

    png_files = sorted(paths["raw"].glob("*.png"))
    if not png_files:
        print(f"No PNGs found in {paths['raw']}. Run scrape.py first.")
        return

    print(f"Found {len(png_files)} sprites in {paths['raw']}")

    # 1. Build palette from all unique colors
    palette = build_palette(paths["raw"])
    palette_path = paths["processed"] / "palette.npy"
    np.save(palette_path, palette)

    # Build reverse lookup: (R, G, B) -> token ID
    color_to_id = {tuple(palette[i]): i for i in range(palette.shape[0])}

    # 2. Tokenize each sprite
    all_tokens: list[np.ndarray] = []
    for path in tqdm(png_files, desc="Tokenizing sprites"):
        rgb, mask = load_sprite(path)
        tokens = tokenize_sprite(rgb, mask, color_to_id)
        all_tokens.append(tokens.flatten())

    # 3. Save as a single array: (num_sprites, H * W)
    dataset = np.stack(all_tokens, axis=0)
    dataset_path = paths["processed"] / "tokens.npy"
    np.save(dataset_path, dataset)
    print(f"Dataset shape: {dataset.shape}")
    print(f"Vocab size: {palette.shape[0]} (0 = background, 1–{palette.shape[0] - 1} = colors)")

    # 4. Optional contact sheet preview
    if args.preview:
        sheet_path = paths["processed"] / "contact_sheet.png"
        generate_contact_sheet(dataset, palette, sprite_size, sheet_path)


if __name__ == "__main__":
    main()
