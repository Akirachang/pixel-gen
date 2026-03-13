"""Generate pixel art sprites from text descriptions using a trained PixelGPT."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.model import PixelGPT
from src.tokenizer import DescriptionTokenizer


def pixels_to_image(pixel_ids: list[int], palette: np.ndarray, sprite_size: int) -> Image.Image:
    """Convert a sequence of palette indices to an RGB image.

    Args:
        pixel_ids: List of palette index integers.
        palette: (K, 3) uint8 color palette.
        sprite_size: Width/height of the sprite.

    Returns:
        PIL Image.
    """
    arr = np.array(pixel_ids, dtype=np.int32).reshape(sprite_size, sprite_size)
    rgb = palette[arr]  # (H, W, 3)
    return Image.fromarray(rgb.astype(np.uint8))


def main() -> None:
    """Generate sprites from descriptions."""
    parser = argparse.ArgumentParser(description="Generate pixel art with PixelGPT.")
    parser.add_argument("checkpoint", help="Path to checkpoint .pt file.")
    parser.add_argument("--dataset", default="pokemondb-gen3", help="Dataset name (for palette).")
    parser.add_argument("--description", type=str, required=True, help="Description to condition on.")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of sprites to generate.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    # Resolve device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    hparams = ckpt["hparams"]

    # Load palette
    paths = config.dataset_paths(args.dataset)
    palette = np.load(paths["processed"] / "palette.npy")
    ds_cfg = config.load_dataset_config(args.dataset)
    sprite_size = ds_cfg["sprite_size"]
    num_pixels = sprite_size * sprite_size

    # Load tokenizer from the run directory
    ckpt_path = Path(args.checkpoint)
    run_dir = config.RUNS_DIR / ckpt_path.parent.name
    tokenizer = DescriptionTokenizer.load(run_dir / "vocab.json")

    # Build model
    model = PixelGPT(
        desc_vocab_size=hparams["desc_vocab_size"],
        pixel_vocab_size=hparams["pixel_vocab_size"],
        d_model=hparams["d_model"],
        n_heads=hparams["n_heads"],
        n_layers=hparams["n_layers"],
        max_seq_len=hparams["max_seq_len"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (loss={ckpt['loss']:.4f})")

    # Encode description
    desc_ids = tokenizer.encode(args.description)
    print(f"Description: {args.description}")
    print(f"Encoded: {desc_ids}")

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate
    for i in range(args.num_samples):
        print(f"Generating sample {i + 1}/{args.num_samples}...")
        pixel_ids = model.generate(
            desc_ids=desc_ids,
            img_start_id=tokenizer.img_start_id,
            num_pixels=num_pixels,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )

        img = pixels_to_image(pixel_ids, palette, sprite_size)
        desc_slug = args.description.replace(" ", "_")[:30]
        filename = f"{desc_slug}_{i}.png"
        img.save(out_dir / filename)
        print(f"  Saved: {out_dir / filename}")

    print("Done.")


if __name__ == "__main__":
    main()
