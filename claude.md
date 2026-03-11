# Pixel-Gen

## What is this?

Pixel-Gen trains a small AI model to generate pixel art from scratch. Instead of treating images as images, it treats them as sequences of colors — like words in a sentence — and learns to "write" new sprites one pixel at a time.

## How it works

1. **Data**: Pixel art sprites are converted into sequences of color tokens. Each image is reduced to a small shared color palette, then flattened into a list of palette indices — the same way text is turned into a list of word IDs.
2. **Model**: A GPT-style transformer learns to predict the next color in the sequence given everything before it. It reads pixels left-to-right, top-to-bottom, like reading a book.
3. **Generation**: To create new art, the model starts from a blank canvas token and samples one pixel at a time until the image is complete. Temperature and top-k sampling control how creative or conservative the output is.

## Goals

- Build the full pipeline end-to-end: data processing, model training, and generation.
- Keep it simple and self-contained — no heavy frameworks or cloud infra.
- Generate recognizable, novel pixel art sprites that look like they belong in the training set.

## Coding Conventions

- Type hints on all function signatures.
- Google-style docstrings on public functions and classes.
- Config values in `config.py`, never hardcoded.
- Use `pathlib.Path` for all file paths.
- Set random seeds for reproducibility.
