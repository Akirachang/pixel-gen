"""Train PixelGPT on a tokenized sprite dataset."""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.model import PixelGPT
from src.tokenizer import DescriptionTokenizer


class SpriteDataset(Dataset):
    """PyTorch dataset that pairs pixel token sequences with description tokens.

    Args:
        tokens: (N, seq_len) array of pixel palette indices.
        descriptions: List of description strings, same order as tokens.
        tokenizer: Fitted DescriptionTokenizer.
    """

    def __init__(
        self,
        tokens: np.ndarray,
        descriptions: list[str],
        tokenizer: DescriptionTokenizer,
    ) -> None:
        self.tokens = tokens
        self.descriptions = descriptions
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pixel_seq = self.tokens[idx]  # (4096,)
        desc = self.descriptions[idx]

        # Encode description: [DESC_START, word1, word2, ..., DESC_END]
        desc_ids = self.tokenizer.encode(desc)

        # Build full training sequence:
        # desc_tokens:  [desc_id0, desc_id1, ..., IMG_START, 0,     0,     ..., 0    ]
        # pixel_tokens: [0,        0,        ..., 0,         pix_0, pix_1, ..., pix_N]
        # is_pixel:     [False,    False,    ..., False,      True,  True,  ..., True ]
        #
        # Target (for loss): predict pixel_tokens shifted left by 1 at pixel positions.

        img_start = self.tokenizer.img_start_id
        n_desc = len(desc_ids) + 1  # +1 for IMG_START
        n_pixel = len(pixel_seq)
        seq_len = n_desc + n_pixel

        desc_tok = torch.zeros(seq_len, dtype=torch.long)
        pixel_tok = torch.zeros(seq_len, dtype=torch.long)
        is_pixel = torch.zeros(seq_len, dtype=torch.bool)

        # Fill description portion
        for i, d in enumerate(desc_ids):
            desc_tok[i] = d
        desc_tok[len(desc_ids)] = img_start  # IMG_START as last desc token

        # Fill pixel portion
        for i, p in enumerate(pixel_seq):
            pixel_tok[n_desc + i] = p
            is_pixel[n_desc + i] = True

        # Target: next pixel token at each position (shifted left by 1)
        # We only compute loss on pixel positions.
        # Input position n_desc + i should predict pixel_seq[i+1]
        target = torch.full((seq_len,), -100, dtype=torch.long)  # -100 = ignore
        # The token at IMG_START should predict pixel_seq[0]
        target[len(desc_ids)] = int(pixel_seq[0])
        for i in range(n_pixel - 1):
            target[n_desc + i] = int(pixel_seq[i + 1])

        return {
            "desc_tokens": desc_tok,
            "pixel_tokens": pixel_tok,
            "is_pixel": is_pixel,
            "target": target,
        }


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad sequences in a batch to the same length.

    Args:
        batch: List of sample dicts from SpriteDataset.

    Returns:
        Batched and padded tensors.
    """
    max_len = max(b["desc_tokens"].shape[0] for b in batch)

    desc_tokens = torch.zeros(len(batch), max_len, dtype=torch.long)
    pixel_tokens = torch.zeros(len(batch), max_len, dtype=torch.long)
    is_pixel = torch.zeros(len(batch), max_len, dtype=torch.bool)
    target = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, b in enumerate(batch):
        seq_len = b["desc_tokens"].shape[0]
        desc_tokens[i, :seq_len] = b["desc_tokens"]
        pixel_tokens[i, :seq_len] = b["pixel_tokens"]
        is_pixel[i, :seq_len] = b["is_pixel"]
        target[i, :seq_len] = b["target"]

    return {
        "desc_tokens": desc_tokens,
        "pixel_tokens": pixel_tokens,
        "is_pixel": is_pixel,
        "target": target,
    }


def train_one_epoch(
    model: PixelGPT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    """Train for one epoch and return average loss.

    Args:
        model: The PixelGPT model.
        loader: Training DataLoader.
        optimizer: Optimizer.
        device: Device string.

    Returns:
        Average cross-entropy loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        desc = batch["desc_tokens"].to(device)
        pixel = batch["pixel_tokens"].to(device)
        is_pix = batch["is_pixel"].to(device)
        target = batch["target"].to(device)

        logits = model(desc, pixel, is_pix)  # (B, T, pixel_vocab)

        # Flatten for cross-entropy
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target.view(-1),
            ignore_index=-100,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def main() -> None:
    """Run the training loop."""
    parser = argparse.ArgumentParser(description="Train PixelGPT.")
    parser.add_argument("dataset", help="Name of dataset folder under data/.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs.")
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

    # Load data
    paths = config.dataset_paths(args.dataset)
    tokens = np.load(paths["processed"] / "tokens.npy")
    palette = np.load(paths["processed"] / "palette.npy")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))

    descriptions = [entry.get("description", "") for entry in manifest]
    assert len(descriptions) == tokens.shape[0], "Manifest/tokens count mismatch"

    # Build tokenizer
    tokenizer = DescriptionTokenizer.from_manifest(paths["manifest"])
    print(f"Description vocab size: {tokenizer.vocab_size}")
    print(f"Pixel vocab size (palette): {palette.shape[0]}")
    print(f"Dataset: {tokens.shape[0]} sprites, {tokens.shape[1]} pixels each")

    # Create dataset and loader
    dataset = SpriteDataset(tokens, descriptions, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Compute max sequence length: longest desc + IMG_START + pixels
    max_desc_len = max(len(tokenizer.encode(d)) for d in descriptions) + 1  # +1 IMG_START
    max_seq_len = max_desc_len + tokens.shape[1]
    print(f"Max sequence length: {max_seq_len}")

    # Build model
    model = PixelGPT(
        desc_vocab_size=tokenizer.vocab_size,
        pixel_vocab_size=palette.shape[0],
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=max_seq_len,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Setup logging
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))

    # Save hyperparameters
    hparams = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
        "device": device,
        "desc_vocab_size": tokenizer.vocab_size,
        "pixel_vocab_size": palette.shape[0],
        "max_seq_len": max_seq_len,
        "n_params": n_params,
    }
    (run_dir / "hparams.json").write_text(json.dumps(hparams, indent=2))

    # Save tokenizer vocab for generation
    tokenizer.save(run_dir / "vocab.json")

    # Checkpoint dir for this run
    ckpt_dir = config.CHECKPOINT_DIR / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")

    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        avg_loss = train_one_epoch(model, loader, optimizer, device)
        elapsed = time.time() - t0

        writer.add_scalar("loss/train", avg_loss, epoch)
        print(f"Epoch {epoch:>4d}/{args.epochs}  loss={avg_loss:.4f}  time={elapsed:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "hparams": hparams,
            }, ckpt_dir / "best.pt")

        if epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "hparams": hparams,
            }, ckpt_dir / f"epoch_{epoch}.pt")

    writer.close()
    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Logs: {run_dir}")
    print(f"Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    main()
