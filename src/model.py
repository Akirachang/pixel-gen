"""GPT-style transformer for conditional pixel art generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention.

    Args:
        d_model: Embedding dimension.
        n_heads: Number of attention heads.
        max_seq_len: Maximum sequence length for causal mask.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.resid_drop(self.out_proj(out))


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture.

    Args:
        d_model: Embedding dimension.
        n_heads: Number of attention heads.
        max_seq_len: Maximum sequence length.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class PixelGPT(nn.Module):
    """Autoregressive transformer for conditional pixel art generation.

    The model processes a concatenated sequence of:
        [DESC_START] desc_tok_1 ... desc_tok_n [DESC_END] [IMG_START] pix_0 pix_1 ... pix_N [IMG_END]

    Description tokens and pixel tokens share the same embedding space but use
    separate embedding tables to keep text vocabulary and color palette independent.

    Args:
        desc_vocab_size: Number of description tokens (text vocabulary).
        pixel_vocab_size: Number of pixel tokens (palette size).
        d_model: Embedding dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer blocks.
        max_seq_len: Maximum total sequence length (desc + image).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        desc_vocab_size: int,
        pixel_vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        max_seq_len: int = 4120,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.desc_vocab_size = desc_vocab_size
        self.pixel_vocab_size = pixel_vocab_size

        # Total output vocab = desc tokens + pixel tokens
        # During training, the model predicts over the full vocab
        # but loss is only computed on pixel positions.
        self.total_vocab_size = desc_vocab_size + pixel_vocab_size

        # Separate embeddings for description and pixel tokens
        self.desc_embed = nn.Embedding(desc_vocab_size, d_model)
        self.pixel_embed = nn.Embedding(pixel_vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, max_seq_len, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)

        # Output head predicts over pixel vocab only
        self.head = nn.Linear(d_model, pixel_vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        desc_tokens: torch.Tensor,
        pixel_tokens: torch.Tensor,
        is_pixel: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            desc_tokens: (B, T) description token IDs (0 where pixel).
            pixel_tokens: (B, T) pixel token IDs (0 where description).
            is_pixel: (B, T) bool mask — True for pixel positions.

        Returns:
            logits: (B, T, pixel_vocab_size) predictions over pixel vocabulary.
        """
        _, T = desc_tokens.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"

        pos = torch.arange(T, device=desc_tokens.device).unsqueeze(0)

        # Embed description and pixel tokens separately, then combine
        desc_emb = self.desc_embed(desc_tokens)   # (B, T, D)
        pixel_emb = self.pixel_embed(pixel_tokens) # (B, T, D)

        # Use is_pixel mask to select the right embedding per position
        is_pixel_f = is_pixel.unsqueeze(-1).float()  # (B, T, 1)
        tok_emb = desc_emb * (1 - is_pixel_f) + pixel_emb * is_pixel_f

        x = self.drop(tok_emb + self.pos_embed(pos))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, pixel_vocab_size)
        return logits

    @torch.no_grad()
    def generate(
        self,
        desc_ids: list[int],
        img_start_id: int,
        num_pixels: int,
        temperature: float = 1.0,
        top_k: int = 0,
        device: str = "cpu",
    ) -> list[int]:
        """Autoregressively generate pixel tokens given a description.

        Args:
            desc_ids: Encoded description token IDs (including DESC_START/END).
            img_start_id: Token ID for IMG_START (marks start of pixel generation).
            num_pixels: Number of pixel tokens to generate (e.g. 4096 for 64x64).
            temperature: Sampling temperature.
            top_k: If > 0, only sample from top-k logits.
            device: Device to run generation on.

        Returns:
            List of generated pixel token IDs.
        """
        self.eval()

        # Build initial sequence: desc tokens (as desc) + IMG_START (as desc token)
        seq_desc = torch.tensor(desc_ids + [img_start_id], device=device).unsqueeze(0)
        seq_pixel = torch.zeros_like(seq_desc)
        seq_is_pixel = torch.zeros_like(seq_desc, dtype=torch.bool)

        generated_pixels: list[int] = []

        for _ in range(num_pixels):
            logits = self.forward(seq_desc, seq_pixel, seq_is_pixel)
            next_logits = logits[0, -1, :] / temperature

            if top_k > 0:
                values, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < values[-1]] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            generated_pixels.append(next_token)

            # Append new pixel token to sequence
            new_desc = torch.tensor([[0]], device=device)
            new_pixel = torch.tensor([[next_token]], device=device)
            new_is_pixel = torch.ones(1, 1, device=device, dtype=torch.bool)

            seq_desc = torch.cat([seq_desc, new_desc], dim=1)
            seq_pixel = torch.cat([seq_pixel, new_pixel], dim=1)
            seq_is_pixel = torch.cat([seq_is_pixel, new_is_pixel], dim=1)

            # Truncate if exceeding max length (sliding window)
            if seq_desc.shape[1] > self.max_seq_len:
                seq_desc = seq_desc[:, -self.max_seq_len:]
                seq_pixel = seq_pixel[:, -self.max_seq_len:]
                seq_is_pixel = seq_is_pixel[:, -self.max_seq_len:]

        return generated_pixels
