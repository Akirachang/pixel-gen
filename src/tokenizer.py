"""Tokenizer for text descriptions used in conditional sprite generation."""

import json
from pathlib import Path


# Special tokens
PAD_TOKEN = "<PAD>"
DESC_START_TOKEN = "<DESC_START>"
DESC_END_TOKEN = "<DESC_END>"
IMG_START_TOKEN = "<IMG_START>"
IMG_END_TOKEN = "<IMG_END>"

SPECIAL_TOKENS = [PAD_TOKEN, DESC_START_TOKEN, DESC_END_TOKEN, IMG_START_TOKEN, IMG_END_TOKEN]


class DescriptionTokenizer:
    """Builds a vocabulary from description strings and encodes/decodes them.

    Args:
        vocab: Optional pre-built vocab dict mapping token -> id.
    """

    def __init__(self, vocab: dict[str, int] | None = None) -> None:
        if vocab is not None:
            self.token_to_id = dict(vocab)
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        else:
            self.token_to_id = {}
            self.id_to_token = {}

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> "DescriptionTokenizer":
        """Build a tokenizer from all descriptions in a manifest file.

        Args:
            manifest_path: Path to manifest.json.

        Returns:
            A fitted DescriptionTokenizer.
        """
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        words: set[str] = set()
        for entry in manifest:
            desc = entry.get("description", "")
            words.update(desc.split())

        vocab: dict[str, int] = {}
        for token in SPECIAL_TOKENS:
            vocab[token] = len(vocab)
        for word in sorted(words):
            if word not in vocab:
                vocab[word] = len(vocab)

        return cls(vocab=vocab)

    def encode(self, description: str) -> list[int]:
        """Encode a description string into a list of token IDs.

        Args:
            description: Space-separated description (e.g. "grass poison quadruped green").

        Returns:
            List of integer token IDs including start/end markers.
        """
        ids = [self.token_to_id[DESC_START_TOKEN]]
        for word in description.split():
            ids.append(self.token_to_id[word])
        ids.append(self.token_to_id[DESC_END_TOKEN])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back into a description string.

        Args:
            ids: List of integer token IDs.

        Returns:
            Decoded description string (special tokens excluded).
        """
        words = []
        for token_id in ids:
            token = self.id_to_token[token_id]
            if token in SPECIAL_TOKENS:
                continue
            words.append(token)
        return " ".join(words)

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the vocabulary."""
        return len(self.token_to_id)

    @property
    def img_start_id(self) -> int:
        """Token ID for IMG_START."""
        return self.token_to_id[IMG_START_TOKEN]

    @property
    def img_end_id(self) -> int:
        """Token ID for IMG_END."""
        return self.token_to_id[IMG_END_TOKEN]

    @property
    def pad_id(self) -> int:
        """Token ID for PAD."""
        return self.token_to_id[PAD_TOKEN]

    def save(self, path: Path) -> None:
        """Save vocabulary to a JSON file.

        Args:
            path: Output file path.
        """
        path.write_text(json.dumps(self.token_to_id, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DescriptionTokenizer":
        """Load a tokenizer from a saved vocabulary file.

        Args:
            path: Path to vocab JSON file.

        Returns:
            A DescriptionTokenizer with the loaded vocabulary.
        """
        vocab = json.loads(path.read_text(encoding="utf-8"))
        return cls(vocab=vocab)
