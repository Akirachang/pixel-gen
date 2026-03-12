"""Scrape Gen 3 (Ruby/Sapphire) front-facing sprites and metadata from PokemonDB."""

import hashlib
import json
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config

# ── Pokemon-specific constants ─────────────────────────
DATASET_NAME = "pokemondb-gen3"
SPRITE_BASE_URL = "https://img.pokemondb.net/sprites/ruby-sapphire/normal"
POKEDEX_URL = "https://pokemondb.net/pokedex/national"
POKEMON_DETAIL_URL = "https://pokemondb.net/pokedex"
GEN3_MAX_DEX = 386
USER_AGENT = "pixel-gen sprite scraper (educational project)"


def fetch_pokemon_slugs() -> list[str]:
    """Scrape the national Pokédex page for URL slugs of Gen 1–3 Pokémon.

    Returns:
        List of URL-friendly Pokémon names (e.g. ['bulbasaur', 'ivysaur', ...]).
    """
    req = Request(POKEDEX_URL, headers={"User-Agent": USER_AGENT})
    html = urlopen(req).read().decode("utf-8")

    pattern = re.compile(r'<a[^>]*href="/pokedex/([\w-]+)"')
    seen: set[str] = set()
    slugs: list[str] = []

    for match in pattern.finditer(html):
        slug = match.group(1)
        if slug not in seen:
            seen.add(slug)
            slugs.append(slug)
        if len(slugs) == GEN3_MAX_DEX:
            break

    return slugs


def fetch_pokemon_metadata(slug: str) -> dict:
    """Scrape type, color, and body shape from a Pokémon's detail page.

    Args:
        slug: URL-friendly Pokémon name.

    Returns:
        Dict with keys: types, color, shape, description.
    """
    url = f"{POKEMON_DETAIL_URL}/{slug}"
    req = Request(url, headers={"User-Agent": USER_AGENT})
    html = urlopen(req).read().decode("utf-8")

    # Extract types (e.g. "grass", "poison")
    type_pattern = re.compile(r'class="type-icon type-(\w+)"')
    types = list(dict.fromkeys(type_pattern.findall(html)))

    # Extract color from the vitals table
    color = ""
    color_match = re.search(r'<th>Color</th>\s*<td>([\w\s]+)</td>', html)
    if color_match:
        color = color_match.group(1).strip().lower()

    # Extract body shape
    shape = ""
    shape_match = re.search(
        r'<th>Shape</th>\s*<td>\s*<a[^>]*title="([^"]+)"', html
    )
    if shape_match:
        shape = shape_match.group(1).strip().lower()

    # Build flat description string for training
    parts = [t.lower() for t in types]
    if shape:
        parts.append(shape)
    if color:
        parts.append(color)
    description = " ".join(parts)

    return {
        "types": [t.lower() for t in types],
        "color": color,
        "shape": shape,
        "description": description,
    }


def download_sprite(slug: str, out_dir: Path) -> dict | None:
    """Download a single sprite PNG and return its manifest entry.

    Args:
        slug: URL-friendly Pokémon name.
        out_dir: Directory to save the PNG into.

    Returns:
        Dict with filename, label, url, and sha256, or None on failure.
    """
    url = f"{SPRITE_BASE_URL}/{slug}.png"
    filename = f"{slug}.png"
    filepath = out_dir / filename

    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        data = urlopen(req).read()
    except HTTPError as e:
        print(f"  SKIP {slug}: HTTP {e.code}")
        return None

    filepath.write_bytes(data)
    sha256 = hashlib.sha256(data).hexdigest()

    return {"filename": filename, "label": slug, "url": url, "sha256": sha256}


def main() -> None:
    """Run the full scraping pipeline."""
    paths = config.dataset_paths(DATASET_NAME)
    raw_dir = paths["raw"]
    manifest_path = paths["manifest"]

    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching Pokémon slugs from national dex...")
    slugs = fetch_pokemon_slugs()
    print(f"Found {len(slugs)} Pokémon (expected {GEN3_MAX_DEX})")

    manifest: list[dict] = []
    for slug in tqdm(slugs, desc="Scraping"):
        entry = download_sprite(slug, raw_dir)
        if entry is None:
            continue

        try:
            metadata = fetch_pokemon_metadata(slug)
            entry.update(metadata)
        except HTTPError as e:
            print(f"  WARN: metadata failed for {slug}: HTTP {e.code}")

        manifest.append(entry)
        time.sleep(config.SCRAPE_DELAY)

    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"\nDone — {len(manifest)} sprites saved to {raw_dir}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
