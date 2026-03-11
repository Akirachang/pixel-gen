"""Scrape Gen 3 (Ruby/Sapphire) front-facing sprites from PokemonDB."""

import hashlib
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config

# ── Pokemon-specific constants ─────────────────────────
SPRITE_BASE_URL = "https://img.pokemondb.net/sprites/ruby-sapphire/normal"
POKEDEX_URL = "https://pokemondb.net/pokedex/national"
GEN3_MAX_DEX = 386
USER_AGENT = "pixel-gen sprite scraper (educational project)"


def fetch_pokemon_slugs() -> list[str]:
    """Scrape the national Pokédex page for URL slugs of Gen 1–3 Pokémon.

    Returns:
        List of URL-friendly Pokémon names (e.g. ['bulbasaur', 'ivysaur', ...]).
    """
    import re

    req = Request(POKEDEX_URL, headers={"User-Agent": USER_AGENT})
    html = urlopen(req).read().decode("utf-8")

    # Each entry looks like: <a href="/pokedex/bulbasaur">
    # followed by a dex number in a <small> or <span> tag.
    # We parse all /pokedex/{slug} links and keep the first GEN3_MAX_DEX unique ones.
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


def download_sprite(slug: str, out_dir: Path) -> dict | None:
    """Download a single sprite PNG and return its manifest entry.

    Args:
        slug: URL-friendly Pokémon name.
        out_dir: Directory to save the PNG into.

    Returns:
        Dict with filename, url, and sha256, or None on failure.
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

    return {"filename": filename, "url": url, "sha256": sha256}


def main() -> None:
    """Run the full scraping pipeline."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching Pokémon slugs from national dex...")
    slugs = fetch_pokemon_slugs()
    print(f"Found {len(slugs)} Pokémon (expected {GEN3_MAX_DEX})")

    manifest: list[dict] = []
    for i, slug in enumerate(slugs, 1):
        print(f"[{i}/{len(slugs)}] {slug}")
        entry = download_sprite(slug, config.RAW_DIR)
        if entry:
            manifest.append(entry)
        time.sleep(config.SCRAPE_DELAY)

    config.MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"\nDone — {len(manifest)} sprites saved to {config.RAW_DIR}")
    print(f"Manifest written to {config.MANIFEST_PATH}")


if __name__ == "__main__":
    main()
