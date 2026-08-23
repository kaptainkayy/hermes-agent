---
name: annas-archive-search
description: Search Anna's Archive book/metadata catalogue via local torrent metadata. No web scraping — uses the locally-downloaded `aa_derived_mirror_metadata` torrent.
version: 1.0.0
author: audio-agents
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [search, annas-archive, books, metadata, torrent, research]
    related_skills: [duckduckgo-search]
    fallback_for_toolsets: []
---

# Anna's Archive Search

Search the Anna's Archive metadata catalogue using a local copy of the
`aa_derived_mirror_metadata` torrent. Per Anna's Archive
[guidance](https://annas-archive.gl/blog/llms-txt.html), this does **not**
scrape the live website — it queries the offline torrent metadata instead.

## Prerequisites

### One-time setup

1. **Get the metadata torrent:**
   ```bash
   cd /home/kayai3/AiSetupResearch/annas-archive-toolchain-mcp
   uv run python -m annas.cli torrent list --metadata-only
   ```
   Find the `aa_derived_mirror_metadata` group and grab its magnet link.

2. **Download + expand** on a seedbox/VPS via your BitTorrent client.

3. **Sync locally:**
   ```bash
   rsync -avP seedbox:~/downloads/aa_derived_mirror_metadata/ /path/to/local/
   ```

4. **Set env var:**
   ```bash
   export ANNAS_METADATA_DIR=/path/to/aa_derived_mirror_metadata
   ```

5. **(Optional)** Build a SQLite FTS5 index for faster repeated searches:
   ```bash
   uv run python -m annas.cli torrent index $ANNAS_METADATA_DIR --db ~/annas_index.db
   ```

## Detection Flow

```bash
# Check project exists
test -d /home/kayai3/AiSetupResearch/annas-archive-toolchain-mcp && echo "PROJECT=exists" || echo "PROJECT=missing"

# Check metadata dir
test -d "${ANNAS_METADATA_DIR:-}" && echo "METADATA_DIR=set" || echo "METADATA_DIR=unset"
```

## Usage

### Method 1: CLI Script (Preferred)

Uses the bundled helper script for quick searches:

```bash
# Search by author/title
/home/kayai3/.hermes/hermes-agent/optional-skills/research/annas-archive-search/scripts/annas-archive.sh "Robert A Johnson" 10

# Search with format filter
/home/kayai3/.hermes/hermes-agent/optional-skills/research/annas-archive-search/scripts/annas-archive.sh "Jung" 5 epub

# Search with higher limit
/home/kayai3/.hermes/hermes-agent/optional-skills/research/annas-archive-search/scripts/annas-archive.sh "philosophy" 20 pdf
```

### Method 2: Direct CLI

For more control (flags, JSON output, extension filter):

```bash
# Basic search
cd /home/kayai3/AiSetupResearch/annas-archive-toolchain-mcp && uv run python -m annas.cli search-catalog "Robert A Johnson"

# With extension filter
uv run python -m annas.cli search-catalog "Jung" --ext epub

# JSON output for scripting
uv run python -m annas.cli search-catalog "Plato" --json

# Limit results
uv run python -m annas.cli search-catalog "philosophy" --limit 5
```

### Method 3: Faster Indexed Search

After building the index (see step 5 above):

```bash
uv run python -m annas.cli torrent search "Robert A Johnson" --db ~/annas_index.db

# With format filter
uv run python -m annas.cli torrent search "Jung" --db ~/annas_index.db --ext epub
```

### Listing Available Torrents

```bash
cd /home/kayai3/AiSetupResearch/annas-archive-toolchain-mcp && uv run python -m annas.cli torrent list
```

### Filtering to metadata-only torrents

```bash
cd /home/kayai3/AiSetupResearch/annas-archive-toolchain-mcp && uv run python -m annas.cli torrent list --metadata-only
```

## Output

The CLI renders a Rich table with columns:

| Title | Format / Size | Language | Year | Category | MD5 |
|-------|--------------|----------|------|----------|-----|
| The Republic | EPUB / 1.0MB | English | 380 | Philosophy | a1b2c3d4... |

Titles are clickable links to `https://annas-archive.org/md5/<hash>`.

## Downloading a Found Book

Once you have an MD5 from search results, you can download the actual file:

```bash
cd /home/kayai3/AiSetupResearch/annas-archive-toolchain-mcp && uv run python -m annas.cli download <md5> --secret-key "$ANNAS_SECRET_KEY" --work-dir /tmp/annas
```

Requires `ANNAS_SECRET_KEY` (obtained via donation on Anna's Archive).

## Limitations

- **Requires local metadata download**: The `aa_derived_mirror_metadata` torrent must be downloaded and synced first (~hundreds of GB).
- **Not live**: Metadata freshness depends on how recently you synced the torrent.
- **No web scraping**: By design — this respects Anna's Archive's request that machines use bulk downloads instead.
- **Download requires API key**: Individual file downloads need `ANNAS_SECRET_KEY` from a donation.

## Troubleshooting

| Problem | Likely Cause | What To Do |
|---------|--------------|------------|
| `ANNAS_METADATA_DIR not set` | Missing env var | `export ANNAS_METADATA_DIR=/path/to/metadata` |
| `No .jsonl files found` | Wrong directory or torrent not expanded | Point to the expanded torrent root, not a subdir |
| `metadata_dir not found` | Path doesn't exist | Verify the directory exists |
| Search returns 0 results | Query too specific or metadata not synced | Try broader terms, check torrent is up to date |
| `uv: command not found` | UV not installed | Install via `curl -LsSf https://astral.sh/uv/install.sh | sh` |

## Pitfalls

- **Periods in author names**: `"Robert A Johnson"` with period `"Robert A. Johnson"` — the search strips punctuation so both work.
- **Metadata sync lag**: The torrent is periodically updated. Re-sync to get new additions.
- **Large torrent**: The full metadata torrent is hundreds of GB. Only download if you have the bandwidth and storage.
- **Seedbox recommended**: Your real IP is exposed when torrenting. Use a seedbox or VPN.

## Validated With

Validated against `annas-archive-toolchain-mcp` v0.1.0. Search returns results from locally-indexed JSONL metadata.