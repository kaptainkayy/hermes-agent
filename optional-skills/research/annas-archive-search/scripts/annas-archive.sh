#!/bin/bash
# Anna's Archive Search Helper
# Searches local torrent metadata for matching books/files.
# Prerequisites: annas-archive-toolchain-mcp project, ANNAS_METADATA_DIR set
# Usage: ./annas-archive.sh <query> [limit] [extension]

set -e

PROJECT_DIR="/home/kayai3/AiSetupResearch/annas-archive-toolchain-mcp"

QUERY="$1"
LIMIT="${2:-10}"
EXT="${3:-}"

if [ -z "$QUERY" ]; then
    echo "Usage: $0 <query> [limit] [extension]"
    echo ""
    echo "Examples:"
    echo "  $0 'Robert A Johnson' 10"
    echo "  $0 'Jung' 5 epub"
    echo "  $0 'Plato' 20 pdf"
    echo ""
    echo "Requires: ANNAS_METADATA_DIR env var set to expanded aa_derived_mirror_metadata torrent"
    exit 1
fi

if [ ! -d "$PROJECT_DIR" ]; then
    echo "Error: annas-archive-toolchain-mcp project not found at $PROJECT_DIR"
    exit 1
fi

if [ -z "${ANNAS_METADATA_DIR:-}" ]; then
    echo "Error: ANNAS_METADATA_DIR not set."
    echo "Set it to the expanded aa_derived_mirror_metadata torrent directory."
    exit 1
fi

cd "$PROJECT_DIR"

CMD="uv run python -m annas.cli search-catalog \"$QUERY\" --limit $LIMIT"
if [ -n "$EXT" ]; then
    CMD="$CMD --ext $EXT"
fi

eval "$CMD"