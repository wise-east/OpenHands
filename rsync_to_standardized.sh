#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$(dirname "$SCRIPT_DIR")/standardized/runnable"

mkdir -p "$TARGET"

rsync -av --delete --omit-dir-times \
    --exclude='.git' \
    --exclude='.gitignore' \
    --exclude='.venv' \
    --exclude='.tmp' \
    --exclude='.cache' \
    --exclude='__pycache__' \
    --exclude='uv.lock' \
    --exclude='rsync_to_standardized.sh' \
    "$SCRIPT_DIR/" "$TARGET/"

echo "Synced $SCRIPT_DIR -> $TARGET"
