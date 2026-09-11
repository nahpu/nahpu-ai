#!/usr/bin/env bash
# Fetch model weights into models/. Files are large and gitignored, so each
# machine fetches once.
#
# This mirrors what the NAHPU app will have to do at runtime (look up a
# manifest, download, verify, cache), so treat changes here as changes to that
# spec too.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_DIR="${NAHPU_MODEL_DIR:-$ROOT/models}"

# name | url | sha256 ("-" until pinned)
MODELS="
all-MiniLM-L6-v2.onnx|https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx|-
"

usage() {
    echo "usage: $0 [model-name ...]    (default: all)"
    echo
    echo "available:"
    echo "$MODELS" | while IFS='|' read -r name _ _; do
        [ -n "$name" ] && echo "  $name"
        true
    done
}

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | cut -d' ' -f1
    else
        echo ""
    fi
}

download() {
    local url="$1" dest="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -fSL --retry 3 -o "$dest.part" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$dest.part" "$url"
    else
        echo "error: need curl or wget" >&2
        exit 1
    fi
    mv "$dest.part" "$dest"
}

fetch_one() {
    local name="$1" url="$2" want="$3"
    local dest="$DEST_DIR/$name"

    if [ -f "$dest" ]; then
        echo "already present: $name"
    else
        echo "downloading $name"
        mkdir -p "$DEST_DIR"
        download "$url" "$dest"
    fi

    local got
    got="$(sha256_of "$dest")"
    if [ -z "$got" ]; then
        echo "  warning: no sha256 tool, skipping verification" >&2
    elif [ "$want" = "-" ]; then
        echo "  sha256 $got"
        echo "  (not pinned — paste this into scripts/fetch-model.sh)"
    elif [ "$got" != "$want" ]; then
        echo "  error: sha256 mismatch for $name" >&2
        echo "    expected $want" >&2
        echo "    got      $got" >&2
        exit 1
    else
        echo "  sha256 ok"
    fi
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

# Space-padded list so a plain substring test works on bash 3.2 (macOS ships it).
WANTED=" $* "

echo "$MODELS" | while IFS='|' read -r name url sha; do
    [ -z "$name" ] && continue
    if [ $# -gt 0 ] && [ "${WANTED#* $name }" = "$WANTED" ]; then
        continue
    fi
    fetch_one "$name" "$url" "$sha"
done

echo "models in $DEST_DIR"
