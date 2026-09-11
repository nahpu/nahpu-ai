#!/usr/bin/env bash
# Fetch model weights into models/. Large and gitignored, so fetch once per machine.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_DIR="${NAHPU_MODEL_DIR:-$ROOT/models}"

# name|url|sha256 ("-" until pinned)
MODELS="
all-MiniLM-L6-v2.onnx|https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx|-
"

print_usage() {
    echo "usage: $0 [model-name ...]    (default: all)"
    echo
    echo "available:"
    list_model_names
}

list_model_names() {
    echo "$MODELS" | while IFS='|' read -r name _ _; do
        [ -n "$name" ] && echo "  $name"
        true
    done
}

checksum_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

download_to() {
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

verify_or_report() {
    local name="$1" dest="$2" expected="$3"
    local actual
    actual="$(checksum_of "$dest")"

    if [ -z "$actual" ]; then
        echo "  warning: no sha256 tool, skipping verification" >&2
    elif [ "$expected" = "-" ]; then
        echo "  sha256 $actual"
        echo "  (not pinned — paste this into scripts/fetch-model.sh)"
    elif [ "$actual" != "$expected" ]; then
        echo "  error: sha256 mismatch for $name" >&2
        echo "    expected $expected" >&2
        echo "    actual   $actual" >&2
        exit 1
    else
        echo "  sha256 ok"
    fi
}

fetch_model() {
    local name="$1" url="$2" expected="$3"
    local dest="$DEST_DIR/$name"

    if [ -f "$dest" ]; then
        echo "already present: $name"
    else
        echo "downloading $name"
        mkdir -p "$DEST_DIR"
        download_to "$url" "$dest"
    fi

    verify_or_report "$name" "$dest" "$expected"
}

case "${1:-}" in
    -h|--help) print_usage; exit 0 ;;
esac

# Space-padded so a substring test works on the bash 3.2 that macOS ships.
REQUESTED=" $* "

echo "$MODELS" | while IFS='|' read -r name url sha; do
    [ -z "$name" ] && continue
    if [ -n "$*" ] && [ "${REQUESTED#* $name }" = "$REQUESTED" ]; then
        continue
    fi
    fetch_model "$name" "$url" "$sha"
done

echo "models in $DEST_DIR"
