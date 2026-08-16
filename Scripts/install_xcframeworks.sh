#!/usr/bin/env bash
#
# install_xcframeworks.sh
#
# Downloads prebuilt XCFrameworks for llama.cpp and whisper.cpp from the
# official GitHub Releases and unpacks them into ./Vendor/ at the repo
# root. These XCFrameworks export the `llama` and `whisper` Clang
# modules, which the app's engines import via `#if canImport(llama)`
# and `#if canImport(whisper)`.
#
# After running this script you still need to drag the resulting
# `Vendor/llama.xcframework` and `Vendor/whisper.xcframework` into the
# `local_ai_test` target in Xcode (Frameworks, Libraries, and Embedded
# Content → Embed & Sign). See MULTI_BACKEND_INTEGRATION.md §3.1.
#
# Versions are pinned here for reproducibility; bump as needed.
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="$ROOT/Vendor"
mkdir -p "$VENDOR"

LLAMA_TAG="${LLAMA_TAG:-b8882}"
WHISPER_TAG="${WHISPER_TAG:-v1.8.4}"

LLAMA_URL="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/llama-${LLAMA_TAG}-xcframework.zip"
WHISPER_URL="https://github.com/ggml-org/whisper.cpp/releases/download/${WHISPER_TAG}/whisper-${WHISPER_TAG}-xcframework.zip"

fetch_and_extract() {
    local name="$1"
    local url="$2"
    local tmp_dir
    tmp_dir="$(mktemp -d)"
    local zip_path="$tmp_dir/${name}.zip"

    echo "→ Downloading $name from $url"
    curl -fL --progress-bar -o "$zip_path" "$url"

    echo "→ Unzipping $name"
    unzip -q "$zip_path" -d "$tmp_dir"

    # Expect exactly one *.xcframework at the top level of the archive.
    local xcf
    xcf="$(find "$tmp_dir" -maxdepth 2 -name '*.xcframework' -print -quit)"
    if [[ -z "$xcf" ]]; then
        echo "✗ $name: .xcframework not found inside archive" >&2
        exit 1
    fi

    local dest="$VENDOR/$(basename "$xcf")"
    rm -rf "$dest"
    mv "$xcf" "$dest"
    rm -rf "$tmp_dir"
    echo "✓ $name installed at $dest"
}

fetch_and_extract "llama" "$LLAMA_URL"
fetch_and_extract "whisper" "$WHISPER_URL"

echo
echo "Done. Next steps in Xcode:"
echo "  1. Drag  Vendor/llama.xcframework   onto the local_ai_test target"
echo "     → General → Frameworks, Libraries, and Embedded Content"
echo "     → Embed: Embed & Sign"
echo "  2. Repeat for Vendor/whisper.xcframework."
echo "  3. Build. BackendAvailability.hasLlama and hasWhisper will flip to true."
