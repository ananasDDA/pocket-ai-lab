#!/usr/bin/env bash
#
# build_llama_mtmd_xcframework.sh
#
# Builds a custom llama.cpp XCFramework that includes `libmtmd` (the
# multimodal projector runtime) and the iOS-friendly C API from
# `tools/mtmd/mtmd-ios.h`. The official GitHub Releases XCFrameworks
# (e.g. b8882) are built with `LLAMA_BUILD_TOOLS=OFF` and therefore
# contain only the text-only `llama` module.
#
# The script:
#   1. Clones ggml-org/llama.cpp into build-tmp/llama.cpp at $LLAMA_REV.
#   2. Applies the four modifications from upstream commit 4d32fd29
#      ("feat: mtmd support xcframework") to build-xcframework.sh so
#      that mtmd headers + libmtmd.a + libcommon.a end up in the
#      framework.
#   3. Optionally trims the script to only build for the platforms
#      listed in $PLATFORMS (default: ios-device,ios-sim,macos) to
#      keep build time reasonable.
#   4. Runs the patched build-xcframework.sh.
#   5. Replaces Vendor/llama.xcframework with the mtmd-enabled build.
#
# Environment:
#   LLAMA_REV    git revision / tag / branch to build.        (default: master)
#   PLATFORMS    comma-separated list of platforms to build.  (default: ios-device,ios-sim,macos)
#                allowed values: ios-device, ios-sim, macos,
#                visionos, visionos-sim, tvos-sim, tvos-device
#   JOBS         cmake parallel jobs.                         (default: $(sysctl -n hw.ncpu))
#
# Host requirements:
#   * macOS with Xcode 15+ installed (xcode-select -p must point to Xcode)
#   * CMake >= 3.28 (brew install cmake)
#   * ~5 GB free disk space, ~15–45 min build time on an M-series Mac.
#
# Upstream reference: https://github.com/ggml-org/llama.cpp/commit/4d32fd29c54417f7f2037efbcc38f77b3772d223
#

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="$ROOT/Vendor"
TMP="$ROOT/build-tmp"
SRC="$TMP/llama.cpp"

LLAMA_REV="${LLAMA_REV:-master}"
PLATFORMS="${PLATFORMS:-ios-device,ios-sim,macos}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu)}"

mkdir -p "$TMP" "$VENDOR"

require() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "error: '$1' not found on PATH. $2" >&2
        exit 1
    }
}

require git   "install Xcode Command Line Tools or Homebrew"
require cmake "brew install cmake"
require xcrun "xcode-select --install (or sudo xcode-select -s /Applications/Xcode.app/Contents/Developer)"

# Active developer dir must be a full Xcode, not just CLT.
if ! xcrun --find xcodebuild >/dev/null 2>&1; then
    echo "error: xcodebuild not found. Run: sudo xcode-select -s /Applications/Xcode.app/Contents/Developer" >&2
    exit 1
fi

# --------------------------------------------------------------------
# Step 1 — clone / checkout
# --------------------------------------------------------------------

if [[ ! -d "$SRC/.git" ]]; then
    echo "→ Cloning ggml-org/llama.cpp into $SRC"
    git clone --filter=blob:none https://github.com/ggml-org/llama.cpp "$SRC"
fi

pushd "$SRC" >/dev/null
if [[ "${NO_CLEAN:-0}" != "1" ]]; then
    git fetch --tags --force --prune
    git reset --hard
    git clean -fdx
    git checkout "$LLAMA_REV"
else
    echo "→ NO_CLEAN=1: reusing existing checkout and cmake build dirs"
    # Still restore build-xcframework.sh so the patcher below sees a clean baseline.
    git checkout -- build-xcframework.sh
fi
LLAMA_SHA="$(git rev-parse --short HEAD)"
echo "→ Using llama.cpp at $LLAMA_SHA ($LLAMA_REV)"

if [[ ! -f build-xcframework.sh ]]; then
    echo "error: build-xcframework.sh not present in $LLAMA_REV. Pin LLAMA_REV to a revision that still ships it (e.g. tag b8882)." >&2
    exit 1
fi

# --------------------------------------------------------------------
# Step 2 — patch build-xcframework.sh to enable mtmd
# (mirrors upstream commit 4d32fd29)
# --------------------------------------------------------------------

echo "→ Patching build-xcframework.sh (enabling LLAMA_BUILD_TOOLS + mtmd headers/libs)"

# 2a. Flip the flag.
sed -i '' 's/^LLAMA_BUILD_TOOLS=OFF$/LLAMA_BUILD_TOOLS=ON/' build-xcframework.sh

# 2b. Copy mtmd headers alongside the ggml ones. We anchor on the line
#     that copies gguf.h, which is the last header copy in setup_framework_structure.
#
# Note: upstream used to ship a thin iOS-only wrapper (`mtmd-ios.h`); it was
# removed from master and the main C API lives in `mtmd.h` + `mtmd-helper.h`.
# We guard each cp with `[ -f ... ]` so the script works on both older and
# newer revisions without breaking.
python3 - <<'PY'
import pathlib, re
p = pathlib.Path("build-xcframework.sh")
s = p.read_text()

# Insert header copies right after gguf.h (dynamic so missing headers are skipped).
header_snippet = (
    "\n"
    "    # Copy mtmd headers and dependencies (added by our script)\n"
    "    for _h in mtmd.h mtmd-helper.h mtmd-audio.h mtmd-image.h mtmd-ios.h; do\n"
    "        if [ -f \"tools/mtmd/${_h}\" ]; then\n"
    "            cp \"tools/mtmd/${_h}\" \"${header_path}\"\n"
    "        fi\n"
    "    done\n"
)
s = s.replace(
    "    cp ggml/include/gguf.h         ${header_path}\n",
    "    cp ggml/include/gguf.h         ${header_path}\n" + header_snippet,
    1,
)

# Regenerate the modulemap heredoc to only reference headers that were actually copied.
#
# Design: `llama.h`, `mtmd.h`, `mtmd-helper.h` are exposed as real `header`
# directives so Swift `import llama` sees the whole public API (llama.cpp
# text engine + libmtmd vision/audio bridge).
#
# `ggml*.h` and `gguf.h` are `textual header` on purpose: they define
# conflicting enums (e.g. `ggml_type`) if a second framework (like
# whisper.xcframework) is ever linked that ships its own `ggml.h`. Keeping
# them textual isolates each framework's ODR world.
modulemap_snippet = (
    '    cat > ${module_path}module.modulemap << EOF\n'
    'framework module llama {\n'
    '    header "llama.h"\n'
    '    textual header "ggml.h"\n'
    '    textual header "ggml-alloc.h"\n'
    '    textual header "ggml-backend.h"\n'
    '    textual header "ggml-metal.h"\n'
    '    textual header "ggml-cpu.h"\n'
    '    textual header "ggml-blas.h"\n'
    '    textual header "ggml-opt.h"\n'
    '    textual header "gguf.h"\n'
)
# The existing snippet begins with `cat > ${module_path}module.modulemap` and
# ends at the line right before `link "c++"`. Locate and replace it.
pat_start = s.find('    cat > ${module_path}module.modulemap << EOF\n')
if pat_start == -1:
    raise SystemExit("patch failed: modulemap heredoc not found")
pat_end_needle = '\n    link "c++"\n'
pat_end = s.find(pat_end_needle, pat_start)
if pat_end == -1:
    raise SystemExit("patch failed: end of modulemap heredoc not found")
# Build new modulemap body that conditionally adds mtmd headers via shell.
# These are NOT textual — they form the public vision/audio C API that
# Swift must see through `import llama`.
new_block = (
    modulemap_snippet
    + '$(\n'
    + '    for _h in mtmd.h mtmd-helper.h mtmd-audio.h mtmd-image.h mtmd-ios.h; do\n'
    + '        [ -f "tools/mtmd/${_h}" ] && printf \'    header "%s"\\n\' "${_h}"\n'
    + '    done\n'
    + ')'
)
s = s[:pat_start] + new_block + s[pat_end:]

# Append libmtmd.a to the static libs list. libmtmd only depends on
# ggml + llama + Threads (see tools/mtmd/CMakeLists.txt), so we
# intentionally do *not* drag in libllama-common (which pulls a full
# httplib/Boost dependency graph that isn't needed for on-device inference).
s = s.replace(
    '        "${base_dir}/${build_dir}/ggml/src/ggml-blas/${release_dir}/libggml-blas.a"\n    )',
    '        "${base_dir}/${build_dir}/ggml/src/ggml-blas/${release_dir}/libggml-blas.a"\n'
    '        "${base_dir}/${build_dir}/tools/mtmd/${release_dir}/libmtmd.a"\n    )',
    1,
)

p.write_text(s)
PY

# Sanity: verify the patch actually landed.
if ! grep -q "libmtmd.a" build-xcframework.sh; then
    echo "error: mtmd patch did not apply cleanly. Check upstream script layout at $LLAMA_REV." >&2
    exit 1
fi

# --------------------------------------------------------------------
# Step 3 — optionally trim the script to a subset of platforms
# (the default set skips visionOS and tvOS which saves ~60% of the build time).
# --------------------------------------------------------------------

wants() { [[ ",$PLATFORMS," == *",$1,"* ]]; }

strip_block() {
    # Removes a build-and-combine block identified by the "Building for $1..." comment.
    # Uses a line-based state machine (via python) for robustness.
    local label="$1"
    python3 - "$label" <<'PY'
import re, sys, pathlib
label = sys.argv[1]
# Map platform label → pretty "Building for ..." banner
banner_map = {
    "ios-sim":      "iOS simulator",
    "ios-device":   "iOS devices",
    "macos":        "macOS",
    "visionos":     "visionOS",
    "visionos-sim": "visionOS simulator",
    "tvos-sim":     "tvOS simulator",
    "tvos-device":  "tvOS devices",
}
banner = banner_map[label]

p = pathlib.Path("build-xcframework.sh")
lines = p.read_text().splitlines(keepends=True)
out = []
i = 0
removed = False
while i < len(lines):
    line = lines[i]
    # Match the start of this platform's build block.
    if not removed and line.strip() == f'echo "Building for {banner}..."':
        # Skip forward until the next block start ("Building for" / "# Setup" / "# Add").
        j = i + 1
        while j < len(lines):
            nxt = lines[j].lstrip()
            if nxt.startswith('echo "Building for ') \
               or nxt.startswith('# Setup frameworks') \
               or nxt.startswith('# Add tvOS builds'):
                break
            j += 1
        i = j
        removed = True
        continue
    # Remove the per-platform setup/combine lines.
    if (f'setup_framework_structure "build-{label}"' in line
        or f'combine_static_libraries "build-{label}"' in line):
        i += 1
        continue
    # Remove the per-platform -framework / -debug-symbols pair in xcodebuild -create-xcframework.
    if f'-framework $(pwd)/build-{label}/framework/llama.framework' in line:
        i += 2  # also skip the accompanying -debug-symbols line
        continue
    out.append(line)
    i += 1

if not removed:
    print(f"warn: could not find build block for {label}", file=sys.stderr)
p.write_text("".join(out))
PY
}

for platform in ios-sim ios-device macos visionos visionos-sim tvos-sim tvos-device; do
    if ! wants "$platform"; then
        echo "→ Skipping platform: $platform"
        strip_block "$platform"
    fi
done

# --------------------------------------------------------------------
# Step 4 — run the patched script
# --------------------------------------------------------------------

echo "→ Running patched build-xcframework.sh (this takes 15–45 min; logs stream below)"
chmod +x build-xcframework.sh
./build-xcframework.sh

if [[ ! -d "build-apple/llama.xcframework" ]]; then
    echo "error: build-apple/llama.xcframework not produced. Check logs above." >&2
    exit 1
fi

# --------------------------------------------------------------------
# Step 5 — install into Vendor/
# --------------------------------------------------------------------

rm -rf "$VENDOR/llama.xcframework"
mv "build-apple/llama.xcframework" "$VENDOR/llama.xcframework"

popd >/dev/null

echo
echo "✓ Custom mtmd-enabled llama.xcframework installed at:"
echo "    $VENDOR/llama.xcframework"
echo "  Built from llama.cpp $LLAMA_SHA"
echo "  Platforms: $PLATFORMS"
echo
echo "Next steps:"
echo "  1. In Xcode, remove any previous llama.xcframework reference from the"
echo "     local_ai_test target and drag the new one from $VENDOR."
echo "  2. Verify the produced module.modulemap contains \`header \"mtmd.h\"\`"
echo "     and \`header \"mtmd-helper.h\"\`. LlamaCppVisionEngine + BackendAvailability"
echo "     are already wired to these symbols and will flip live on a clean build."
