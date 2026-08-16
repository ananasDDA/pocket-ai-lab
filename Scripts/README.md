# Scripts

Shell scripts for bootstrapping the backends that live outside the app
source tree. Run everything from the project root (`local_ai_test/`).

## `install_xcframeworks.sh`

Downloads prebuilt XCFrameworks from official GitHub Releases and
unpacks them into `Vendor/`.

| Framework | Default tag | Size (download) | Notes                     |
| --------- | ----------- | --------------- | ------------------------- |
| `llama`   | `b8882`     | ~100 MB         | text-only (no mtmd)       |
| `whisper` | `v1.8.4`    | ~30 MB          | full ASR                  |

Override pins with env vars:

```bash
LLAMA_TAG=b9000 WHISPER_TAG=v1.8.5 ./Scripts/install_xcframeworks.sh
```

After it finishes, drag `Vendor/llama.xcframework` and
`Vendor/whisper.xcframework` into the `local_ai_test` Xcode target
(**General → Frameworks, Libraries, and Embedded Content → Embed &
Sign**).

## `build_llama_mtmd_xcframework.sh`

Builds a **custom llama.cpp XCFramework that includes `libmtmd`** (the
multimodal runtime used by Gemma 3 Vision / Qwen2-VL / Pixtral in GGUF
form). Official releases are built with `LLAMA_BUILD_TOOLS=OFF`, so
they don't contain mtmd — this script re-enables it by patching
`build-xcframework.sh` per upstream commit
[`4d32fd29`](https://github.com/ggml-org/llama.cpp/commit/4d32fd29c54417f7f2037efbcc38f77b3772d223).

### Host requirements

- macOS (Apple Silicon recommended)
- Xcode 15+ with active developer dir set:
  ```bash
  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
  ```
- CMake ≥ 3.28 (`brew install cmake`)
- Python 3 (ships with macOS)
- ~5 GB free disk space
- 15–45 minutes build time depending on platform set and CPU

### Usage

```bash
# Default: iOS device + iOS simulator + macOS, master branch of llama.cpp
./Scripts/build_llama_mtmd_xcframework.sh

# Pin a specific llama.cpp revision (tag or SHA)
LLAMA_REV=b8882 ./Scripts/build_llama_mtmd_xcframework.sh

# Trim to iOS device only (fastest)
PLATFORMS=ios-device ./Scripts/build_llama_mtmd_xcframework.sh

# Full matrix including visionOS and tvOS
PLATFORMS=ios-device,ios-sim,macos,visionos,visionos-sim,tvos-sim,tvos-device \
  ./Scripts/build_llama_mtmd_xcframework.sh
```

### What the script does

1. Clones `ggml-org/llama.cpp` into `build-tmp/llama.cpp` at `$LLAMA_REV`.
2. Applies four deterministic patches to `build-xcframework.sh`:
   - `LLAMA_BUILD_TOOLS=OFF` → `ON` so `tools/mtmd` gets compiled.
   - Copies `mtmd.h`, `mtmd-helper.h`, `mtmd-audio.h`, `mtmd-image.h` (and legacy `mtmd-ios.h` if present) into the framework headers. The guards are defensive — upstream deleted `mtmd-ios.h` from master, so the script skips missing files.
   - Regenerates the framework's `module.modulemap` so `llama.h`, `mtmd.h`, and `mtmd-helper.h` are listed as real `header` directives (visible to Swift via `import llama`). `ggml*.h` and `gguf.h` are emitted as `textual header` to avoid ODR conflicts if `whisper.xcframework` — which ships its own `ggml.h` — ever gets linked in the same target.
   - Appends only `libmtmd.a` (not `libllama-common.a`) to the static library list that is combined into the framework dylib. `llama-common` would drag in cpp-httplib / Boost dependencies that are unused on-device.
3. Optionally strips build blocks for platforms not in `$PLATFORMS`.
4. Runs the patched script.
5. Moves the resulting XCFramework to `Vendor/llama.xcframework`.

> **Keep the modulemap in sync when you edit by hand.** If you ever tweak `Vendor/llama.xcframework/*/llama.framework/Modules/module.modulemap` directly (e.g. to add a new header), make sure to mirror the change in this script's modulemap heredoc as well. Otherwise the next rebuild will silently overwrite your fix.

### Known limitations / debugging

- **If the patch fails to apply**: upstream restructured the script
  since this tool was written. Re-read the sentinel lines that Python
  uses in step 2 (e.g. `cp ggml/include/gguf.h` and `libggml-blas.a`)
  and adjust if needed.
- **If CMake can't find an SDK**: verify `xcode-select -p` points to a
  full Xcode install, not the Command Line Tools.
- **If macOS build fails while iOS succeeds**: run with
  `PLATFORMS=ios-device,ios-sim` — the iOS artifacts are enough to
  ship for iPhone.
- **mtmd C API reference**: the current supported entry points live in
  [`tools/mtmd/mtmd.h`](https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/mtmd.h)
  and [`tools/mtmd/mtmd-helper.h`](https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/mtmd-helper.h).
  Inside the cloned repo they are at `build-tmp/llama.cpp/tools/mtmd/mtmd.h` and `mtmd-helper.h`. The old iOS-specific wrapper (`mtmd-ios.h`) has been removed upstream; use the generic C API instead. The canonical usage example is [`tools/mtmd/mtmd-cli.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/mtmd-cli.cpp).

### After a successful build

1. Replace the framework reference in Xcode
   (General → Frameworks → remove the old `llama.xcframework`, drag the new one from `Vendor/`, Embed & Sign).
2. Nothing else to do — `BackendAvailability.isLinked(.llamaCppVision)` already follows `hasLlama`, and `LlamaCppVisionEngine` + `LlamaVisionContext` are wired through `libmtmd`.

Sanity-check the produced modulemap:

```bash
cat Vendor/llama.xcframework/ios-arm64/llama.framework/Modules/module.modulemap
# Should list:
#   header "llama.h"
#   header "mtmd.h"
#   header "mtmd-helper.h"
# and the ggml*.h lines as `textual header` (not plain `header`).
```

If `mtmd.h` / `mtmd-helper.h` don't show up, re-run with `NO_CLEAN=0` to rebuild from scratch; something in `tools/mtmd/` might have failed silently.

## ggml header sync between llama and whisper XCFrameworks

Both `llama.xcframework` (custom mtmd build) and `whisper.xcframework`
(official release) textually embed their own copies of `ggml*.h`. When the
two are built from different ggml revisions, importing both modules in one
target fails with *"'ggml_type' has different definitions in different
modules"*. Identical textual definitions are allowed — so after (re)fetching
either framework, sync the headers from llama into whisper:

```bash
for slice in ios-arm64 ios-arm64_x86_64-simulator; do
  src=Vendor/llama.xcframework/$slice/llama.framework/Headers
  dst=Vendor/whisper.xcframework/$slice/whisper.framework/Headers
  [ -d "$src" ] && [ -d "$dst" ] || continue
  cp "$src"/{ggml.h,ggml-alloc.h,ggml-backend.h,ggml-metal.h,ggml-cpu.h,ggml-blas.h,gguf.h} "$dst"/
done
```

Headers only affect compile-time declarations; the whisper dylib itself is
unchanged. The app calls only `whisper_*` C entry points, which do not take
ggml types, so the declaration swap is ABI-safe.
