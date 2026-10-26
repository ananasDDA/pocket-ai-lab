#!/usr/bin/env bash
#
# daily_use_night.sh - run the daily-use device suite over a list of models,
# one xcodebuild invocation per model, on the iPhone, unattended.
#
#   Scripts/daily_use_night.sh [options] [model-id ...]
#
# Each model runs in its own test process (a crash or a jetsam kill takes out
# one model, not the night). Before every run the script takes the shared
# phone lock (mkdir $PAL_PHONE_LOCK, default /tmp/pal-phone.lock, waiting 30 s between
# tries) and releases it afterwards, so nothing else drives the phone while a
# model runs. Between models it pauses (default 1 min); the real wait is in
# the suite, which before the load and before every scenario waits for the
# phone to reach `nominal` and settle for 60 s, for Low Power Mode off and for
# power (charging, or 40% battery or more), up to 20 min per wait (see the
# conditions gate in DailyUseDeviceTests.swift). When the power is still bad
# at the cap the suite prints DAILY_NIGHT_STOP and this script ends the night
# instead of measuring garbage for every model left.
#
# Models run heavy/light interleaved by catalog ramRequiredGB (heaviest,
# lightest, second heaviest, ...), so a light model gives the phone a break;
# --no-interleave keeps the given order.
#
# Output (under --out):
#   logs/<model>[-nosys][-<sampling>].log  full xcodebuild log per run
#   extract/<model>/<scenario>.json  per-scenario transcripts (daily_use_extract.py)
#   tables/<model>[-nosys][-<sampling>].txt  the per-model table
#   combined.txt                     one row per run, printed at the end too
#   night.log                        what this script did, with timestamps
#
# Options:
#   --out DIR          output directory (default build-tmp/daily-night/<timestamp>)
#   --scenarios LIST   comma-separated subset, passed as DAILY_SCENARIOS
#   --pause SEC        pause between runs (default 60)
#   --timeout MIN      kill a run after this many minutes (default 240; the
#                      conditions gate can add up to 20 min per wait)
#   --no-interleave    run the models in the order given
#   --ab               also run the A/B models again with no system prompt
#                      (DAILY_NO_SYSTEM_PROMPT=1, the chat before 1.2), right
#                      after their normal run; the model is kept on the phone
#                      between the two runs and deleted after the second
#   --ab-only          only the no-system-prompt runs of the A/B models
#   --ab-models "A B"  models for the A/B (default: gemma-4-E2B and Qwen3.5-2B)
#   --sampling MODE    catalog (default), global or authors, passed as
#                      DAILY_SAMPLING to every run: `global` sends the global
#                      chat parameters, `authors` the model authors' values
#                      from the test bundle's DailyUseAuthorsSampling.json.
#                      Runs not in `catalog` mode get a -<mode> suffix and
#                      an "[sampling <mode>]" label in the tables.
#   --no-build         skip build-for-testing (reuse the last build in
#                      --derived-data)
#   --derived-data DIR (default build-tmp/daily-night-dd)
#   --dry-run          print the commands, run nothing, take no lock
#
# Model ids given on the command line replace the default list.
#
# Passed through to the suite when set in this script's environment:
# DAILY_COOLDOWN_TARGET (nominal|fair|off), DAILY_COOLDOWN_MAX_SEC (1200),
# DAILY_COOLDOWN_SETTLE_SEC (60), DAILY_STOP_ON_BAD_POWER (1; 0 = measure and
# flag), DAILY_DOWNLOAD_STALL_SEC (300), DAILY_NO_THERMAL_GATE.
#
# Before starting: phone unlocked, Auto-Lock set to Never, cable in, Wi-Fi on
# (a locked phone sleeps Wi-Fi and blocks xcodebuild). Apple Intelligence must
# be turned on in Settings for the apple-intelligence run; if it is not, that
# run ends with "load failed" and the night goes on.

set -u

UDID="00008150-001845263687801C"
LOCK="${PAL_PHONE_LOCK:-/tmp/pal-phone.lock}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

DEFAULT_MODELS=(
  "unsloth/gemma-4-E2B-it-GGUF"
  "LiquidAI/LFM2-2.6B-GGUF"
  "LiquidAI/LFM2-1.2B-GGUF"
  "unsloth/Qwen3.5-2B-GGUF"
  "unsloth/Qwen3.5-4B-GGUF"
  "unsloth/gemma-3-4b-it-GGUF"
  "bartowski/Llama-3.2-3B-Instruct-GGUF"
  "apple-intelligence"
)
AB_MODELS=("unsloth/gemma-4-E2B-it-GGUF" "unsloth/Qwen3.5-2B-GGUF")

OUT=""
SCENARIOS=""
PAUSE=60
TIMEOUT_MIN=240
INTERLEAVE=1
AB=0
AB_ONLY=0
BUILD=1
DERIVED="$ROOT/build-tmp/daily-night-dd"
DRY=0
SAMPLING="catalog"
MODELS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --scenarios) SCENARIOS="$2"; shift 2 ;;
    --pause) PAUSE="$2"; shift 2 ;;
    --timeout) TIMEOUT_MIN="$2"; shift 2 ;;
    --ab) AB=1; shift ;;
    --ab-only) AB=1; AB_ONLY=1; shift ;;
    --ab-models) read -r -a AB_MODELS <<< "$2"; shift 2 ;;
    --no-build) BUILD=0; shift ;;
    --derived-data) DERIVED="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --no-interleave) INTERLEAVE=0; shift ;;
    --sampling) SAMPLING="$2"; shift 2 ;;
    -h|--help) sed -n '2,/^set -/p' "$0" | sed '$d'; exit 0 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) MODELS+=("$1"); shift ;;
  esac
done
case "$SAMPLING" in
  catalog|global|authors) ;;
  *) echo "--sampling must be catalog, global or authors, not '$SAMPLING'" >&2; exit 2 ;;
esac
[ ${#MODELS[@]} -eq 0 ] && MODELS=("${DEFAULT_MODELS[@]}")
if [ "$AB_ONLY" = 1 ]; then MODELS=("${AB_MODELS[@]}"); fi

# Heaviest, lightest, second heaviest, ... by catalog ramRequiredGB (ids the
# catalog does not know count as light). Same rule as
# RunOrdering.interleaveHeavyLight in the app.
interleave_models() {
  python3 - "$ROOT/local_ai_test/Resources/catalog.json" "$@" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    catalog = json.load(f)
ram = {m["id"]: float(m.get("ramRequiredGB") or 0) for m in catalog.get("models", [])}
ids = sys.argv[2:]
order = sorted(range(len(ids)), key=lambda i: (-ram.get(ids[i], 0.0), i))
out, lo, hi, heavy = [], 0, len(order) - 1, True
while lo <= hi:
    if heavy:
        out.append(ids[order[lo]]); lo += 1
    else:
        out.append(ids[order[hi]]); hi -= 1
    heavy = not heavy
print("\n".join(out))
PY
}
if [ "$INTERLEAVE" = 1 ] && [ ${#MODELS[@]} -gt 2 ]; then
  ORDERED=()
  while IFS= read -r line; do
    [ -n "$line" ] && ORDERED+=("$line")
  done < <(interleave_models "${MODELS[@]}")
  if [ ${#ORDERED[@]} -eq ${#MODELS[@]} ]; then
    MODELS=("${ORDERED[@]}")
  else
    echo "could not interleave the models (catalog unreadable?) - keeping the given order" >&2
  fi
fi

# Gate settings passed through to the suite.
FORWARD_VARS=(DAILY_COOLDOWN_TARGET DAILY_COOLDOWN_MAX_SEC DAILY_COOLDOWN_SETTLE_SEC
  DAILY_STOP_ON_BAD_POWER DAILY_DOWNLOAD_STALL_SEC DAILY_NO_THERMAL_GATE)

[ -z "$OUT" ] && OUT="$ROOT/build-tmp/daily-night/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT/logs" "$OUT/tables" "$OUT/extract"
NIGHT_LOG="$OUT/night.log"

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$NIGHT_LOG"; }
safe() { echo "$1" | tr '/' '_'; }
# Log/table name of one run: model, -nosys for the no-system-prompt twin,
# -<mode> outside the default sampling mode.
run_name() {
  local name
  name="$(safe "$1")"
  [ "$2" = "nosys" ] && name="$name-nosys"
  [ "$SAMPLING" != "catalog" ] && name="$name-$SAMPLING"
  echo "$name"
}
in_ab() { local m; for m in "${AB_MODELS[@]}"; do [ "$m" = "$1" ] && return 0; done; return 1; }

HOLDING=0
CHILD=""
acquire_lock() {
  [ "$DRY" = 1 ] && return
  local waited=0
  until mkdir "$LOCK" 2>/dev/null; do
    [ $((waited % 300)) -eq 0 ] && say "phone lock busy ($LOCK), waiting"
    sleep 30
    waited=$((waited + 30))
  done
  HOLDING=1
}
release_lock() {
  if [ "$HOLDING" = 1 ]; then
    rmdir "$LOCK" 2>/dev/null
    HOLDING=0
  fi
}
cleanup() {
  [ -n "$CHILD" ] && kill "$CHILD" 2>/dev/null
  release_lock
}
trap cleanup EXIT
trap 'say "interrupted"; exit 130' INT TERM

# Runs "$@" with stdout+stderr to $LOG, killing it after TIMEOUT_MIN.
run_logged() {
  local log="$1"; shift
  if [ "$DRY" = 1 ]; then
    echo "  $*"
    return 0
  fi
  "$@" > "$log" 2>&1 &
  CHILD=$!
  local limit=$((TIMEOUT_MIN * 60)) elapsed=0
  while kill -0 "$CHILD" 2>/dev/null; do
    sleep 10
    elapsed=$((elapsed + 10))
    if [ "$elapsed" -ge "$limit" ]; then
      say "  timeout after $TIMEOUT_MIN min - killing xcodebuild"
      kill "$CHILD" 2>/dev/null
      sleep 5
      kill -9 "$CHILD" 2>/dev/null
      echo "DAILY_NIGHT_TIMEOUT after $TIMEOUT_MIN min" >> "$log"
      break
    fi
  done
  wait "$CHILD" 2>/dev/null
  local status=$?
  CHILD=""
  return $status
}

say "daily-use night: ${#MODELS[@]} model(s), out $OUT"
# Read in the morning: first, whether the phone was set up for the night.
say "Night checklist (the gates wait for these, but cannot fix them):"
say "  - Auto-Lock: Never (a locked phone sleeps Wi-Fi and blocks xcodebuild)"
say "  - cable to the Mac, charging"
say "  - Low Power Mode off"
say "  - no case"
say "  - phone on a cool hard surface, not on the laptop"
say "  - screen brightness at minimum"
say "models$([ "$INTERLEAVE" = 1 ] && echo ' (heavy/light interleaved)'): ${MODELS[*]}"
gate_settings=""
for var in "${FORWARD_VARS[@]}"; do
  [ -n "${!var:-}" ] && gate_settings="${gate_settings:+$gate_settings }$var=${!var}"
done
say "conditions gate: ${gate_settings:-suite defaults (nominal + 60 s settle, cap 1200 s, stop on bad power)}"
[ "$AB" = 1 ] && say "A/B without system prompt: ${AB_MODELS[*]}$([ "$AB_ONLY" = 1 ] && echo ' (only)')"
[ -n "$SCENARIOS" ] && say "scenarios: $SCENARIOS"
say "sampling: $SAMPLING"
say "repo $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null) ($(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null))"

# 1. One signed build for the device, reused by every run.
if [ "$BUILD" = 1 ]; then
  say "build-for-testing (derived data $DERIVED)"
  acquire_lock
  run_logged "$OUT/logs/build.log" \
    xcodebuild build-for-testing \
      -project "$ROOT/local_ai_test.xcodeproj" -scheme local_ai_test \
      -destination "platform=iOS,id=$UDID" \
      -derivedDataPath "$DERIVED"
  status=$?
  release_lock
  if [ "$DRY" = 0 ] && [ $status -ne 0 ]; then
    say "BUILD FAILED (exit $status) - see $OUT/logs/build.log"
    grep -E "error:" "$OUT/logs/build.log" | sort -u | head -20 | tee -a "$NIGHT_LOG"
    exit 1
  fi
fi

# 2. One run per model (plus its no-system-prompt twin in A/B mode).
RUNS=()
for model in "${MODELS[@]}"; do
  variants=()
  if [ "$AB_ONLY" = 1 ]; then
    variants=("nosys")
  else
    variants=("main")
    if [ "$AB" = 1 ] && in_ab "$model"; then variants+=("nosys"); fi
  fi
  for variant in "${variants[@]}"; do
    RUNS+=("$model|$variant|${#variants[@]}")
  done
done

index=0
for entry in "${RUNS[@]}"; do
  IFS='|' read -r model variant count <<< "$entry"
  index=$((index + 1))
  name="$(run_name "$model" "$variant")"
  env_vars=(TEST_RUNNER_DAILY_ENABLED=1 "TEST_RUNNER_DAILY_MODEL_ID=$model"
            "TEST_RUNNER_DAILY_SAMPLING=$SAMPLING")
  [ -n "$SCENARIOS" ] && env_vars+=("TEST_RUNNER_DAILY_SCENARIOS=$SCENARIOS")
  for var in "${FORWARD_VARS[@]}"; do
    [ -n "${!var:-}" ] && env_vars+=("TEST_RUNNER_$var=${!var}")
  done
  if [ "$variant" = "nosys" ]; then
    env_vars+=(TEST_RUNNER_DAILY_NO_SYSTEM_PROMPT=1)
  elif [ "$count" = 2 ]; then
    # Keep the download for the no-system-prompt run that follows; that
    # run deletes it (the harness's ownership marker survives).
    env_vars+=(TEST_RUNNER_DAILY_KEEP_MODEL=1)
  fi
  log="$OUT/logs/$name.log"

  if [ $index -gt 1 ] && [ "$DRY" = 0 ]; then
    say "pause ${PAUSE}s before the next run"
    sleep "$PAUSE"
  fi

  say "[$index/${#RUNS[@]}] $model ($variant) -> $log"
  acquire_lock
  started=$(date +%s)
  # The TEST_RUNNER_ variables must be in the environment of xcodebuild
  # (env ... xcodebuild), not after it as build settings.
  run_logged "$log" env "${env_vars[@]}" \
    xcodebuild test-without-building \
      -project "$ROOT/local_ai_test.xcodeproj" -scheme local_ai_test \
      -destination "platform=iOS,id=$UDID" \
      -derivedDataPath "$DERIVED" \
      -only-testing:local_ai_testTests/DailyUseDeviceTests
  status=$?
  release_lock
  [ "$DRY" = 1 ] && continue
  minutes=$(( ($(date +%s) - started) / 60 ))

  verdict="exit $status, ${minutes} min"
  if ! grep -q "DAILY_SUMMARY_JSON" "$log"; then
    verdict="$verdict, NO SUMMARY (crash, timeout or the test did not run)"
  fi
  if grep -q "Executed 0 tests" "$log" && ! grep -qE "Test run with [1-9][0-9]* test" "$log"; then
    verdict="$verdict, 0 TESTS RAN"
  fi
  say "  $verdict"
  grep -E "\[daily-use\] .*(ERROR|load failed|skipped|failed in|partial in)" "$log" | tail -8 | tee -a "$NIGHT_LOG"

  python3 "$ROOT/Scripts/daily_use_extract.py" "$log" --out "$OUT/extract" > "$OUT/tables/$name.txt" 2>&1 \
    || say "  extract found nothing in $log"

  # The suite gave up on the power (Low Power Mode, or a draining battery
  # that did not recover within the cap): every later run would measure the
  # power state, not the model.
  stop_line="$(grep -m1 "DAILY_NIGHT_STOP" "$log")"
  # A locked phone never starts the test runner, so the suite cannot say
  # anything; every later run would fail the same way after a long timeout.
  if [ -z "$stop_line" ] && grep -q "Unlock iPhone" "$log"; then
    stop_line="DAILY_NIGHT_STOP the phone is locked (set Auto-Lock to Never)"
  fi
  if [ -n "$stop_line" ]; then
    say "NIGHT STOPPED by the conditions gate: ${stop_line#*DAILY_NIGHT_STOP }"
    left=()
    for later in "${RUNS[@]:$index}"; do left+=("${later%%|*}"); done
    [ ${#left[@]} -gt 0 ] && say "  not run: ${left[*]}"
    say "  check the phone: Low Power Mode off, cable in and charging"
    break
  fi
done

[ "$DRY" = 1 ] && exit 0

# 3. One combined table.
say "combined table"
logs=()
for entry in "${RUNS[@]}"; do
  IFS='|' read -r model variant _ <<< "$entry"
  name="$(run_name "$model" "$variant")"
  [ -f "$OUT/logs/$name.log" ] && logs+=("$OUT/logs/$name.log")
done
if [ ${#logs[@]} -gt 0 ]; then
  python3 "$ROOT/Scripts/daily_use_extract.py" "${logs[@]}" --out "$OUT/extract-combined" --combined \
    | tee "$OUT/combined.txt"
fi
if [ "$AB" = 1 ]; then
  echo
  echo "A/B: rows marked [no system prompt] are the DAILY_NO_SYSTEM_PROMPT=1 runs; the transcripts"
  echo "are in extract/<model>/ and extract/<model> [no system prompt]/ (spaces become _)."
fi
say "done: $OUT"
