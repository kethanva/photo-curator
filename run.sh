#!/usr/bin/env bash
# =============================================================================
#  run.sh — Photo Curator  |  Full Setup & Launcher
# =============================================================================
#
#  This script does EVERYTHING needed to run the application:
#    1. Finds a compatible Python (3.10+)
#    2. Creates / reuses the .venv virtual environment
#    3. Installs all dependencies from requirements.txt
#    4. Installs OpenAI CLIP (required, git-based)
#    5. Creates required directories
#    6. Checks for photos in data/input_photos/
#    7. Launches the 9-stage pipeline (main.py)
#
#  USAGE
#    ./run.sh                          # full pipeline with defaults
#    ./run.sh --input /path/to/photos  # custom input folder
#    ./run.sh --output ~/Desktop/Best  # custom output folder
#    ./run.sh --dry-run                # score only, no files copied
#    ./run.sh --from-stage 6           # resume from deduplication
#    ./run.sh --from-stage 9           # re-rank and re-select only
#    ./run.sh --config my.yaml         # use alternate config file
#    ./run.sh --check-only             # verify setup, don't run pipeline
#    ./run.sh --reinstall              # force reinstall all dependencies
#
#  WHERE TO ADD PHOTOS
#    data/input_photos/   ← put your photos here (subdirs are scanned too)
#
#  OUTPUT
#    data/output_photos/  ← curated JPEG photos written here
#    data/output_photos/output.json  ← curation report
#
# =============================================================================

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
  RESET='\033[0m'; BOLD='\033[1m'; DIM='\033[2m'
  GREEN='\033[92m'; YELLOW='\033[93m'; RED='\033[91m'; CYAN='\033[96m'
else
  RESET=''; BOLD=''; DIM=''; GREEN=''; YELLOW=''; RED=''; CYAN=''
fi

ok()     { echo -e "  ${GREEN}✔${RESET}  $*"; }
warn()   { echo -e "  ${YELLOW}⚠${RESET}  $*"; }
fail()   { echo -e "  ${RED}✘${RESET}  $*"; }
info()   { echo -e "  ${CYAN}→${RESET}  $*"; }
header() { echo -e "\n${BOLD}$*${RESET}"; }
dim()    { echo -e "  ${DIM}$*${RESET}"; }
die()    { echo -e "\n${RED}${BOLD}ERROR: $*${RESET}\n" >&2; exit 1; }

# ── Project root (directory containing this script) ───────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Record wall-clock start time
RUN_START=$(date +%s)
RUN_START_LABEL=$(date '+%Y-%m-%d %H:%M:%S')

VENV_DIR="$SCRIPT_DIR/.venv"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"
MAIN_PY="$SCRIPT_DIR/main.py"
CONFIG_FILE="config.yaml"

# Photo extensions (mirrors config.yaml)
PHOTO_EXTS=("jpg" "jpeg" "png" "heic" "heif" "tiff" "bmp" "webp")

# ── Argument parsing ──────────────────────────────────────────────────────────
CHECK_ONLY=false
REINSTALL=false
CLEAN=true   # always clean cache + output before running
PASSTHROUGH_ARGS=()   # forwarded to main.py
INPUT_DIR=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only)  CHECK_ONLY=true;              shift ;;
    --reinstall)   REINSTALL=true;               shift ;;
    --no-clean)    CLEAN=false;                  shift ;;
    --input)       INPUT_DIR="$2";  PASSTHROUGH_ARGS+=("$1" "$2"); shift 2 ;;
    --output)      OUTPUT_DIR="$2"; PASSTHROUGH_ARGS+=("$1" "$2"); shift 2 ;;
    --config)      CONFIG_FILE="$2"; PASSTHROUGH_ARGS+=("$1" "$2"); shift 2 ;;
    *)             PASSTHROUGH_ARGS+=("$1");      shift ;;
  esac
done

# ── Resolve input/output from config if not overridden ───────────────────────
# (used only for the photo-count check — main.py reads config itself)
if [[ -z "$INPUT_DIR" && -f "$SCRIPT_DIR/$CONFIG_FILE" ]]; then
  INPUT_DIR=$(grep -m1 'input:' "$SCRIPT_DIR/$CONFIG_FILE" \
    | sed 's/.*input:[[:space:]]*//' | tr -d '"' | xargs)
fi
INPUT_DIR="${INPUT_DIR:-data/input_photos}"

# Make relative paths absolute
[[ "$INPUT_DIR" != /* ]] && INPUT_DIR="$SCRIPT_DIR/$INPUT_DIR"

if [[ -z "$OUTPUT_DIR" && -f "$SCRIPT_DIR/$CONFIG_FILE" ]]; then
  OUTPUT_DIR=$(grep -m1 'output:' "$SCRIPT_DIR/$CONFIG_FILE" \
    | sed 's/.*output:[[:space:]]*//' | tr -d '"' | xargs)
fi
OUTPUT_DIR="${OUTPUT_DIR:-data/output_photos}"
[[ "$OUTPUT_DIR" != /* ]] && OUTPUT_DIR="$SCRIPT_DIR/$OUTPUT_DIR"

# =============================================================================
#  STEP 1 — Find Python 3.10+
# =============================================================================
header "① Finding Python 3.10+"

PYTHON=""
VENV_GOOD=false

# Search common locations for a compatible interpreter.
# Prefer 3.12 — PyTorch wheels are not yet published for 3.13.
CANDIDATES=(
  python3.12 python3.11 python3.10 python3.13
  /usr/local/bin/python3.12 /usr/local/bin/python3.11
  /usr/local/bin/python3.10 /usr/local/bin/python3.13
  /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11
  /opt/homebrew/bin/python3.10 /opt/homebrew/bin/python3.13
)

for candidate in "${CANDIDATES[@]}"; do
  if command -v "$candidate" &>/dev/null; then
    _ver=$("$candidate" -c "import sys; print(sys.version_info[:2])" 2>/dev/null || true)
    _maj=$(echo "$_ver" | tr -d '() ' | cut -d',' -f1)
    _min=$(echo "$_ver" | tr -d '() ' | cut -d',' -f2)
    if [[ -n "$_maj" && "$_maj" -ge 3 && "$_min" -ge 10 ]]; then
      PYTHON="$(command -v "$candidate")"
      break
    fi
  fi
done

if [[ -z "$PYTHON" ]]; then
  fail "No Python 3.10+ found on this system."
  echo
  echo "  Install Python 3.12+ from:"
  echo "  → https://www.python.org/downloads/"
  echo "  → or: brew install python@3.12"
  exit 1
fi

PY_VER=$("$PYTHON" --version 2>&1)
ok "$PY_VER at $PYTHON"

# =============================================================================
#  STEP 2 — Create / reuse virtual environment
# =============================================================================
header "② Virtual environment"

# If --reinstall, wipe it
if [[ "$REINSTALL" == true && -d "$VENV_DIR" ]]; then
  warn "--reinstall: removing existing venv"
  rm -rf "$VENV_DIR"
fi

# If venv exists but its Python is < 3.10, recreate it with the good interpreter
if [[ -d "$VENV_DIR" && -x "$VENV_DIR/bin/python3" ]]; then
  _venv_ver=$("$VENV_DIR/bin/python3" -c "import sys; print(sys.version_info[:2])" 2>/dev/null || echo "(0, 0)")
  _venv_maj=$(echo "$_venv_ver" | tr -d '() ' | cut -d',' -f1)
  _venv_min=$(echo "$_venv_ver" | tr -d '() ' | cut -d',' -f2)
  if [[ -n "$_venv_maj" && ( "$_venv_maj" -lt 3 || ( "$_venv_maj" -eq 3 && "$_venv_min" -lt 10 ) ) ]]; then
    warn "Existing .venv has Python ${_venv_maj}.${_venv_min} (< 3.10) — recreating with $PY_VER"
    rm -rf "$VENV_DIR"
  fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
  info "Creating .venv with $PY_VER …"
  "$PYTHON" -m venv "$VENV_DIR"
  ok "Created .venv"
else
  ok ".venv already exists — reusing"
fi

# Activate
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
PYTHON="$VENV_DIR/bin/python3"
PIP="$VENV_DIR/bin/pip"

ok "Activated: $($PYTHON --version)"

# =============================================================================
#  STEP 3 — Upgrade pip silently
# =============================================================================
header "③ Pip"
"$PIP" install --upgrade pip --quiet
ok "pip up to date ($($PIP --version | awk '{print $2}'))"

# =============================================================================
#  STEP 4 — Install requirements.txt
# =============================================================================
header "④ Installing dependencies  (requirements.txt)"

if [[ ! -f "$REQUIREMENTS" ]]; then
  die "requirements.txt not found at $REQUIREMENTS"
fi

# Check if requirements are already satisfied (skip reinstall on repeat runs)
NEED_INSTALL=false
if [[ "$REINSTALL" == true ]]; then
  NEED_INSTALL=true
else
  # Quick check: see if torch is importable (proxy for "already installed")
  if ! "$PYTHON" -c "import torch" &>/dev/null 2>&1; then
    NEED_INSTALL=true
  else
    ok "Dependencies appear already installed — skipping reinstall"
    dim "Run with --reinstall to force a fresh install"
  fi
fi

if [[ "$NEED_INSTALL" == true ]]; then
  info "Installing from requirements.txt …"
  "$PIP" install -r "$REQUIREMENTS" --quiet
  ok "All requirements installed"
fi

# =============================================================================
#  STEP 5 — Install OpenAI CLIP (required, not on PyPI)
# =============================================================================
header "⑤ OpenAI CLIP"

if ! "$PYTHON" -c "import clip" &>/dev/null 2>&1; then
  info "CLIP not found — installing from GitHub (this may take a minute) …"
  if "$PIP" install "git+https://github.com/openai/CLIP.git"; then
    ok "CLIP installed successfully"
  else
    fail "CLIP installation failed"
    info "Try manually: pip install git+https://github.com/openai/CLIP.git"
    exit 1
  fi
else
  ok "CLIP already installed"
fi

# =============================================================================
#  STEP 6 — Optional: mediapipe  (smile / eye sentiment stage)
# =============================================================================
header "⑥ Optional: mediapipe  (sentiment / smile detection)"

if "$PYTHON" -c "import mediapipe" &>/dev/null 2>&1; then
  ok "mediapipe installed — smile/eye scoring will run"
else
  warn "mediapipe not installed — sentiment stage will be skipped"
  info "Install with:  pip install mediapipe"
  dim "Re-run ./run.sh after installing to enable smile/eye scoring."
fi

# =============================================================================
#  STEP 7 — Create required directories
# =============================================================================
header "⑦ Project directories"

DIRS=(
  "data/input_photos"
  "data/output_photos"
  "cache"
  "models"
)

for d in "${DIRS[@]}"; do
  if [[ -d "$SCRIPT_DIR/$d" ]]; then
    ok "$d/  (exists)"
  else
    mkdir -p "$SCRIPT_DIR/$d"
    ok "$d/  (created)"
  fi
done

# =============================================================================
#  STEP 8 — Check for photos
# =============================================================================
header "⑧ Input photos"
info "Scanning: $INPUT_DIR"

if [[ ! -d "$INPUT_DIR" ]]; then
  warn "Input directory does not exist (will be created): $INPUT_DIR"
  mkdir -p "$INPUT_DIR"
fi

# Count photos with supported extensions
PHOTO_COUNT=0
for ext in "${PHOTO_EXTS[@]}"; do
  COUNT=$(find "$INPUT_DIR" -type f \( -iname "*.${ext}" \) 2>/dev/null | wc -l | xargs)
  PHOTO_COUNT=$((PHOTO_COUNT + COUNT))
done

if [[ "$PHOTO_COUNT" -eq 0 ]]; then
  echo
  warn "No photos found in $INPUT_DIR"
  echo
  echo -e "  ${BOLD}Where to add your photos:${RESET}"
  echo -e "  Copy or move your photos into:"
  echo -e "  ${CYAN}$INPUT_DIR${RESET}"
  echo
  echo "  Supported formats: jpg, jpeg, png, heic, heif, tiff, bmp, webp"
  echo
  echo "  Examples:"
  echo -e "  ${DIM}cp -r ~/Pictures/  \"$INPUT_DIR/\"${RESET}"
  echo -e "  ${DIM}cp -r /Volumes/iPhone/DCIM/  \"$INPUT_DIR/\"${RESET}"
  echo
  echo "  Subdirectories are scanned recursively — you can organise"
  echo "  your photos into folders (e.g. 2024/, holidays/, etc.)"
  echo

  if [[ "$CHECK_ONLY" == false ]]; then
    warn "No photos to process. Add photos and re-run ./run.sh"
    exit 0
  fi
else
  ok "$PHOTO_COUNT photos found (subdirs scanned recursively)"
  dim "Supported: jpg, jpeg, png, heic, heif, tiff, bmp, webp"
fi

# =============================================================================
#  STEP 9 — Disk space check
# =============================================================================
header "⑨ Disk space"

FREE_KB=$(df -k "$SCRIPT_DIR" | awk 'NR==2 {print $4}')
FREE_GB=$(echo "scale=1; $FREE_KB / 1048576" | bc 2>/dev/null || echo "?")
if [[ "$FREE_GB" != "?" && $(echo "$FREE_GB < 2" | bc 2>/dev/null) -eq 1 ]]; then
  warn "Low disk space: ${FREE_GB} GB free — at least 2 GB recommended"
else
  ok "${FREE_GB} GB free on disk"
fi

# =============================================================================
#  STEP 10 — Detect ML accelerator
# =============================================================================
header "⑩ ML Accelerator"

ACCEL=$("$PYTHON" -c "
import torch
if torch.backends.mps.is_available():
    print('Apple MPS (Metal) — Apple Silicon GPU acceleration')
elif torch.cuda.is_available():
    print('NVIDIA CUDA —', torch.cuda.get_device_name(0))
else:
    print('CPU only — no GPU (will be slower)')
" 2>/dev/null || echo "Could not detect (torch import failed)")
ok "$ACCEL"

# =============================================================================
#  STEP 11 — Quick reference
# =============================================================================
header "━━ Quick Reference ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "  ${CYAN}./run.sh${RESET}                              Full pipeline (uses config.yaml)"
echo -e "  ${CYAN}./run.sh --input /path/to/photos${RESET}      Custom input folder"
echo -e "  ${CYAN}./run.sh --output ~/Desktop/Best${RESET}      Custom output folder"
echo -e "  ${CYAN}./run.sh --dry-run${RESET}                    Score photos, don't copy files"
echo -e "  ${CYAN}./run.sh --from-stage 6${RESET}               Resume from deduplication"
echo -e "  ${CYAN}./run.sh --from-stage 9${RESET}               Re-rank and re-select only"
echo -e "  ${CYAN}./run.sh --no-clean${RESET}                   Skip cache wipe (keep existing cache)"
echo -e "  ${CYAN}./run.sh --config my.yaml${RESET}             Use alternate config file"
echo -e "  ${CYAN}./run.sh --check-only${RESET}                 Check environment, don't run"
echo -e "  ${CYAN}./run.sh --reinstall${RESET}                  Force reinstall all dependencies"
echo

# =============================================================================
#  STEP 12 — Summary & gate
# =============================================================================
header "━━ Pre-flight Summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

READY=true

# Python OK (already confirmed above)
ok "Python 3.10+              ($PY_VER)"

# Packages installed
if "$PYTHON" -c "import torch, PIL, chromadb, yaml, tqdm, sklearn, imagehash, piexif, facenet_pytorch, pillow_heif" &>/dev/null 2>&1; then
  ok "Required packages         all installed"
else
  fail "Required packages         some missing — run: pip install -r requirements.txt"
  READY=false
fi

# CLIP
if "$PYTHON" -c "import clip" &>/dev/null 2>&1; then
  ok "OpenAI CLIP               installed"
else
  fail "OpenAI CLIP               missing"
  info "pip install git+https://github.com/openai/CLIP.git"
  READY=false
fi

# Photos
if [[ "$PHOTO_COUNT" -gt 0 ]]; then
  ok "Photos in input           $PHOTO_COUNT photos"
else
  warn "Photos in input           0 photos found in $INPUT_DIR"
  READY=false
fi

echo

if [[ "$READY" != true ]]; then
  fail "Fix the issues above, then re-run ./run.sh"
  exit 1
fi

if [[ "$CHECK_ONLY" == true ]]; then
  ok "All checks passed — environment is ready"
  echo
  info "Run ./run.sh to start the pipeline"
  exit 0
fi

echo -e "  ${GREEN}${BOLD}All checks passed — starting pipeline!${RESET}"
echo

# =============================================================================
#  STEP 12b — Optional cleanup before launch
# =============================================================================
CLEANUP_SH="$SCRIPT_DIR/cleanup.sh"

if [[ "$CLEAN" == true ]]; then
  header "━━ Pre-run Cleanup ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  if [[ ! -x "$CLEANUP_SH" ]]; then
    warn "cleanup.sh not found or not executable — skipping cleanup"
  else
    info "Clearing cache + output before fresh run…"
    dim "  cache/photo_db.sqlite        (SQLite metadata DB)"
    dim "  cache/vector_store/          (ChromaDB HNSW index)"
    dim "  data/output_photos/          (previous curated output)"
    echo
    bash "$CLEANUP_SH" --cache --output
    ok "Cleanup done — starting fresh run"
  fi
  echo
fi

# =============================================================================
#  STEP 13 — Launch the pipeline
# =============================================================================
header "━━ Launching Photo Curator Pipeline ━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "  Input  → ${CYAN}$INPUT_DIR${RESET}"
echo -e "  Output → ${CYAN}$OUTPUT_DIR${RESET}"
echo

CMD=("$PYTHON" "$MAIN_PY" ${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"})
info "Running: ${CMD[*]}"
echo
echo -e "  ${DIM}Output photos will be written to: $OUTPUT_DIR${RESET}"
echo -e "  ${DIM}Curation report: $OUTPUT_DIR/output.json${RESET}"
echo
echo -e "  ${BOLD}Started at:${RESET}  $RUN_START_LABEL"
echo

# Run the pipeline (not exec so we can print timing after it exits)
"${CMD[@]}"
EXIT_CODE=$?

# ── Print timing summary ──────────────────────────────────────────────────────
RUN_END=$(date +%s)
RUN_END_LABEL=$(date '+%Y-%m-%d %H:%M:%S')
ELAPSED=$(( RUN_END - RUN_START ))
ELAPSED_MIN=$(( ELAPSED / 60 ))
ELAPSED_SEC=$(( ELAPSED % 60 ))

echo
header "━━ Run Complete ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ok "Started:   $RUN_START_LABEL"
ok "Finished:  $RUN_END_LABEL"
ok "Elapsed:   ${ELAPSED_MIN}m ${ELAPSED_SEC}s  (${ELAPSED}s total)"
echo -e "  ${DIM}Output: $OUTPUT_DIR${RESET}"
echo

exit $EXIT_CODE
