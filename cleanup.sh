#!/usr/bin/env bash
# =============================================================================
#  cleanup.sh — Photo Curator  |  Cleanup Utility
# =============================================================================
#
#  WHERE CACHE IS STORED
#  ─────────────────────
#  cache/photo_db.sqlite        SQLite metadata DB (scores, flags, EXIF, etc.)
#  cache/photo_db.sqlite-wal    WAL journal (auto-cleared on next DB open)
#  cache/photo_db.sqlite-shm    Shared memory file (safe to delete when idle)
#  cache/vector_store/          ChromaDB HNSW index (CLIP + face embeddings)
#    └── chroma.sqlite3         ChromaDB internal metadata
#    └── <uuid>/                HNSW index files per collection
#
#  WHAT IS SAFE TO DELETE
#  ──────────────────────
#  cache/               → forces full reprocess on next run (slow, ~60 min)
#  data/output_photos/  → deletes curated output (originals untouched)
#  models/              → re-downloaded automatically on next run
#  .venv/               → re-created by run.sh automatically
#  **/__pycache__/      → Python bytecode cache (always safe)
#
#  USAGE
#    ./cleanup.sh --help          show this help
#    ./cleanup.sh --cache         delete SQLite DB + vector store (force reprocess)
#    ./cleanup.sh --output        delete curated output photos + report
#    ./cleanup.sh --models        delete downloaded ML model weights
#    ./cleanup.sh --venv          delete Python virtual environment
#    ./cleanup.sh --pycache       delete __pycache__ and .pyc files
#    ./cleanup.sh --all           delete everything above
#    ./cleanup.sh --dry-run       show what would be deleted, don't delete
#
#  Multiple flags can be combined:
#    ./cleanup.sh --cache --output
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
info()   { echo -e "  ${CYAN}→${RESET}  $*"; }
header() { echo -e "\n${BOLD}$*${RESET}"; }
dim()    { echo -e "  ${DIM}$*${RESET}"; }
skip()   { echo -e "  ${DIM}–  $* (skipped)${RESET}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Argument parsing ──────────────────────────────────────────────────────────
DO_CACHE=false
DO_OUTPUT=false
DO_MODELS=false
DO_VENV=false
DO_PYCACHE=false
DO_ALL=false
DRY_RUN=false
SHOW_HELP=false

if [[ $# -eq 0 ]]; then SHOW_HELP=true; fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cache)    DO_CACHE=true;   shift ;;
    --output)   DO_OUTPUT=true;  shift ;;
    --models)   DO_MODELS=true;  shift ;;
    --venv)     DO_VENV=true;    shift ;;
    --pycache)  DO_PYCACHE=true; shift ;;
    --all)      DO_ALL=true;     shift ;;
    --dry-run)  DRY_RUN=true;    shift ;;
    --help|-h)  SHOW_HELP=true;  shift ;;
    *) echo -e "${RED}Unknown option: $1${RESET}" >&2; SHOW_HELP=true; BAD_OPT=true; shift ;;
  esac
done

if [[ "$DO_ALL" == true ]]; then
  DO_CACHE=true; DO_OUTPUT=true; DO_MODELS=true
  DO_VENV=true; DO_PYCACHE=true
fi

# ── Help ─────────────────────────────────────────────────────────────────────
if [[ "$SHOW_HELP" == true ]]; then
  echo
  echo -e "${BOLD}Photo Curator — Cleanup Utility${RESET}"
  echo
  echo -e "${BOLD}WHERE CACHE IS STORED:${RESET}"
  echo -e "  ${CYAN}cache/photo_db.sqlite${RESET}     SQLite metadata DB  (scores, EXIF, flags per photo)"
  echo -e "  ${CYAN}cache/vector_store/${RESET}        ChromaDB HNSW index  (CLIP + face embeddings)"
  echo
  echo -e "${BOLD}USAGE:${RESET}"
  printf "  ${CYAN}%-35s${RESET} %s\n" "./cleanup.sh --cache"   "Delete SQLite DB + vector store  ← forces full reprocess"
  printf "  ${CYAN}%-35s${RESET} %s\n" "./cleanup.sh --output"  "Delete curated output photos + report"
  printf "  ${CYAN}%-35s${RESET} %s\n" "./cleanup.sh --models"  "Delete downloaded ML model weights"
  printf "  ${CYAN}%-35s${RESET} %s\n" "./cleanup.sh --venv"    "Delete Python virtual environment"
  printf "  ${CYAN}%-35s${RESET} %s\n" "./cleanup.sh --pycache" "Delete __pycache__ / .pyc bytecode"
  printf "  ${CYAN}%-35s${RESET} %s\n" "./cleanup.sh --all"     "Delete everything above"
  printf "  ${CYAN}%-35s${RESET} %s\n" "./cleanup.sh --dry-run" "Preview what would be deleted"
  echo
  echo -e "  Flags can be combined:  ${DIM}./cleanup.sh --cache --output --dry-run${RESET}"
  echo
  [[ "${BAD_OPT:-false}" == true ]] && exit 1 || exit 0
fi

# ── Helper: remove path with size reporting ───────────────────────────────────
remove() {
  local target="$1"
  local label="$2"

  if [[ ! -e "$target" && ! -L "$target" ]]; then
    skip "$label  (not found)"
    return
  fi

  local size
  size=$(du -sh "$target" 2>/dev/null | awk '{print $1}' || echo "?")

  if [[ "$DRY_RUN" == true ]]; then
    warn "[dry-run] Would delete  $label  ($size)"
  else
    if rm -rf "$target" 2>/dev/null; then
      ok "Deleted  $label  ($size freed)"
    else
      warn "Failed to delete $label (permission denied?)"
    fi
  fi
}

# ── Helper: recreate empty placeholder dirs ───────────────────────────────────
keep_dir() {
  local dir="$1"
  if [[ "$DRY_RUN" == false ]]; then
    mkdir -p "$dir"
    # restore .gitkeep if the dir is tracked
    [[ -d "$dir" ]] && touch "$dir/.gitkeep" 2>/dev/null || true
  fi
}

# ── Header ────────────────────────────────────────────────────────────────────
BAR="═══════════════════════════════════════════════════════════"
echo
echo -e "${BOLD}${CYAN}${BAR}${RESET}"
echo -e "${BOLD}${CYAN}  Photo Curator — Cleanup$([ "$DRY_RUN" == true ] && echo "  [DRY RUN]")${RESET}"
echo -e "${BOLD}${CYAN}${BAR}${RESET}"
[[ "$DRY_RUN" == true ]] && warn "Dry-run mode — nothing will actually be deleted"

# ── 1. Cache ──────────────────────────────────────────────────────────────────
header "① Processing cache"
echo -e "  ${DIM}Location: cache/${RESET}"
echo -e "  ${DIM}Contains: SQLite metadata DB + ChromaDB vector store${RESET}"
echo

if [[ "$DO_CACHE" == true ]]; then
  warn "Deleting cache forces a full reprocess on next run (~60 min for 5k photos)"
  remove "$SCRIPT_DIR/cache/photo_db.sqlite"      "cache/photo_db.sqlite      (metadata DB)"
  remove "$SCRIPT_DIR/cache/photo_db.sqlite-wal"  "cache/photo_db.sqlite-wal  (WAL journal)"
  remove "$SCRIPT_DIR/cache/photo_db.sqlite-shm"  "cache/photo_db.sqlite-shm  (shared memory)"
  remove "$SCRIPT_DIR/cache/vector_store"         "cache/vector_store/        (CLIP + face HNSW index)"
  [[ "$DRY_RUN" == false ]] && keep_dir "$SCRIPT_DIR/cache"
  echo
  dim "Next run will rebuild everything from scratch."
else
  skip "cache (use --cache to clear)"
fi

# ── 2. Output photos ──────────────────────────────────────────────────────────
header "② Curated output"
echo -e "  ${DIM}Location: data/output_photos/${RESET}"
echo -e "  ${DIM}Contains: curated JPEG photos + output.json report${RESET}"
echo

if [[ "$DO_OUTPUT" == true ]]; then
  # Count photos before removing
  OUT_DIR="$SCRIPT_DIR/data/output_photos"
  PHOTO_COUNT=0
  if [[ -d "$OUT_DIR" ]]; then
    PHOTO_COUNT=$(find "$OUT_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.json" \) 2>/dev/null | wc -l | xargs || echo 0)
  fi
  warn "This will delete $PHOTO_COUNT output file(s) — original photos in data/input_photos/ are NOT touched"
  remove "$OUT_DIR" "data/output_photos/  (curated output)"
  [[ "$DRY_RUN" == false ]] && keep_dir "$SCRIPT_DIR/data/output_photos"
else
  skip "output (use --output to clear)"
fi

# ── 3. Model weights ──────────────────────────────────────────────────────────
header "③ ML model weights"
echo -e "  ${DIM}Location: models/${RESET}"
echo -e "  ${DIM}Contains: aesthetic_predictor.pth (only if LAION mode enabled)${RESET}"
echo

if [[ "$DO_MODELS" == true ]]; then
  remove "$SCRIPT_DIR/models" "models/  (downloaded ML weights)"
  [[ "$DRY_RUN" == false ]] && keep_dir "$SCRIPT_DIR/models"
else
  skip "models (use --models to clear)"
fi

# ── 4. Python virtual environment ─────────────────────────────────────────────
header "④ Python virtual environment"
echo -e "  ${DIM}Location: .venv/${RESET}"
echo -e "  ${DIM}Contains: all pip packages (~2 GB: torch, CLIP, chromadb, etc.)${RESET}"
echo

if [[ "$DO_VENV" == true ]]; then
  warn "Deleting .venv means all packages will be reinstalled on next ./run.sh"
  remove "$SCRIPT_DIR/.venv" ".venv/  (Python virtual environment)"
else
  skip ".venv (use --venv to clear)"
fi

# ── 5. Python bytecode cache ──────────────────────────────────────────────────
header "⑤ Python bytecode cache"
echo -e "  ${DIM}Location: **/__pycache__/, **/*.pyc${RESET}"
echo -e "  ${DIM}Contains: compiled bytecode — always safe to delete${RESET}"
echo

if [[ "$DO_PYCACHE" == true ]]; then
  if [[ "$DRY_RUN" == true ]]; then
    COUNT=$(find "$SCRIPT_DIR" -type d -name '.venv' -prune -o \
      \( -type d -name '__pycache__' -o -type f -name '*.pyc' -o -type f -name '*.pyo' \) -print 2>/dev/null | wc -l | xargs || echo 0)
    warn "[dry-run] Would delete $COUNT __pycache__ dirs / .pyc files"
  else
    find "$SCRIPT_DIR" -type d -name '.venv' -prune -o \
      \( -type d -name '__pycache__' -o -type f -name '*.pyc' -o -type f -name '*.pyo' \) \
      -exec rm -rf {} + 2>/dev/null || true
    ok "Deleted all __pycache__ dirs and .pyc / .pyo files"
  fi
else
  skip "__pycache__ (use --pycache to clear)"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
header "━━ Summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ "$DRY_RUN" == true ]]; then
  warn "Dry-run complete — no files were deleted"
  info "Remove --dry-run to actually delete"
else
  ok "Cleanup complete"
  echo
  echo -e "  ${DIM}Run ./run.sh to reprocess your photos from scratch.${RESET}"
  echo -e "  ${DIM}If you kept the cache, ./run.sh will skip already-processed photos.${RESET}"
fi
echo
