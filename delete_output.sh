#!/usr/bin/env bash
# =============================================================================
#  delete_output.sh — Photo Curator  |  Delete Output Utility
# =============================================================================
#
#  USAGE
#    ./delete_output.sh           delete all files in data/output_photos
#    ./delete_output.sh --dry-run  show what would be deleted
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$SCRIPT_DIR/data/output_photos"
DRY_RUN=false

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    *) echo -e "${RED}Unknown option: $1${RESET}" >&2; exit 1 ;;
  esac
done

# ── Header ────────────────────────────────────────────────────────────────────
BAR="═══════════════════════════════════════════════════════════"
echo -e "${BOLD}${CYAN}${BAR}${RESET}"
echo -e "${BOLD}${CYAN}  Photo Curator — Delete Output$([ "$DRY_RUN" == true ] && echo "  [DRY RUN]")${RESET}"
echo -e "${BOLD}${CYAN}${BAR}${RESET}"

if [[ ! -d "$TARGET_DIR" ]]; then
  warn "Target directory not found: $TARGET_DIR"
  exit 0
fi

# Count files
FILE_COUNT=$(find "$TARGET_DIR" -type f ! -name ".gitkeep" 2>/dev/null | wc -l | xargs)

if [[ "$FILE_COUNT" -eq 0 ]]; then
  ok "Directory is already empty (no files to delete)."
  exit 0
fi

info "Target: $TARGET_DIR"
warn "Found $FILE_COUNT file(s) to delete."

if [[ "$DRY_RUN" == true ]]; then
  info "[dry-run] Listing files that would be deleted:"
  find "$TARGET_DIR" -type f ! -name ".gitkeep" -maxdepth 1
  echo
  warn "Dry-run mode — nothing was deleted."
else
  # Delete files but keep directory and .gitkeep
  find "$TARGET_DIR" -type f ! -name ".gitkeep" -delete
  ok "Successfully deleted $FILE_COUNT file(s)."
  # Ensure the directory still exists (should since we only deleted files)
  mkdir -p "$TARGET_DIR"
  touch "$TARGET_DIR/.gitkeep" 2>/dev/null || true
fi

echo -e "${DIM}${BAR}${RESET}"
echo
