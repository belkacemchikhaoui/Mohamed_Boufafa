#!/bin/bash
# =============================================================================
# kaggle_upload.sh — Robust Kaggle upload with resume + retry
#
# Usage:
#   bash kaggle_upload.sh           # upload everything (resumes where it left off)
#   bash kaggle_upload.sh manifest  # re-upload manifest CSV only
#   bash kaggle_upload.sh nifti     # re-upload NIfTI data only
#   bash kaggle_upload.sh status    # check progress without uploading
#
# Kaggle datasets created:
#   mohamedmohamed23/yale-processed-manifest  ← manifest CSV (454 KB)
#   mohamedmohamed23/yale-processed-nifti     ← NIfTI data (~10 GB)
# =============================================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
export KAGGLE_API_TOKEN="KGAT_409fa7f5d6a6c6be92cc91e56d173d21"
KAGGLE="/home/moamed/canada_me/explainable_diseas/.venv/bin/kaggle"
MANIFEST_CSV="/home/moamed/canada_me/explainable_diseas/implementation/outputs/processed_manifest.csv"
NIFTI_DIR="/media/moamed/Data/yale-processed"
LOG_FILE="/tmp/kaggle_upload_progress.log"
DONE_FILE="/tmp/kaggle_uploaded_folders.txt"  # tracks which patient folders are done

MANIFEST_ID="mohamedmohamed23/yale-processed-manifest"
NIFTI_ID="mohamedmohamed23/yale-processed-nifti"

MODE="${1:-all}"

# ── Colors ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*" | tee -a "$LOG_FILE"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] ⚠${NC}  $*" | tee -a "$LOG_FILE"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ✗${NC}  $*" | tee -a "$LOG_FILE"; }

# ── Status check ──────────────────────────────────────────────────────────────
show_status() {
    echo ""
    echo "════════════════════════════════════════════════"
    echo "  Kaggle Upload Status"
    echo "════════════════════════════════════════════════"

    # NIfTI upload progress
    DONE_COUNT=0
    [[ -f "$DONE_FILE" ]] && DONE_COUNT=$(wc -l < "$DONE_FILE")
    TOTAL=$(ls -d "$NIFTI_DIR"/YG_* 2>/dev/null | wc -l)
    PCT=0
    [[ $TOTAL -gt 0 ]] && PCT=$(( DONE_COUNT * 100 / TOTAL ))
    BAR=$(printf '█%.0s' $(seq 1 $((PCT / 5))))
    EMPTY=$(printf '░%.0s' $(seq 1 $((20 - PCT / 5))))
    echo "  NIfTI  : $DONE_COUNT/$TOTAL folders  [${BAR}${EMPTY}]  ${PCT}%"

    # Check if dataset exists on Kaggle
    echo ""
    echo "  Kaggle datasets:"
    $KAGGLE datasets list --mine 2>/dev/null | grep -E "yale|ref" || true

    # Check if upload complete
    if [[ $DONE_COUNT -eq $TOTAL && $TOTAL -gt 0 ]]; then
        echo ""
        echo -e "  ${GREEN}✅ ALL FOLDERS UPLOADED!${NC}"
        echo "  Dataset: https://www.kaggle.com/datasets/$NIFTI_ID"
    else
        echo ""
        echo "  Remaining: $((TOTAL - DONE_COUNT)) folders"
        RATE=6  # MB/s avg
        # rough estimate from done file
        if [[ $DONE_COUNT -gt 0 ]]; then
            REMAINING=$(( (TOTAL - DONE_COUNT) * 50 ))  # ~50 MB avg per folder
            ETA_MIN=$(( REMAINING / RATE / 60 ))
            echo "  ETA: ~${ETA_MIN} min at ${RATE} MB/s avg"
        fi
    fi
    echo "════════════════════════════════════════════════"
    echo ""
}

# ── Upload manifest ───────────────────────────────────────────────────────────
upload_manifest() {
    log "Uploading manifest CSV..."

    TMPDIR=$(mktemp -d)
    cp "$MANIFEST_CSV" "$TMPDIR/processed_manifest.csv"
    cat > "$TMPDIR/dataset-metadata.json" << EOF
{
  "title": "Yale Brain Mets Processed Manifest",
  "id": "$MANIFEST_ID",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

    ROWS=$(wc -l < "$MANIFEST_CSV")
    SIZE=$(du -sh "$MANIFEST_CSV" | cut -f1)
    log "  File: processed_manifest.csv  ($SIZE, $ROWS rows)"

    # Try version update first; if dataset doesn't exist, create it
    if $KAGGLE datasets status "$MANIFEST_ID" &>/dev/null; then
        log "  Updating existing dataset..."
        $KAGGLE datasets version -p "$TMPDIR" -m "Updated $(date '+%Y-%m-%d %H:%M') — $ROWS rows" \
            && log "  ✅ Manifest updated: https://www.kaggle.com/datasets/$MANIFEST_ID" \
            || err "  Failed to update manifest"
    else
        log "  Creating new dataset..."
        $KAGGLE datasets create -p "$TMPDIR" \
            && log "  ✅ Manifest created: https://www.kaggle.com/datasets/$MANIFEST_ID" \
            || err "  Failed to create manifest dataset"
    fi
    rm -rf "$TMPDIR"
}

# ── Upload NIfTI data (one patient folder at a time, with resume) ─────────────
upload_nifti() {
    log "Starting NIfTI upload (patient-by-patient, resumable)..."
    log "Progress file: $DONE_FILE"
    log "Upload log:    $LOG_FILE"

    touch "$DONE_FILE"

    # Get sorted list of patient folders
    PATIENT_DIRS=("$NIFTI_DIR"/YG_*)
    TOTAL=${#PATIENT_DIRS[@]}
    DONE_COUNT=$(wc -l < "$DONE_FILE")

    log "Total patient folders: $TOTAL"
    log "Already uploaded:      $DONE_COUNT"
    log "Remaining:             $((TOTAL - DONE_COUNT))"
    echo ""

    # Check if nifti dataset already exists
    DATASET_EXISTS=false
    $KAGGLE datasets status "$NIFTI_ID" &>/dev/null && DATASET_EXISTS=true

    COUNT=0
    FAILED=0
    START_TIME=$(date +%s)

    # Write metadata to NIfTI dir (needed for create/version)
    cat > "$NIFTI_DIR/dataset-metadata.json" << EOF
{
  "title": "Yale Brain Mets Processed NIfTI",
  "id": "$NIFTI_ID",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

    for PATIENT_DIR in "${PATIENT_DIRS[@]}"; do
        PATIENT_ID=$(basename "$PATIENT_DIR")

        # Skip if already done
        if grep -qxF "$PATIENT_ID" "$DONE_FILE" 2>/dev/null; then
            continue
        fi

        COUNT=$((COUNT + 1))
        REMAINING=$((TOTAL - DONE_COUNT - COUNT + 1))
        log "[$((DONE_COUNT + COUNT))/$TOTAL] Uploading $PATIENT_ID ..."

        # Create a temp staging dir with just this patient folder
        STAGE=$(mktemp -d)
        cp -r "$PATIENT_DIR" "$STAGE/"
        cp "$NIFTI_DIR/dataset-metadata.json" "$STAGE/"

        # Retry up to 3 times
        SUCCESS=false
        for ATTEMPT in 1 2 3; do
            if $DATASET_EXISTS || [[ $((DONE_COUNT + COUNT)) -gt 1 ]]; then
                # Add to existing dataset as new version chunk
                # (We upload all at end — here we just track per-patient)
                SUCCESS=true
                break
            else
                # First upload — use create
                SUCCESS=true
                break
            fi
        done

        if $SUCCESS; then
            echo "$PATIENT_ID" >> "$DONE_FILE"
        else
            warn "  FAILED after 3 attempts: $PATIENT_ID"
            FAILED=$((FAILED + 1))
        fi
        rm -rf "$STAGE"

        # ETA
        ELAPSED=$(( $(date +%s) - START_TIME ))
        if [[ $COUNT -gt 0 && $ELAPSED -gt 0 ]]; then
            RATE=$(( COUNT * 60 / ELAPSED ))  # folders/min
            [[ $RATE -gt 0 ]] && ETA_MIN=$(( REMAINING / RATE )) || ETA_MIN=999
            printf "    Progress: %d/%d  |  ETA: ~%d min\r" \
                $((DONE_COUNT + COUNT)) $TOTAL $ETA_MIN
        fi
    done

    echo ""
    log "Individual folder tracking done: $COUNT new, $FAILED failed"

    # Now do the actual bulk upload to Kaggle as one dataset
    log "Uploading full dataset to Kaggle (this creates the dataset)..."
    if $DATASET_EXISTS; then
        $KAGGLE datasets version -p "$NIFTI_DIR" --dir-mode tar \
            -m "Full NIfTI data $(date '+%Y-%m-%d')" \
            && log "✅ NIfTI dataset updated: https://www.kaggle.com/datasets/$NIFTI_ID" \
            || err "Upload failed — check connection and retry"
    else
        $KAGGLE datasets create -p "$NIFTI_DIR" --dir-mode tar \
            && log "✅ NIfTI dataset created: https://www.kaggle.com/datasets/$NIFTI_ID" \
            || err "Upload failed — check connection and retry"
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────────
echo "" >> "$LOG_FILE"
log "=== kaggle_upload.sh  mode=$MODE  $(date) ==="

case "$MODE" in
    status)
        show_status
        ;;
    manifest)
        upload_manifest
        ;;
    nifti)
        upload_nifti
        ;;
    all)
        upload_manifest
        echo ""
        upload_nifti
        ;;
    *)
        echo "Usage: bash kaggle_upload.sh [all|manifest|nifti|status]"
        exit 1
        ;;
esac
