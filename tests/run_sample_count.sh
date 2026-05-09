#!/usr/bin/env bash
# Sparse-vs-dense sample-count comparison.
#
# By default, picks an example (brain_id, segmentation_id) from the sibling
# repo's test set (../neurobase_mergedetection/data/merge_proofreading/) and
# resolves it to the GCS SWC pointer that merge_dataloading.load_fragments()
# would use.
#
# Usage:
#   tests/run_sample_count.sh                          # auto-pick from test_idxs (brain 747807)
#   tests/run_sample_count.sh <brain_id>               # auto-pick for given brain
#   tests/run_sample_count.sh <brain_id> <seg_id>      # explicit override
#   tests/run_sample_count.sh <full_swc_pointer>       # gs://, s3://, or absolute local path

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${MERGEDET_DATA_DIR:-$REPO_DIR/../neurobase_mergedetection/data/merge_proofreading}"
ROOT="gs://allen-nd-goog/automated_proofreading_dataset/raw_merge_sites"
DEFAULT_BRAIN="747807"

VENV_PY="$REPO_DIR/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
    echo "ERROR: venv python not found at $VENV_PY -- run 'uv sync' first." >&2
    exit 1
fi

# If the first arg looks like a full URI/path, pass it through unchanged.
if [[ $# -ge 1 ]]; then
    case "$1" in
        gs://*|s3://*|/*)
            POINTER="$1"
            echo "Using SWC pointer: $POINTER"
            cd "$REPO_DIR"
            exec "$VENV_PY" tests/manual_sample_count.py "$POINTER"
            ;;
    esac
fi

BRAIN_ID="${1:-$DEFAULT_BRAIN}"
SEG_ID="${2:-}"

if [[ -z "$SEG_ID" ]]; then
    if [[ ! -f "$DATA_DIR/merge_sites_df.csv" || ! -f "$DATA_DIR/test_idxs.csv" ]]; then
        echo "ERROR: cannot find test CSVs under $DATA_DIR" >&2
        echo "Set MERGEDET_DATA_DIR or pass <brain_id> <seg_id> explicitly." >&2
        exit 1
    fi
    SEG_ID="$(
        BRAIN_ID="$BRAIN_ID" DATA_DIR="$DATA_DIR" \
        "$VENV_PY" - <<'PY'
import os
import pandas as pd

data_dir = os.environ["DATA_DIR"]
brain_id = os.environ["BRAIN_ID"]

df = pd.read_csv(os.path.join(data_dir, "merge_sites_df.csv"))
df["brain_id"] = df["brain_id"].astype(str)

idxs = pd.read_csv(os.path.join(data_dir, "test_idxs.csv"))["Indices"].tolist()
test_df = df.iloc[idxs]
matches = test_df.loc[test_df["brain_id"] == brain_id, "segmentation_id"].unique()
if len(matches) == 0:
    raise SystemExit(f"no test rows for brain_id={brain_id}")
print(matches[0])
PY
    )"
fi

POINTER="$ROOT/$BRAIN_ID/$SEG_ID/merged_fragments.zip"
echo "Resolved test fragment:"
echo "  brain_id:        $BRAIN_ID"
echo "  segmentation_id: $SEG_ID"
echo "  pointer:         $POINTER"
echo

cd "$REPO_DIR"
EXTRA_ARGS=()
if [[ -f "$DATA_DIR/merge_sites_df.csv" ]]; then
    EXTRA_ARGS+=(
        --merge-sites-csv "$DATA_DIR/merge_sites_df.csv"
        --brain-id "$BRAIN_ID"
    )
    if [[ -f "$DATA_DIR/test_idxs.csv" ]]; then
        EXTRA_ARGS+=(--test-idxs-csv "$DATA_DIR/test_idxs.csv")
    fi
fi
exec "$VENV_PY" tests/manual_sample_count.py "$POINTER" "${EXTRA_ARGS[@]}"
