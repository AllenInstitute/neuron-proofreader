#!/usr/bin/env bash
# Sweep every test fragment in the merge-detection test set and tabulate
# dense vs biased-sparse sample counts and recall against ground-truth
# merge sites. Resumable: persists each fragment to the output CSV
# immediately, and re-running with --resume picks up where it left off.
#
# Usage:
#   tests/run_full_test_set.sh                                 # default output: tests/test_set_recall.csv
#   tests/run_full_test_set.sh --resume                        # skip fragments already in CSV
#   tests/run_full_test_set.sh --output-csv path/to/out.csv    # custom output
#   tests/run_full_test_set.sh --branch-radius 40 ...          # tune sparse params
#
# Override the merge-detection data dir with MERGEDET_DATA_DIR.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${MERGEDET_DATA_DIR:-$REPO_DIR/../neurobase_mergedetection/data/merge_proofreading}"

VENV_PY="$REPO_DIR/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
    echo "ERROR: venv python not found at $VENV_PY -- run 'uv sync' first." >&2
    exit 1
fi

if [[ ! -f "$DATA_DIR/merge_sites_df.csv" || ! -f "$DATA_DIR/test_idxs.csv" ]]; then
    echo "ERROR: cannot find test CSVs under $DATA_DIR" >&2
    echo "Set MERGEDET_DATA_DIR to override." >&2
    exit 1
fi

cd "$REPO_DIR"
exec "$VENV_PY" tests/run_full_test_set.py \
    --merge-sites-csv "$DATA_DIR/merge_sites_df.csv" \
    --test-idxs-csv   "$DATA_DIR/test_idxs.csv" \
    "$@"
