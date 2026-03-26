#!/usr/bin/env bash
# Example: run DR-Learner pipeline from model_build with an explicit MySQL JSON path.
# Copy and edit paths; do not commit real credentials.

set -euo pipefail
cd "$(dirname "$0")/.."  # model_build

python -m dr_learner_pipeline.run_pipeline \
  --config dr_learner_pipeline/config/pipeline_grid.yaml \
  --split_dates_path dr_learner_pipeline/example_split_dates.json \
  --db-config-json ../db_config.json \
  --db-table workmagic_bi.your_wide_table
