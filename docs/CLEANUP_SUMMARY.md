# Cleanup Summary

## Completed

- renamed the active Django root from `server/` to `backend/`
- reorganized runtime code into focused directories
- archived deprecated Django templates, backup views, experiments, and helper scripts
- moved non-runtime outputs under `artifacts/`
- normalized local test data layout
- introduced a unified CLI at `scripts/run_inference.py`
- simplified the active Django app to inference-only behavior
- removed the old product and ingredient schema from the active backend
- extracted shared runtime helpers into `pipelines/runtime_utils.py`

## Current Active Entry Points

- `scripts/run_inference.py`
- `backend/manage.py`
- `backend/atomom/views.py`
- `backend/atomom/predictor.py`

## Important Local-Only Assets

- model weights under `models/weights/`
- local datasets and samples under `test_data/`
- generated outputs under `artifacts/`
