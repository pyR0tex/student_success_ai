# Results

This directory contains selected **derived project outputs** exported by:

```bash
python main.py --export-results
```

It is intentionally tracked in Git so report and presentation work can reference stable project evidence without committing the professor-provided raw CSV files.

Key files after export:

- `presentation_data.json` — compact summary of final metrics and project findings for report/presentation generation.
- `manifest.json` — file sizes and SHA-256 hashes for exported artifacts.
- `model/` — development, held-out, fairness, and failure-analysis outputs.
- `planner/` — academic-plan summaries and detailed course plans.
- `cases/` — end-to-end advisor case-study outputs.
- `logs/` — structured decision logs.
- `figures/` — generated evaluation figures.

The synthetic source datasets remain local under `data/` and are not exported here.
