# BaCuSe₂ Thin-Film Analysis and Bayesian Optimization

This project analyzes BaCuSe₂ thin films using synthesis parameters and X-ray
diffraction (XRD) scans from the NOMAD Oasis. The analysis pipeline matches
observed XRD peaks to reference cards to identify the BaCu₂Se₂ target phase and
the Cu₂Se, BaSe, BaSeO₄, and BaSeO₃ impurity phases. It computes phase purity
and crystallinity for each film, then uses a Gaussian-process Bayesian
optimizer to suggest synthesis conditions that maximize a combined score.

The optimization parameters are spin speed, bake temperatures, selenization
temperature, and selenization time.

## Repository structure

- `xrd_bo_optimization_tf/` — reusable NOMAD, synthesis extraction, XRD
  processing, phase matching, plotting, and Bayesian-optimization functions.
- `notebooks/` — notebooks for synthesis extraction, XRD analysis, and
  Bayesian optimization.
- `data/raw/` — local `.xrdml` backups.
- `data/references/` — ICDD reference cards (`.rtf`).
- `data/processed/` — synthesis tables, metrics, and pulse data.
- `data/results/` — Bayesian-optimization suggestions and computed outputs.
- `outputs/figures/` and `outputs/reports/` — generated figures and reports.
- `docs/` — project documentation and handoff material.

## Getting started

Use a Python virtual environment outside the repository if you want to keep
environment files out of version control. Install the listed dependencies with:

```bash
python -m pip install -r requirements.txt
```

The notebooks are workflow entry points that call the reusable package
functions. The synthesis notebook reads a local NOMAD archive export when
available and otherwise loads the supplied synthesis table. The XRD notebook
analyzes local `.xrdml` scans when present and otherwise uses the supplied
metrics table. Bayesian optimization uses the complete rows in that metrics
table.

## Data and reference materials

Place local XRD backups in `data/raw/`, reference cards in `data/references/`,
and generated or curated tables in the corresponding `data/processed/` and
`data/results/` directories. A local NOMAD archive export is read from
`.volumes/fs/staging/` if present; the project does not make NOMAD network
requests yet. Check the terms governing any NOMAD data and reference-card
materials before redistributing them.
