# α-BaCu₂Se₂ Thin-Film Analysis and Bayesian Optimization

This project analyzes α-BaCu₂Se₂ thin films using synthesis parameters and
X-ray diffraction (XRD) scans pulled directly from the NOMAD Oasis REST API.
The pipeline matches observed XRD peaks to ICDD reference cards to identify the
α-BaCu₂Se₂ target phase and the Cu₂Se and BaSeO₃ impurity phases (BaSe and
BaSeO₄ are matched for purity but not drawn in plots). It computes phase purity
and crystallinity for each film, then uses a Gaussian-process Bayesian
optimizer to suggest synthesis conditions that maximize a combined score.

The optimization space has nine parameters: spin speed, soft-bake and
hard-bake temperatures, selenization temperature, selenization time, the
selenium pulse on/off times, the precursor Cu/(Ba+Cu) ratio (CBC), and the
spin-to-selenization day gap. Each synthesis record also carries the number of
spin-coating cycles (deposited layers, excluding dynamic-dispense/seed-layer
preparation steps) and a Mo-coating flag (substrate) as reported descriptors.
The exploratory correlation plot in notebook 3 ranks all twelve tracked
parameters, not just the nine in the optimization space. Two optimization
strategies are provided:

- **Single objective** (notebook 3, scikit-optimize): maximize
  `0.7 × phase_purity + 0.3 × crystallinity_norm`.
- **Multi-objective** (notebook 4, Ax/BoTorch): maximize phase purity and
  normalized crystallinity separately and trace the Pareto front (qNEHVI).

## Repository structure

- `xrd_bo_optimization_tf/` — reusable package: NOMAD API client
  (`nomad_api.py`), local-archive helpers (`nomad_client.py`), synthesis
  extraction (`synthesis.py`), XRD loading (`xrd_io.py`), peak analysis
  (`peak_analysis.py`), phase matching (`phase_matching.py`), plotting
  (`plotting.py`), Bayesian optimization (`bayesian_opt.py`), and PDF report
  generation (`report.py`).
- `notebooks/` — workflow entry points:
  1. `1. synthesis_parameters_extraction.ipynb` — pulls synthesis records from
     NOMAD (falls back to a local export, then a CSV) and writes the complete
     database to `data/processed/bacuse2_synthesis_database_nomad.csv`.
  2. `2.xrd_analysis.ipynb` — loads XRD scans from the staged NOMAD archives,
     matches peaks, computes metrics, and writes
     `data/results/bacuse2_xrd_analysis_metrics.csv` plus figures.
  3. `3.bo_xrd.ipynb` — exploratory analysis (Spearman correlations, GP
     objective surface) and single-objective Bayesian-optimization suggestions
     (scikit-optimize, weighted objective).
  4. `4.bo_xrd_ax.ipynb` — multi-objective Bayesian optimization with Ax
     (BoTorch): phase purity and crystallinity are separate objectives, the
     Pareto front is found via qNEHVI, and suggestions are compared against
     notebook 3.
- `data/raw/` — local `.xrdml` backups; `data/raw/nomad_staging/` is a
  gitignored cache of NOMAD archives downloaded by notebook 1.
- `data/references/` — ICDD reference cards (`.rtf`):
  `01-080-2950 BaCu2Se2` (target), `04-024-2128 Cu2Se`, `00-053-0141 BaSeO3`,
  plus `96-900-8602 BaSe` and `96-810-2887 BaSeO4` (matched, not plotted).
- `data/processed/` — synthesis tables and the BaCuSe upload-ID list
  (`bacuse_upload_ids.txt`).
- `data/results/` — generated XRD metrics and BO suggestions.
- `outputs/figures/` and `outputs/reports/` — generated figures and the PDF
  report.

## Getting started

Create a virtual environment (for example `xrd_bo_env/`, which is gitignored)
and install the pinned dependencies:

```bash
python3.12 -m venv xrd_bo_env
source xrd_bo_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Select the `xrd_bo_env` interpreter as the kernel for the notebooks.

## NOMAD access

The pipeline reads from a NOMAD Oasis over its REST API. Create a personal
access token in the Oasis GUI (Profile → Settings → Tokens, READ ONLY scopes),
copy `.env.example` to `.env`, and paste the token:

```bash
NOMAD_API_URL="https://nomad-sadewasser.inl.local/nomad-oasis/api/v1"
NOMAD_PAT="paste-your-token-here"
NOMAD_VERIFY_SSL="0"   # the Oasis serves a self-signed certificate
```

`.env` is gitignored and never committed. BaCuSe uploads are routed by ID via
`data/processed/bacuse_upload_ids.txt` (entry search by element is not reliable
for these ELN records). Samples are the `INLThinFilmStack` entries; XRD scans
come from `ELNXRayDiffraction` entries, whose archives carry the parsed 2θ and
intensity arrays directly.

## Reports

Rebuild the PDF report from the latest tables and figures with:

```bash
xrd_bo_env/bin/python -c "from xrd_bo_optimization_tf.report import build_report; build_report()"
```

The report is written to `outputs/reports/bacuse2_xrd_bo_report.pdf`.

## Data and reference materials

Check the terms governing any NOMAD data and ICDD reference-card materials
before redistributing them.
