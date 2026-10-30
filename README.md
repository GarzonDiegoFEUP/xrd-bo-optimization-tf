# α-BaCu₂Se₂ Thin-Film Analysis and Bayesian Optimization

This project analyzes α-BaCu₂Se₂ thin films using synthesis parameters and
X-ray diffraction (XRD) scans pulled directly from the NOMAD Oasis REST API.
The pipeline matches observed XRD peaks to ICDD reference cards to identify the
α-BaCu₂Se₂ target phase and the Cu₂Se and BaSeO₃ impurity phases (BaSe and
BaSeO₄ are matched for purity but not drawn in plots). It computes phase purity
and crystallinity for each film, then uses a Gaussian-process Bayesian
optimizer to suggest synthesis conditions that maximize a combined score.

The optimization space has five inputs: hard-bake temperature, selenization
temperature, selenization time, precursor Cu/(Ba+Cu) ratio (CBC), and continuous
Se pulse duty cycle. Pulse on/off durations are derived from a fixed 3200 ms
(3.2 s) period; frequency is therefore constant at 0.3125 Hz and is reported
as metadata, not fitted as an optimizer dimension. Suggestions quantize the on
time to 1 ms increments and report the resulting on/off recipe values. Only
three duty-cycle levels are observed, so intermediate suggestions rely on an
exploratory smooth-interpolation assumption. Spin speed,
soft-bake temperature, spin-to-selenization gap, spin-cycle count, and Mo
coating remain recorded descriptors, not optimizer inputs. The gap and coating
effects need deliberate matched experiments before inclusion. Two optimization
strategies are provided:

- **Single objective** (notebook 3, scikit-optimize): maximize
  `(0.7 × phase_purity + 0.3 × crystallinity_norm) × (1 − 0.25 × A)`, where
  `A` is the unmatched/overlap detected-peak-area fraction. This fixed penalty
  is a provisional screening proxy, not a phase fraction or validated impurity
  metric; it does not identify the ~23° feature. The four numeric inputs are
  restricted to the shared experimental recipe grids and duty cycle is modeled
  continuously over the observed timing bounds.
- **Multi-objective** (notebook 4, Ax/BoTorch): maximize phase purity and
  normalized crystallinity separately, each multiplied by the same configured
  `(1 − 0.25 × A)` screening factor, and trace the Pareto front (qNEHVI).
  The same five inputs and numeric grids are used (including 25 °C hard-bake
  steps), with duty cycle as a continuous input. This remains a Pareto formulation, not the same scalar
  objective as notebook 3.

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
  `01-080-2950 BaCu2Se2` (target), `96-101-0582 Cu2Se` (α-Cu2Se), `00-053-0141 BaSeO3`,
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
