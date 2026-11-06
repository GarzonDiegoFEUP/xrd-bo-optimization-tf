# α-BaCu₂Se₂ Thin-Film Analysis and Bayesian Optimization

This project analyzes α-BaCu₂Se₂ thin films using synthesis parameters and
X-ray diffraction (XRD) scans pulled directly from the NOMAD Oasis REST API.
The pipeline matches observed XRD peaks to ICDD reference cards to identify the
α-BaCu₂Se₂ target phase, impurities (β-BaCu₂Se₂, Cu₂Se, CuSe, BaSe, BaSe₂,
BaSe₃, BaSeO₃, BaSeO₄), and Au/In
contact-layer peaks. In is considered a contact only in series 260929A; Au may
be matched across series. Contacts are classified separately and plotted with
one shared marker, but their reference-card bars are hidden. Contact intervals
are excluded from both the target-vs-impurity phase-purity calculation and the
unmatched/overlap screening penalty. The α target is always drawn in the XRD
plots; impurity reference bars and markers appear only for phases with matched
peaks. The pipeline computes phase purity and
crystallinity for each film, then uses a Gaussian-process Bayesian optimizer to
suggest synthesis conditions that maximize a combined score.

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
  `A` is the unmatched/overlap detected-peak-area fraction after known Au/In
  contact intervals are excluded (In only in series 260929A). This fixed penalty is a provisional screening
  proxy, not a phase fraction or validated impurity metric; it does not
  independently identify the ~23° feature. The four numeric inputs are
  restricted to the shared experimental recipe grids and duty cycle is modeled
  continuously over the observed timing bounds.
- **Multi-objective** (notebook 4, Ax/BoTorch): maximize phase purity and
  normalized crystallinity separately, each multiplied by the same configured
  `(1 − 0.25 × A)` screening factor, and trace the Pareto front (qNEHVI). Au/In
  contact intervals are excluded from `A` in this objective as well.
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
  2.1. `2.1.xrd_260901B2_omega.ipynb` — extra XRD analysis for sample 260901B2
     only: seven grazing-incidence scans at omega 0.1–1.1° (20–35° 2θ). It fits
     each peak per omega (`data/results/260901B2_omega_peak_fits.csv`), plots
     intensity vs omega, and compares the ~23° feature with the other 260901
     samples to test whether it belongs to the BaCuSe layer or to an
     impurity/secondary phase. It uses a lowered 3 % card-intensity threshold to include the weak alpha/beta lines near 23°. It does not feed the optimizer; notebook 2 uses the full-range
     GI 0.2° scan for this sample.
  3. `3.bo_xrd.ipynb` — exploratory analysis (Spearman correlations, GP
     objective surface) and single-objective Bayesian-optimization suggestions
     (scikit-optimize, weighted objective). It also compares four treatments of
     the unassigned ~23° feature (as-is, excluded, as target, as impurity;
     `data/results/bacuse2_bo_feature23_sensitivity.csv`) without changing the
      production objective. A separate 0.5/0.5, 0.7/0.3, 0.9/0.1 purity/
      crystallinity weight sweep saves observed scores and GP proposals to
      `data/results/bacuse2_bo_weight_sensitivity_observed.csv` and
      `data/results/bacuse2_bo_weight_sensitivity_suggestions.csv`.
  4. `4.bo_xrd_ax.ipynb` — multi-objective Bayesian optimization with Ax
     (BoTorch): phase purity and crystallinity are separate objectives, the
     Pareto front is found via qNEHVI, and suggestions are compared against
      notebook 3. A separate experimental Ax run maximizes raw purity and
      normalized crystallinity while minimizing unassigned/overlap area fraction
      A, with hypervolume reference thresholds (0, 0, 1.05). Its observed
      nondominated films and model-generated suggestions are saved separately as
      `bacuse2_ax_experimental_3objective_observed_nondominated.csv` and
      `bacuse2_ax_experimental_3objective_suggestions.csv` in `data/results/`.
      These are not replacements for the screened production suggestions: A is a
      screening proxy, includes the ambiguous ~23° feature, and is not an impurity
      fraction. Candidate predictions and the observed front are not the exact
      underlying trade-off surface.
- `data/raw/` — local `.xrdml` backups; `data/raw/nomad_staging/` is a
  gitignored cache of NOMAD archives downloaded by notebook 1.
- `data/references/` — ICDD reference cards (`.rtf`), keyed in
  `REFERENCE_CARD_FILES` in `config.py`: `01-080-2950 alpha BaCu2Se2` (target),
  `04-008-2861 beta BaCu2Se2` (tetragonal polymorph, counted as impurity),
  `01-080-4377 Cu2Se`, `00-049-1456 CuSe`, `01-071-4770 BaSe`,
  `04-004-8560 BaSe2`, `04-004-8559 BaSe3`, `04-014-7500 BaSeO3`,
  `04-011-7418 BaSeO4`, and the contact layers `00-004-0784 Au` and
  `00-005-0642 In` (excluded from purity and penalty). Only reflections of at
  least 10 % relative intensity are matched; when several impurity cards match,
  the closest reflection wins, and the α target wins any overlap.
- `data/processed/` — synthesis tables and the BaCuSe upload-ID list
  (`bacuse_upload_ids.txt`).
- `data/results/` — generated XRD metrics, classified per-peak export
  (`bacuse2_xrd_peak_list.csv`), ~23° feature descriptors
  (`bacuse2_xrd_feature23_descriptors.csv`: position, FWHM, area, ratio to the
  α 29.4° cluster; not used by the optimizer), and BO suggestions.
- `data/cif_files/` — CIF structures for the exploratory Le Bail fit in notebook
  2. The standard fit uses α- and β-BaCu2Se2, BaSeO3 and BaSe (the most
  frequently matched impurities); the other CIFs (Cu2Se, CuSe, BaSe2, BaSe3,
  BaSeO4) are kept for comparison runs, summarized in
  `data/results/bacuse2_lebail_phase_set_comparison.csv`.
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
