# BaCuSe₂ XRD + BO pipeline: handoff for the NOMAD-connected agent

Date of handoff: 2026-10-23

## 1. Goal

Feed the XRD analysis and Bayesian optimization (BO) pipeline with data pulled directly from NOMAD. Local files are only a fallback for when the NOMAD API is down.

The pipeline needs, per film (sample):

1. **Synthesis parameters** (from the NOMAD archive JSON).
2. **Selenization pulse parameters** (not yet reliably recorded; see section 4).
3. **Raw XRD scan** (`.xrdml`), or the 2θ and intensity arrays from it.

Everything downstream is already written and runs. The NOMAD agent's job is to deliver the input tables below in the expected format.

## 2. What the pipeline consumes

### 2a. Synthesis table (one row per film)

Current extraction code: [bacuse2_synthesis_extraction.ipynb](/home/dagar/nomad-oasis-inl/xrd_references/bacuse2_synthesis_extraction.ipynb). It reads archive JSON and produces `bacuse2_synthesis_extraction.csv`.

Columns expected (names as they appear in the CSV):

| Column | Meaning | Archive path used by the current code |
|---|---|---|
| `sample_id` | Film ID, e.g. `260429A11` | from upload name `raw/<sample_id> BaCuSe ...archive.json` |
| `series` | Series letter/date group | derived |
| `solution_name` | Precursor solution | |
| `solution_volume_ml` | | |
| `spin_speed_rpm` | Spin speed | `spin.speed_rpm` |
| `spin_duration_s` | | `spin.duration_s` |
| `spin_acceleration` | | |
| `soft_bake_temp_c` | Soft bake temperature | |
| `soft_bake_duration_s` / `_min` | | |
| `hard_bake_temp_c` | Hard bake temperature | |
| `hard_bake_duration_s` / `_min` | | |
| `selenization_temp_c` | Selenization temperature | `selenization.temperature_c` |
| `selenization_time_s` | **Total** selenization time | `selenization.time_s` |
| `selenization_pressure_pa` | | `selenization.working_pressure_pa` |

BO uses these five inputs: `spin_speed_rpm`, `soft_bake_temp_c`, `hard_bake_temp_c`, `selenization_temp_c`, `selenization_time_s`.

Notes:
- The code lists upload IDs in `BACUSE_UPLOAD_IDS`. The last run found **90 uploads** with BaCuSe archives. Several uploads per sample exist (`BaCuSe`, `BaCuSe GI02`, `_processing`, `_Se_processing`). Pick the archive that holds the process record, and check for duplicates per `sample_id`.
- The NOMAD notebook reads from `./.volumes/fs/staging/` (local volume). The API path does not exist yet. It has to be written, with the archive JSON as the record format.

### 2b. Metrics table (output of the XRD notebook, input to BO)

File: [bacuse2_xrd_analysis_metrics.csv](/home/dagar/nomad-oasis-inl/xrd_references/bacuse2_xrd_analysis_metrics.csv). 25 rows.

Key columns: `sample_id`, `crystallinity` (1 / primary FWHM of the strongest BaCu₂Se₂ match), `primary_fwhm`, `phase_purity` (target / (target + impurity), integrated over ±0.5°), plus the synthesis columns above.

The BO notebook reads this file. If NOMAD is the new source, the XRD notebook must write this file, and the BO notebook must read it.

## 3. XRD data

- Format: Panalytical-style `.xrdml`, 2500 points over 2θ 10.01–59.99° (step ≈ 0.02°).
- Parsing: read the `dataPoints` element, the `positions axis="2Theta"` values, and the space-separated `intensities`. Strip XML namespaces by splitting tags on `}`.
- Local backup copies: `xrd_references/experimental_files/` (25 `.xrdml`).
- Current XRD loader: `XRD_SOURCE = "nomad" | "local" | "auto"` in the XRD notebook. `auto` tries NOMAD first and falls back to local files. NOMAD returned 0 scans the last time it was tried, so that path is unverified.

Sample list with XRD (25): the 25 IDs in `experimental_files/`. The synthesis DB has 37 samples. 12 have no XRD. `260901B3` and `260929A4` have no XRD metrics.

## 4. Selenization pulse data (open issue)

The selenization step is pulsed. Pulse parameters matter to the user and are not yet reliable.

Fields in [bacuse2_optimization_database_24samples.csv](/home/dagar/nomad-oasis-inl/xrd_references/bacuse2_optimization_database_24samples.csv):

- `selen_time_min`: total time (20 min = 1200 s for every film with pulse data).
- `se_pulse_ms`: pulse on-time (300, 500, or 1000 ms).
- `se_pulse_interval_ms`: pulse off-time (2900, 2700, or 2200 ms).

On + off = 3200 ms in every case, so about 375 pulses per 20 min. Duty cycle is 9–31%.

Coverage: only 6 of 24 films have pulse data (`260721D1`, `D2`, `D3`, `D4`, `D6`). Four of these are in the BO set.

The synthesis DB (`bacuse2_synthesis_database_37samples.csv`) has only total time (600, 1200, 1800 s).

**Please check:** do the NOMAD archives record pulse on-time, off-time, and count? If yes, add them to the synthesis table as `se_pulse_on_ms`, `se_pulse_off_ms`, `se_pulse_count`. The user is correcting this data by hand, so ask before overwriting anything.

## 5. Reference cards (local, not on NOMAD)

Folder: `xrd_references/`. These are ICDD-style cards.

| File | Phase | Role |
|---|---|---|
| `01-080-2950 BaCu2Se2.rtf` | BaCu₂Se₂ | Target |
| `04-024-2128 Cu2Se.rtf` | Cu₂Se | Impurity |
| `96-900-8602 BaSe.rtf` | BaSe | Matched for purity, not drawn in plots |
| `96-810-2887 BaSeO4.rtf` | BaSeO₄ | Matched for purity, not drawn in plots |
| `BaSeO3_XRD_Pattern_Reference-v2.rtf` | BaSeO₃ | Impurity, drawn. **AI-generated and unverified** (its lattice does not index its listed 2θ values) |

Keep these. Drop `96-100-0034 BaCO3.rtf` and `96-810-2887 BaSeO3.rtf` (the latter is a duplicate of BaSeO₄ with the wrong name).

Matching settings: `REF_MIN_INTENSITY = 10` (%), `MATCH_TOL = 0.15°`.

## 6. Objective (BO)

`objective = 0.7 × phase_purity + 0.3 × crystallinity_norm`

- `crystallinity_norm` is min-max normalized over the observed films, so the score shifts when new films widen the range.
- Weights are an assumption from the user's note "0.7, 0.5" (read as 0.7 purity, 0.3 crystallinity). Not yet confirmed.
- Current BO data: 21 films after dropping rows with missing values (`260429A2`, `260721D11`, `260721D3`, `260721D5`).

## 7. Known issues to keep in mind

- Films with no BaCu₂Se₂ match get `crystallinity = NaN` and `purity = 0`.
- Purity is 1.0 when no impurity matches, which may overstate it.
- A possible BaCu₂Se₂ (011) line at 22.64° (I ≈ 4.6%) sits just outside the 10% cutoff. The ~23° peak may be texture, not BaSeO₃.
- The XRD notebook still contains an old NOMAD cell (~line 2010) that fails with `KeyError: 'spin_speed_rpm'`. Replace it with the new loader; do not run it as-is.
- The legacy `.py` scripts in `xrd_references/` are broken. Do not use them.

## 8. Requested deliverables from the NOMAD agent

1. Confirm the NOMAD API access method, and the archive upload pattern per `sample_id`.
2. Export a synthesis table with the columns in section 2a for every film with an archive. Flag duplicates.
3. Add pulse fields (section 4) if the archives record them. Report which films have them.
4. Export the `.xrdml` scans for the 25 XRD samples, or confirm they are still in `experimental_files/`.
5. Return the list of `sample_id`s that have both XRD and synthesis data.

Please do not change the metrics or objective definitions in section 2b or 6 without asking the user.
