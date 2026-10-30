"""Exploratory multiphase Le Bail fit for the best GIXRD sample.

The selected scan is GIXRD at omega=0.2 deg. rietx 1.10 does not implement
fixed-incidence GIXRD geometry, so this script uses its flat-plate
Bragg-Brentano model only as a documented profile-fitting proxy. Do not treat
its fit statistics or phase quantities as a validated GIXRD quantitative
phase analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import rietx as rx


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ID = "260429B2"
CIF_DIR = ROOT / "data" / "cif_files"
NOMAD_ROOT = ROOT / "data" / "raw" / "nomad_staging"
OUTPUT_DIR = ROOT / "outputs" / "rietx_lebail_trial_fixed_cells"
CIF_FILES = (
    "BaCu2Se2.cif",
    "Cu2Se.cif",
    "BaSe.cif",
    "BaSeO3.cif",
    "BaSeO4.cif",
)


def _find_gixrd_archive() -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in NOMAD_ROOT.rglob("*.archive.json"):
        try:
            archive = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        data = archive.get("data", {})
        if not isinstance(data, dict):
            continue
        if "ELNXRayDiffraction" not in str(data.get("m_def", "")):
            continue
        if not str(data.get("name", "")).startswith(SAMPLE_ID):
            continue
        results = data.get("results")
        if not isinstance(results, list) or not results or not isinstance(results[0], dict):
            continue
        result = results[0]
        if result.get("two_theta") is None or result.get("intensity") is None:
            continue
        if "Grazing Incidence" not in str(data.get("diffraction_method_name", "")):
            continue
        omega = result.get("omega")
        if not isinstance(omega, list) or not omega or not np.isclose(float(omega[0]), 0.2):
            continue
        matches.append((path, data))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one GIXRD archive for {SAMPLE_ID} at omega=0.2 deg; "
            f"found {len(matches)}"
        )
    return matches[0]


def _diagnostic_record(diagnostic: Any) -> dict[str, Any]:
    if hasattr(diagnostic, "model_dump"):
        return diagnostic.model_dump(mode="json")
    return {"text": str(diagnostic)}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    archive_path, archive_data = _find_gixrd_archive()
    result_data = archive_data["results"][0]
    two_theta = np.asarray(result_data["two_theta"], dtype=np.float64)
    intensity = np.asarray(result_data["intensity"], dtype=np.float64)
    n_points = min(two_theta.size, intensity.size)
    two_theta, intensity = two_theta[:n_points], intensity[:n_points]
    if n_points < 2 or not np.all(np.diff(two_theta) > 0):
        raise ValueError("NOMAD scan must have at least two increasing 2-theta points")

    xy_path = OUTPUT_DIR / f"{SAMPLE_ID}_gixrd_omega_0p2.xy"
    np.savetxt(
        xy_path,
        np.column_stack((two_theta, intensity)),
        fmt="%.8g",
        header="two_theta_deg intensity_counts (exported from NOMAD archive)",
    )
    pattern = rx.PatternData(
        two_theta=two_theta.tolist(),
        intensity=intensity.tolist(),
        metadata={
            "sample_id": SAMPLE_ID,
            "source": "NOMAD ELNXRayDiffraction archive",
            "archive": str(archive_path.relative_to(ROOT)),
            "measurement_geometry": "Grazing Incidence X-Ray Diffraction (GIXRD)",
            "omega_deg": "0.2",
            "data_file": str(archive_data.get("data_file", "")),
        },
    )

    all_phases = []
    cif_diagnostics: list[dict[str, Any]] = []
    for cif_name in CIF_FILES:
        cif_path = CIF_DIR / cif_name
        if not cif_path.is_file():
            raise FileNotFoundError(f"Missing CIF: {cif_path}")
        notes: list[Any] = []
        cif_structure = rx.Structure.from_cif(
            cif_path, phase_name=cif_path.stem, diagnostics=notes
        )
        all_phases.extend(cif_structure.phases)
        cif_diagnostics.extend(
            {"cif": cif_name, **_diagnostic_record(note)} for note in notes
        )
    structure = rx.Structure(phases=all_phases)

    wavelength_kalpha1 = float(result_data["source_peak_wavelength"])
    settings = archive_data.get("xrd_settings", {}).get("source", {})
    wavelength_kalpha2 = float(settings.get("kalpha_two", 1.544426))
    ka2_ratio = float(settings.get("ratio_kalphatwo_kalphaone", 0.5))
    instrument = rx.Instrument.bragg_brentano(radiation="CuKa", ka2_ratio=ka2_ratio)
    instrument.source.lines[0].wavelength.value = wavelength_kalpha1
    instrument.source.lines[1].wavelength.value = wavelength_kalpha2
    instrument.background = rx.BackgroundChebyshev.with_terms(6)
    instrument.background.coefficients[0].value = float(np.percentile(intensity, 5))

    # rietx has no fixed-incidence GIXRD geometry; this proxy is explicitly
    # exploratory and is not a physical replacement for a GIXRD correction.
    geometry_note = (
        "rietx 1.10 has no fixed-incidence GIXRD geometry. The instrument uses "
        "flat-plate Bragg-Brentano reflection geometry as a profile-fitting proxy; "
        "interpret the fit and phase quantities as exploratory only."
    )
    # Keep CIF unit cells fixed for this phase-screening trial. The first
    # profile_only attempt let the triclinic BaSeO3 cell run away; fixed cells
    # make the phase fingerprint test stable and avoid pretending those cells
    # were measured from this thin-film scan.
    plan = rx.RefinementPlan(
        stages=[
            rx.Stage(stage.name, stage.turn_on, max_iter=200)
            for stage in rx.PLAN_PRESETS["profile_only"]().stages
            if stage.name != "cell"
        ],
        correlation_guard=0.98,
        intermediate_ftol=1e-6,
        lebail_passes=3,
        phase_support_sigma=3.0,
        closing_window_tol=1e-3,
    )
    refinement = rx.Refinement(structure, instrument, backend="numpy", solver="lm")
    fit = refinement.fit(
        pattern,
        mode="lebail",
        plan=plan,
        two_theta_limits=(float(two_theta.min()), float(two_theta.max())),
        label=f"{SAMPLE_ID}-GIXRD-omega0.2-multiphase-LeBail",
    )

    figure_path = OUTPUT_DIR / f"{SAMPLE_ID}_multiphase_lebail_fit.png"
    fit.plot(
        path=str(figure_path),
        wavelength=wavelength_kalpha1,
        figsize=(12, 7),
        title=f"{SAMPLE_ID} GIXRD, omega=0.2° — exploratory Le Bail fit",
    )
    peak_zoom_path = OUTPUT_DIR / f"{SAMPLE_ID}_22p637deg_peak_zoom.png"
    fit.plot(
        path=str(peak_zoom_path),
        wavelength=wavelength_kalpha1,
        two_theta_range=(20.0, 25.0),
        figsize=(11, 6),
        title=f"{SAMPLE_ID}: fit near BaCu2Se2 22.637° (4.6% reference line)",
    )
    fit_theta = np.asarray(fit.two_theta, dtype=np.float64)
    fit_observed = np.asarray(fit.y_obs, dtype=np.float64)
    fit_calculated = np.asarray(fit.y_calc, dtype=np.float64)
    weak_peak_mask = (fit_theta >= 22.137) & (fit_theta <= 23.137)
    weak_peak_indices = np.flatnonzero(weak_peak_mask)
    observed_peak_index = weak_peak_indices[np.argmax(fit_observed[weak_peak_mask])]
    calculated_peak_index = weak_peak_indices[np.argmax(fit_calculated[weak_peak_mask])]
    weak_peak_check = {
        "reference_two_theta_deg": 22.637,
        "window_deg": [22.137, 23.137],
        "observed_peak_two_theta_deg": float(fit_theta[observed_peak_index]),
        "observed_window_max_counts": float(fit_observed[weak_peak_mask].max()),
        "calculated_peak_two_theta_deg": float(fit_theta[calculated_peak_index]),
        "calculated_window_max_counts": float(fit_calculated[weak_peak_mask].max()),
        "window_residual_rmse_counts": float(
            np.sqrt(
                np.mean(
                    (fit_observed[weak_peak_mask] - fit_calculated[weak_peak_mask]) ** 2
                )
            )
        ),
        "window_point_count": int(weak_peak_mask.sum()),
    }
    selected_parameter_names = {
        "instrument.zero_shift",
        "instrument.profile.u",
        "instrument.profile.v",
        "instrument.profile.w",
        "instrument.profile.x",
        "instrument.profile.y",
    }
    selected_parameters = {
        row.path: {"value": row.value, "stderr": row.stderr, "vary": row.vary}
        for row in fit.parameters
        if row.path in selected_parameter_names
    }
    report_path = OUTPUT_DIR / f"{SAMPLE_ID}_lebail_summary.txt"
    report_path.write_text(
        "EXPLORATORY MULTIPHASE LE BAIL FIT\n"
        f"Sample: {SAMPLE_ID}\n"
        f"NOMAD archive: {archive_path.relative_to(ROOT)}\n"
        f"Measurement: {archive_data.get('diffraction_method_name')}\n"
        f"Omega (deg): {result_data.get('omega')}\n"
        f"2theta range (deg): {two_theta.min():.4f} to {two_theta.max():.4f}\n"
        f"Point count: {n_points}; step median (deg): {np.median(np.diff(two_theta)):.5f}\n"
        f"Cu Kalpha1 (A): {wavelength_kalpha1:.7f}\n"
        f"Cu Kalpha2 (A): {wavelength_kalpha2:.7f}; Kalpha2/Kalpha1: {ka2_ratio:.4f}\n"
        f"Geometry caveat: {geometry_note}\n\n"
        f"Phases: {', '.join(phase.name for phase in structure.phases)}\n"
        f"\n{fit}\n"
        f"\nResidual check around the 22.637° target reference line (window "
        f"{weak_peak_check['window_deg'][0]:.3f}–{weak_peak_check['window_deg'][1]:.3f}°): "
        f"observed max {weak_peak_check['observed_window_max_counts']:.2f} counts, "
        f"calculated max {weak_peak_check['calculated_window_max_counts']:.2f} counts; "
        f"peak positions observed/calculated {weak_peak_check['observed_peak_two_theta_deg']:.3f}/"
        f"{weak_peak_check['calculated_peak_two_theta_deg']:.3f}°, "
        f"window RMSE {weak_peak_check['window_residual_rmse_counts']:.2f} counts.\n"
        f"\nSelected fitted parameters: {json.dumps(selected_parameters, default=str)}\n"
        "\nUnit cells are held at the CIF values in this pass; only background, zero "
        "shift, and profile parameters are refined.\n"
        "\nNOTE: Le Bail intensities are extracted per reflection. This fit does not "
        "establish validated phase weight fractions; no quantitative phase fractions "
        "are reported from this trial.\n",
        encoding="utf-8",
    )
    metadata_path = OUTPUT_DIR / f"{SAMPLE_ID}_lebail_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "sample_id": SAMPLE_ID,
                "archive": str(archive_path.relative_to(ROOT)),
                "data_file": archive_data.get("data_file"),
                "measurement_geometry": archive_data.get("diffraction_method_name"),
                "omega": result_data.get("omega"),
                "source_peak_wavelength_A": wavelength_kalpha1,
                "kalpha_two_A": wavelength_kalpha2,
                "ratio_kalpha2_kalpha1": ka2_ratio,
                "point_count": n_points,
                "two_theta_range_deg": [float(two_theta.min()), float(two_theta.max())],
                "median_step_deg": float(np.median(np.diff(two_theta))),
                "geometry_used_by_rietx": "bragg_brentano",
                "geometry_limitation": geometry_note,
                "fit_mode": fit.mode,
                "cell_parameters_refined": False,
                "fit_statistics": fit.statistics.model_dump(mode="json"),
                "target_weak_reflection_check": weak_peak_check,
                "selected_fitted_parameters": selected_parameters,
                "fit_plot": str(figure_path.relative_to(ROOT)),
                "target_reflection_zoom_plot": str(peak_zoom_path.relative_to(ROOT)),
                "stages": [stage.model_dump(mode="json") for stage in fit.stages],
                "diagnostics": [_diagnostic_record(item) for item in fit.diagnostics],
                "cif_diagnostics": cif_diagnostics,
                "phase_names": [phase.name for phase in structure.phases],
                "weight_fractions_reported": False,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(fit)
    print(f"\nWrote pattern: {xy_path}")
    print(f"Wrote fit plot: {figure_path}")
    print(f"Wrote target reflection zoom: {peak_zoom_path}")
    print(f"Wrote fit summary: {report_path}")
    print(f"Wrote metadata: {metadata_path}")
    print(f"Caveat: {geometry_note}")


if __name__ == "__main__":
    main()
