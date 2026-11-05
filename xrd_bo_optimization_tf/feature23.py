"""Descriptors and purity scenarios for the unassigned ~23 deg XRD feature.

The feature lies on weak alpha (011) and beta (101) BaCu2Se2 reflections and is
not assigned in the production objective. The scenarios here only recompute the
metrics under an assumed treatment; they never change the production results.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    ALPHA_294_AREA_WINDOW,
    FEATURE_23_AREA_WINDOW,
    FEATURE_23_PEAK_RANGE,
    IN_CONTACT_ALLOWED_SERIES,
    NOMAD_STAGING_DIR,
    TARGET_PHASE,
)
from .intensity_diagnostics import (
    detected_peak_area_fractions,
    integrate_window_area,
)
from .phase_matching import (
    analyze_pattern,
    calculate_phase_metrics,
    load_reference_cards,
)
from .xrd_io import load_xrd_data

# as_is reproduces production: the feature stays unmatched and penalized.
FEATURE_SCENARIOS = ("as_is", "excluded", "as_target", "as_impurity")
FEATURE_IMPURITY_PHASE = "BaCu2Se2_beta"
_PEAK_KEYS = ("target", "impurity", "unmatched", "contact")
_METRIC_COLUMNS = ("phase_purity", "crystallinity", "primary_fwhm", "n_target_matches")


def _in_range(peak: Mapping[str, Any], peak_range: tuple[float, float]) -> bool:
    return peak_range[0] <= float(peak["detected_theta"]) <= peak_range[1]


def analyze_scans(
    synthesis: pd.DataFrame, nomad_root=NOMAD_STAGING_DIR
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load NOMAD scans for the synthesis samples and run the production analysis."""
    references = load_reference_cards()
    series_by_sample = (
        synthesis.drop_duplicates("sample_id")
        .set_index("sample_id")["series"]
        .astype(str)
        .to_dict()
    )
    xrd_raw = load_xrd_data(
        synthesis["sample_id"].astype(str).tolist(),
        source="nomad",
        nomad_archive_root=nomad_root,
    )
    xrd_results = {
        sample_id: analyze_pattern(
            scan["two_theta"],
            scan["intensity"],
            references,
            contact_phases=(
                ("Au", "In")
                if series_by_sample.get(sample_id) in IN_CONTACT_ALLOWED_SERIES
                else ("Au",)
            ),
        )
        for sample_id, scan in xrd_raw.items()
    }
    return xrd_raw, xrd_results


def reclassify_feature(
    result: Mapping[str, Any],
    scenario: str,
    peak_range: tuple[float, float] = FEATURE_23_PEAK_RANGE,
) -> dict[str, float | int | None]:
    """Recompute purity, crystallinity and the unassigned-area fraction for a scenario.

    Every detected peak inside ``peak_range`` is removed from its current class and
    then dropped (excluded from purity and the penalty), counted as target, or
    counted as the beta impurity.
    """
    if scenario not in FEATURE_SCENARIOS:
        raise ValueError(f"scenario must be one of {FEATURE_SCENARIOS}")
    target = list(result["target"])
    impurity = list(result["impurity"])
    unmatched = list(result["unmatched"])
    if scenario != "as_is":
        moved = [p for p in (*target, *impurity, *unmatched) if _in_range(p, peak_range)]
        target = [p for p in target if not _in_range(p, peak_range)]
        impurity = [p for p in impurity if not _in_range(p, peak_range)]
        unmatched = [p for p in unmatched if not _in_range(p, peak_range)]
        if scenario == "as_target":
            target += [{**p, "match_type": TARGET_PHASE} for p in moved]
        elif scenario == "as_impurity":
            impurity += [{**p, "match_type": FEATURE_IMPURITY_PHASE} for p in moved]
    analysis = {**result, "target": target, "impurity": impurity, "unmatched": unmatched}
    metrics = calculate_phase_metrics(
        result["two_theta"], result["bg_removed"], target, impurity
    )
    area = detected_peak_area_fractions(analysis, result["two_theta"])
    return {
        **metrics,
        "unassigned_or_overlap_area_fraction": area["unassigned_or_overlap_area_fraction"],
    }


def feature_scenario_tables(
    xrd_results: Mapping[str, Mapping[str, Any]], metrics: pd.DataFrame
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Return (metrics, diagnostics) per scenario, ready for prepare_optimization_data."""
    tables = {}
    for scenario in FEATURE_SCENARIOS:
        computed = {
            sample_id: reclassify_feature(result, scenario)
            for sample_id, result in xrd_results.items()
        }
        updated = metrics.copy()
        for column in _METRIC_COLUMNS:
            values = []
            for sample_id, old in zip(updated["sample_id"], updated[column]):
                if sample_id not in computed:
                    values.append(old)
                else:
                    new = computed[sample_id][column]
                    values.append(np.nan if new is None else new)
            updated[column] = values
        diagnostics = pd.DataFrame(
            {
                "sample_id": list(computed),
                "unassigned_or_overlap_area_fraction": [
                    row["unassigned_or_overlap_area_fraction"] for row in computed.values()
                ],
            }
        )
        tables[scenario] = (updated, diagnostics)
    return tables


def feature_descriptors(
    analysis: Mapping[str, Any],
    two_theta: np.ndarray,
    intensity: np.ndarray,
    peak_range: tuple[float, float] = FEATURE_23_PEAK_RANGE,
) -> dict[str, float]:
    """Position, width, area and ratio to the alpha 29.4 deg cluster for one scan."""
    peaks = [
        peak
        for key in _PEAK_KEYS
        for peak in analysis.get(key, [])
        if _in_range(peak, peak_range)
    ]
    main = max(peaks, key=lambda peak: peak["intensity"]) if peaks else None

    def window_area(window) -> float:
        try:
            return integrate_window_area(two_theta, intensity, *window)["area_counts_deg"]
        except ValueError:
            return np.nan

    feature_area = window_area(FEATURE_23_AREA_WINDOW)
    reference_area = window_area(ALPHA_294_AREA_WINDOW)
    return {
        "feature23_two_theta": float(main["detected_theta"]) if main else np.nan,
        "feature23_fwhm": float(main["fwhm"]) if main else np.nan,
        "feature23_peak_counts": float(main["intensity"]) if main else np.nan,
        "feature23_area_counts_deg": feature_area,
        "alpha294_area_counts_deg": reference_area,
        "feature23_to_alpha294_area_ratio": (
            feature_area / reference_area if reference_area > 0 else np.nan
        ),
    }


def build_feature_descriptors(
    xrd_raw: Mapping[str, Mapping[str, Any]],
    xrd_results: Mapping[str, Mapping[str, Any]],
    series_by_sample: Mapping[str, str],
) -> pd.DataFrame:
    """Build one descriptor row per scan."""
    rows = []
    for sample_id in sorted(set(xrd_raw).intersection(xrd_results)):
        scan = xrd_raw[sample_id]
        rows.append(
            {
                "sample_id": sample_id,
                "series": series_by_sample.get(sample_id),
                **feature_descriptors(
                    xrd_results[sample_id],
                    np.asarray(scan["two_theta"], dtype=np.float64),
                    np.asarray(scan["intensity"], dtype=np.float64),
                ),
            }
        )
    return pd.DataFrame(rows)
