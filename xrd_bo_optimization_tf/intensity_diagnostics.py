"""Exploratory XRD peak-window and unassigned-peak intensity diagnostics.

These metrics are screening diagnostics, not quantitative phase fractions. Fixed
windows can contain overlapping reflections, especially in the 22.2-23.4° and
26.85-27.35° regions for this project.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def integrate_window_area(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    bounds: tuple[float, float],
    sidebands: tuple[tuple[float, float], tuple[float, float]],
) -> dict[str, float | int]:
    """Integrate positive signal above a fitted local linear baseline.

    The baseline is fitted by least squares to two user-selected sidebands, then
    subtracted over the integration window. This does not resolve overlapping
    peaks; the caller must label such regions as mixed/ambiguous.
    """
    theta = np.asarray(two_theta, dtype=np.float64)
    values = np.asarray(intensity, dtype=np.float64)
    if theta.ndim != 1 or values.ndim != 1 or theta.size != values.size:
        raise ValueError("two_theta and intensity must be 1D arrays of equal length")
    if theta.size < 3 or np.any(np.diff(theta) <= 0):
        raise ValueError("two_theta must contain at least three increasing values")
    lo, hi = map(float, bounds)
    if lo >= hi:
        raise ValueError("bounds must be an increasing (low, high) pair")
    window = (theta >= lo) & (theta <= hi)
    background = np.zeros(theta.size, dtype=bool)
    for side_lo, side_hi in sidebands:
        side_lo, side_hi = float(side_lo), float(side_hi)
        if side_lo >= side_hi or not (side_hi <= lo or side_lo >= hi):
            raise ValueError("sidebands must be increasing and outside the peak window")
        background |= (theta >= side_lo) & (theta <= side_hi)
    if window.sum() < 3 or background.sum() < 4:
        raise ValueError("integration window or sidebands contain too few data points")

    slope, intercept = np.polyfit(theta[background], values[background], deg=1)
    baseline = slope * theta[window] + intercept
    net = np.clip(values[window] - baseline, 0.0, None)
    area = float(np.trapezoid(net, theta[window]))
    return {
        "area_counts_deg": area,
        "max_net_counts": float(net.max()),
        "baseline_counts_median": float(np.median(baseline)),
        "n_points": int(window.sum()),
    }


def detected_peak_area_fractions(
    analysis: Mapping[str, Any],
    two_theta: np.ndarray,
    step_degrees: float | None = None,
) -> dict[str, float | int]:
    """Estimate matched, unmatched, and overlapping detected-peak area shares.

    Peak intervals use their measured FWHM. Any interval containing both an
    unmatched and matched peak is classified as overlap/ambiguous. Areas use the
    existing nonnegative background-removed trace and are intentionally labeled
    diagnostic proxies, not an absolute intensity balance.
    """
    theta = np.asarray(two_theta, dtype=np.float64)
    values = np.asarray(analysis.get("bg_removed", []), dtype=np.float64)
    if theta.ndim != 1 or values.ndim != 1 or theta.size != values.size:
        raise ValueError("analysis bg_removed and two_theta must have equal 1D shape")
    if theta.size < 2 or np.any(np.diff(theta) <= 0):
        raise ValueError("two_theta must be strictly increasing")
    if step_degrees is None:
        step_degrees = float(np.median(np.diff(theta)))
    if step_degrees <= 0:
        raise ValueError("step_degrees must be positive")

    matched_mask = np.zeros(theta.size, dtype=bool)
    unmatched_mask = np.zeros(theta.size, dtype=bool)
    matched_peaks = [*analysis.get("target", []), *analysis.get("impurity", [])]
    unmatched_peaks = list(analysis.get("unmatched", []))

    def add_intervals(mask: np.ndarray, peaks: Sequence[Mapping[str, Any]]) -> None:
        for peak in peaks:
            center = float(peak["detected_theta"])
            fwhm = float(peak.get("fwhm", 0.0))
            half_width = max(fwhm / 2.0, 2.0 * step_degrees)
            mask |= np.abs(theta - center) <= half_width

    add_intervals(matched_mask, matched_peaks)
    add_intervals(unmatched_mask, unmatched_peaks)
    overlap_mask = matched_mask & unmatched_mask
    matched_only_mask = matched_mask & ~unmatched_mask
    unmatched_only_mask = unmatched_mask & ~matched_mask

    def area(mask: np.ndarray) -> float:
        indices = np.flatnonzero(mask)
        if indices.size < 2:
            return 0.0
        breaks = np.flatnonzero(np.diff(indices) > 1) + 1
        segments = np.split(indices, breaks)
        return float(
            sum(
                np.trapezoid(values[segment], theta[segment])
                for segment in segments
                if segment.size >= 2
            )
        )

    matched_area = area(matched_only_mask)
    unmatched_area = area(unmatched_only_mask)
    overlap_area = area(overlap_mask)
    denominator = matched_area + unmatched_area + overlap_area
    return {
        "n_matched_peaks": len(matched_peaks),
        "n_unmatched_peaks": len(unmatched_peaks),
        "matched_detected_area": matched_area,
        "unmatched_detected_area": unmatched_area,
        "overlap_ambiguous_detected_area": overlap_area,
        "unassigned_or_overlap_area_fraction": (
            (unmatched_area + overlap_area) / denominator if denominator > 0 else np.nan
        ),
    }


def build_peak_area_diagnostics(
    xrd_raw: Mapping[str, Mapping[str, Any]],
    xrd_results: Mapping[str, Mapping[str, Any]],
    series_by_sample: Mapping[str, str],
    windows: Mapping[
        str,
        tuple[
            tuple[float, float],
            tuple[tuple[float, float], tuple[float, float]],
        ],
    ],
) -> list[dict[str, Any]]:
    """Build per-sample local-background-subtracted areas for named windows."""
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(set(xrd_raw).intersection(xrd_results)):
        scan = xrd_raw[sample_id]
        theta = np.asarray(scan["two_theta"], dtype=np.float64)
        intensity = np.asarray(scan["intensity"], dtype=np.float64)
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "series": series_by_sample.get(sample_id),
            "point_count": int(theta.size),
            "median_step_deg": float(np.median(np.diff(theta))),
        }
        for name, (bounds, sidebands) in windows.items():
            measurement = integrate_window_area(theta, intensity, bounds, sidebands)
            row[f"{name}_area_counts_deg"] = measurement["area_counts_deg"]
            row[f"{name}_max_net_counts"] = measurement["max_net_counts"]
        row.update(
            detected_peak_area_fractions(xrd_results[sample_id], theta)
        )
        rows.append(row)
    return rows
