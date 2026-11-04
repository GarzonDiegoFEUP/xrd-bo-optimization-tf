"""Reference-card parsing, phase classification, and film metrics."""

import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal
from striprtf.striprtf import rtf_to_text

from .config import (
    CONTACT_PHASES,
    IMPURITY_PHASES,
    MATCH_TOL,
    MAX_CANDIDATE_PEAKS,
    MIN_FWHM,
    PHASES,
    REFERENCE_CARD_FILES,
    REF_MIN_INTENSITY,
    REFERENCES_DIR,
    TARGET_PHASE,
)
from .peak_analysis import analyze_scan

_ICDD_ROW = re.compile(
    r"^\s*\d+\s+-?\d+\s+-?\d+\s+-?\d+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)"
)
_PIPE_ROW = re.compile(
    r"^\|\s*\d+\s*\|\s*([\d.]+)\s*\|\s*[\d.]+\s*\|\s*([\d.]+)\s*\|"
)


def card_peak_rows(text: str) -> list[tuple[float, float]]:
    """Extract (2-theta, relative intensity percent) pairs from reference tables."""
    rows = []
    for line in text.splitlines():
        match = _ICDD_ROW.match(line) or _PIPE_ROW.match(line)
        if match:
            rows.append((float(match.group(1)), float(match.group(2))))
    return rows


def parse_rtf_reference(
    rtf_file: Path, min_intensity: float = REF_MIN_INTENSITY
) -> list[float]:
    """Return sorted unique 2-theta positions above the card-intensity threshold."""
    text = rtf_to_text(rtf_file.read_text(encoding="utf-8", errors="replace"))
    return sorted(
        {
            theta
            for theta, intensity in card_peak_rows(text)
            if intensity >= min_intensity
        }
    )


def load_reference_cards(
    reference_dir: Path = REFERENCES_DIR,
) -> dict[str, list[float]]:
    """Load the target and impurity phase peak positions from project reference cards."""
    references = {}
    for phase, filename in REFERENCE_CARD_FILES.items():
        path = reference_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing {phase} reference card: {path}")
        reflections = parse_rtf_reference(path)
        if not reflections:
            raise ValueError(f"No reflections parsed from {phase} reference card: {path}")
        references[phase] = reflections
    return references


def classify_peaks(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    peak_indices: np.ndarray,
    references: dict[str, list[float]],
    tolerance: float = MATCH_TOL,
    min_fwhm: float = MIN_FWHM,
    max_candidate_peaks: int = MAX_CANDIDATE_PEAKS,
    contact_phases: tuple[str, ...] | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Match peaks as target, impurity, contact, or unmatched."""
    theta = np.asarray(two_theta, dtype=np.float64)
    values = np.asarray(intensity, dtype=np.float64)
    peaks = np.asarray(peak_indices, dtype=int)
    if theta.ndim != 1 or values.ndim != 1 or theta.size != values.size:
        raise ValueError(
            "two_theta and intensity must be one-dimensional arrays of equal length"
        )
    if peaks.ndim != 1 or np.any(peaks < 0) or np.any(peaks >= theta.size):
        raise ValueError("peak_indices must contain valid indices into the scan")
    if theta.size < 2 or np.any(np.diff(theta) <= 0):
        raise ValueError("two_theta values must be strictly increasing")
    missing_phases = set(PHASES).difference(references)
    if missing_phases:
        raise ValueError(
            f"Missing reference peaks for phases: {sorted(missing_phases)}"
        )
    active_contacts = CONTACT_PHASES if contact_phases is None else tuple(contact_phases)
    unknown_contacts = set(active_contacts).difference(CONTACT_PHASES)
    if unknown_contacts:
        raise ValueError(f"Unknown contact phases: {sorted(unknown_contacts)}")
    active_phases = (
        TARGET_PHASE,
        *IMPURITY_PHASES,
        *active_contacts,
    )
    if peaks.size == 0:
        return [], [], [], []

    widths, _, _, _ = signal.peak_widths(values, peaks, rel_height=0.5)
    step = float(np.mean(np.diff(theta)))
    width_by_index = dict(zip(peaks.tolist(), (widths * step).tolist()))
    candidates = sorted(peaks.tolist(), key=lambda index: -values[index])[
        :max_candidate_peaks
    ]
    accepted: list[dict[str, Any]] = []
    for index in candidates:
        width = float(width_by_index[index])
        if width < min_fwhm:
            continue
        detected_theta = float(theta[index])
        if any(
            abs(detected_theta - item["detected_theta"])
            < 0.5 * max(width, item["fwhm"])
            for item in accepted
        ):
            continue
        accepted.append(
            {
                "detected_theta": detected_theta,
                "fwhm": width,
                "intensity": float(values[index]),
            }
        )

    matches: dict[str, list[dict[str, Any]]] = {
        phase: [] for phase in (*PHASES, "Unmatched")
    }
    for peak in accepted:
        phase = next(
            (
                candidate
                for candidate in active_phases
                if any(
                    abs(peak["detected_theta"] - reference_theta) <= tolerance
                    for reference_theta in references[candidate]
                )
            ),
            "Unmatched",
        )
        matches[phase].append({**peak, "match_type": phase})

    target = matches[TARGET_PHASE]
    impurities = [peak for phase in IMPURITY_PHASES for peak in matches[phase]]
    contacts = [peak for phase in CONTACT_PHASES for peak in matches[phase]]
    return target, impurities, contacts, matches["Unmatched"]


def _window_mask(
    two_theta: np.ndarray, matches: list[dict[str, Any]], half_width: float
) -> np.ndarray:
    mask = np.zeros(len(two_theta), dtype=bool)
    for match in matches:
        mask |= np.abs(two_theta - match["detected_theta"]) <= half_width
    return mask


def calculate_phase_metrics(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    target: list[dict[str, Any]],
    impurities: list[dict[str, Any]],
    half_width: float = 0.5,
) -> dict[str, float | int | None]:
    """Calculate target-phase purity, primary-peak FWHM, and crystallinity."""
    theta = np.asarray(two_theta, dtype=np.float64)
    values = np.asarray(intensity, dtype=np.float64)
    target_mask = _window_mask(theta, target, half_width)
    impurity_mask = _window_mask(theta, impurities, half_width) & ~target_mask
    target_intensity = float(values[target_mask].sum())
    impurity_intensity = float(values[impurity_mask].sum())
    total_intensity = target_intensity + impurity_intensity
    primary = max(target, key=lambda match: match["intensity"]) if target else None
    primary_fwhm = float(primary["fwhm"]) if primary else None
    return {
        "phase_purity": (
            target_intensity / total_intensity if total_intensity > 0 else None
        ),
        "primary_fwhm": primary_fwhm,
        "crystallinity": 1.0 / primary_fwhm if primary_fwhm else None,
        "n_target_matches": len(target),
    }


def analyze_pattern(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    references: dict[str, list[float]],
    contact_phases: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Preprocess, classify, and compute XRD phase metrics for one film."""
    result = analyze_scan(two_theta, intensity)
    target, impurities, contacts, unmatched = classify_peaks(
        result["two_theta"],
        result["smooth"],
        result["detected_peaks"],
        references,
        contact_phases=contact_phases,
    )
    result.update(
        target=target,
        impurity=impurities,
        contact=contacts,
        unmatched=unmatched,
    )
    result.update(
        calculate_phase_metrics(
            result["two_theta"], result["bg_removed"], target, impurities
        )
    )
    return result


def build_peak_list(
    xrd_results: dict[str, dict[str, Any]],
    series_by_sample: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Return one row per classified detected peak for CSV export."""
    series_by_sample = series_by_sample or {}
    peak_groups = (
        ("target", "target"),
        ("impurity", "impurity"),
        ("contact", "contact"),
        ("unmatched", "unmatched"),
    )
    rows = []
    for sample_id, result in sorted(xrd_results.items()):
        for category, key in peak_groups:
            for peak in result.get(key, []):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "series": series_by_sample.get(sample_id),
                        "two_theta_deg": float(peak["detected_theta"]),
                        "fwhm_deg": float(peak["fwhm"]),
                        "detected_intensity": float(peak["intensity"]),
                        "match_category": category,
                        "phase": peak.get("match_type", "Unmatched"),
                    }
                )
    return sorted(rows, key=lambda row: (row["sample_id"], row["two_theta_deg"]))
