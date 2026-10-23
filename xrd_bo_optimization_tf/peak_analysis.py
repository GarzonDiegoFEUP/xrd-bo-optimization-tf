"""XRD signal cleaning, smoothing, and peak detection."""

from typing import Any

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter


def remove_spikes(
    intensity: np.ndarray, kernel_size: int = 3, threshold_factor: float = 2.0
) -> tuple[np.ndarray, int]:
    """Replace isolated positive spikes with a local median."""
    values = np.asarray(intensity, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("intensity must be a one-dimensional array")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if values.size < kernel_size or values.size == 0:
        return values.copy(), 0

    local_median = median_filter(values, size=kernel_size)
    difference = values - local_median
    positive = difference[difference > 0]
    if positive.size == 0:
        return values.copy(), 0
    q75, q25 = np.percentile(positive, [75, 25])
    spikes = difference > q75 + threshold_factor * (q75 - q25)
    cleaned = values.copy()
    cleaned[spikes] = local_median[spikes]
    return cleaned, int(spikes.sum())


def remove_background(intensity: np.ndarray, kernel_size: int = 51) -> tuple[np.ndarray, np.ndarray]:
    """Subtract half of a median-filtered baseline and clip negative counts."""
    values = np.asarray(intensity, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("intensity must be a one-dimensional array")
    if values.size == 0:
        return values.copy(), values.copy()
    size = min(kernel_size, values.size if values.size % 2 else values.size - 1)
    if size < 1:
        return np.clip(values, 0, None), values.copy()
    background = signal.medfilt(values, kernel_size=size)
    return np.clip(values - 0.5 * background, 0, None), background


def detect_peaks(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    smoothing_window_degrees: float = 0.3,
    peak_height_fraction: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Clean and smooth a scan, then return its detected peak indices and arrays."""
    theta = np.asarray(two_theta, dtype=np.float64)
    raw = np.asarray(intensity, dtype=np.float64)
    if theta.ndim != 1 or raw.ndim != 1 or theta.size != raw.size:
        raise ValueError("two_theta and intensity must be one-dimensional arrays of equal length")
    if theta.size < 3 or not np.all(np.isfinite(theta)) or not np.all(np.isfinite(raw)):
        raise ValueError("a scan must contain at least three finite points")
    increments = np.diff(theta)
    if np.any(increments <= 0):
        raise ValueError("two_theta values must be strictly increasing")

    cleaned, spike_count = remove_spikes(raw)
    background_removed, _ = remove_background(cleaned)
    step = float(np.mean(increments))
    window_length = int(round(smoothing_window_degrees / step)) | 1
    window_length = min(window_length, theta.size if theta.size % 2 else theta.size - 1)
    if window_length >= 5:
        smooth = signal.savgol_filter(
            background_removed, window_length=window_length, polyorder=3
        )
    else:
        smooth = background_removed.copy()

    residual = background_removed - smooth
    noise = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    max_intensity = float(np.max(smooth))
    if max_intensity <= 0:
        peaks = np.array([], dtype=int)
    else:
        peaks, _ = signal.find_peaks(
            smooth,
            height=peak_height_fraction * max_intensity,
            prominence=max(peak_height_fraction * max_intensity, 4 * noise),
            distance=max(1, int(round(0.2 / step))),
        )
    return cleaned, background_removed, smooth, peaks, spike_count


def analyze_scan(two_theta: np.ndarray, intensity: np.ndarray) -> dict[str, Any]:
    """Run the reusable preprocessing and peak-detection steps for one scan."""
    cleaned, background_removed, smooth, peaks, spike_count = detect_peaks(
        two_theta, intensity
    )
    return {
        "two_theta": np.asarray(two_theta, dtype=np.float64),
        "raw": np.asarray(intensity, dtype=np.float64),
        "cleaned": cleaned,
        "bg_removed": background_removed,
        "smooth": smooth,
        "detected_peaks": peaks,
        "n_spikes": spike_count,
        "n_detected_peaks": len(peaks),
    }
