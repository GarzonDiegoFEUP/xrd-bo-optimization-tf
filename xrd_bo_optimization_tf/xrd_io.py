"""Loading of XRDML scans and XRD results stored in NOMAD archives."""

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from .config import NOMAD_VOLUME_DIR, RAW_DATA_DIR
from .nomad_client import iter_archive_files, read_archive


def load_xrdml(xrdml_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the 2-theta axis and counts from an XRDML scan."""
    root = ET.parse(xrdml_path).getroot()
    data_points = next(
        (element for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "dataPoints"),
        None,
    )
    if data_points is None:
        raise ValueError(f"XRDML scan has no dataPoints element: {xrdml_path}")

    theta_range = None
    counts = None
    for element in data_points:
        local_tag = element.tag.rsplit("}", 1)[-1]
        if local_tag == "positions" and element.get("axis") == "2Theta":
            positions = {
                child.tag.rsplit("}", 1)[-1]: child.text for child in element
            }
            if positions.get("startPosition") is not None and positions.get(
                "endPosition"
            ) is not None:
                theta_range = (
                    float(positions["startPosition"]),
                    float(positions["endPosition"]),
                )
        elif local_tag == "intensities" and element.text:
            counts = np.fromstring(element.text, sep=" ", dtype=np.float64)

    if theta_range is None or counts is None or counts.size == 0:
        raise ValueError(
            f"XRDML scan is missing its 2-theta range or intensities: {xrdml_path}"
        )

    two_theta = np.linspace(theta_range[0], theta_range[1], counts.size)
    return two_theta, counts


def _sample_id_from_filename(filename: str) -> str | None:
    match = re.match(r"(\d{6}[A-Za-z]\d+)", filename)
    return match.group(1) if match else None


def load_local_xrd_data(
    sample_ids: list[str] | tuple[str, ...] | set[str],
    directory: Path = RAW_DATA_DIR,
) -> dict[str, dict[str, Any]]:
    """Load local XRDML scans whose filenames begin with a requested sample ID."""
    requested = set(sample_ids)
    data: dict[str, dict[str, Any]] = {}
    if not directory.is_dir():
        return data

    for path in sorted(directory.glob("*.xrdml")):
        sample_id = _sample_id_from_filename(path.name)
        if sample_id not in requested or sample_id in data:
            continue
        two_theta, intensity = load_xrdml(path)
        data[sample_id] = {
            "two_theta": two_theta,
            "intensity": intensity,
            "source": path.name,
        }
    return data


def load_nomad_xrd_data(
    sample_ids: list[str] | tuple[str, ...] | set[str],
    archive_root: Path = NOMAD_VOLUME_DIR,
) -> dict[str, dict[str, Any]]:
    """Load XRD result arrays from ELNXRayDiffraction archives in a NOMAD export."""
    requested = set(sample_ids)
    data: dict[str, dict[str, Any]] = {}
    for archive_path in iter_archive_files(archive_root):
        archive_data = read_archive(archive_path).get("data", {})
        if not isinstance(archive_data, dict) or "ELNXRayDiffraction" not in str(
            archive_data.get("m_def", "")
        ):
            continue
        sample_id = str(archive_data.get("name", "")).split()
        sample_id = sample_id[0] if sample_id else None
        if sample_id not in requested or sample_id in data:
            continue
        results = archive_data.get("results", [])
        if (
            not isinstance(results, list)
            or not results
            or not isinstance(results[0], dict)
        ):
            continue
        result = results[0]
        theta = result.get("two_theta")
        intensity = result.get("intensity")
        if theta is None or intensity is None:
            continue
        count = min(len(theta), len(intensity))
        if count == 0:
            continue
        data[sample_id] = {
            "two_theta": np.asarray(theta[:count], dtype=np.float64),
            "intensity": np.asarray(intensity[:count], dtype=np.float64),
            "source": "NOMAD",
        }
    return data


def load_xrd_data(
    sample_ids: list[str] | tuple[str, ...] | set[str],
    source: str = "auto",
    local_directory: Path = RAW_DATA_DIR,
    nomad_archive_root: Path = NOMAD_VOLUME_DIR,
) -> dict[str, dict[str, Any]]:
    """Load scans locally, from NOMAD archives, or prefer NOMAD with local fallback."""
    if source not in {"auto", "local", "nomad"}:
        raise ValueError("source must be 'auto', 'local', or 'nomad'")

    requested = set(sample_ids)
    data: dict[str, dict[str, Any]] = {}
    if source in {"auto", "nomad"} and nomad_archive_root.is_dir():
        data.update(load_nomad_xrd_data(requested, nomad_archive_root))
    if source in {"auto", "local"}:
        local_data = load_local_xrd_data(requested, local_directory)
        for sample_id, record in local_data.items():
            data.setdefault(sample_id, record)
    return data
