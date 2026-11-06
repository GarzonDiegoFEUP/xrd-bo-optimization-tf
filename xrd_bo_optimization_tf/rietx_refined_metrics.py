"""Export the standard four-phase exploratory Le Bail summary as a metrics CSV.

Run with ``python -m xrd_bo_optimization_tf.rietx_refined_metrics``. Blank
numeric cells mean unavailable, not zero. Rwp is copied in the source's
dimensionless fraction convention, not converted to percent. The source
workflow labels the measurements GIXRD and uses a Bragg-Brentano proxy; its
summary does not identify individual NOMAD archives.

This exporter deliberately does not derive quantitative phase fractions from
Le Bail intensities or microstructure from generic peak/profile widths. It
validates the entire input before writing and never silently discards rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs/rietx_lebail_all_samples/lebail_summary_all_samples.csv"
DEFAULT_OUTPUT = ROOT / "data/results/bacuse2_rietx_refined_metrics.csv"
PHASES = frozenset(("α-BaCu₂Se₂", "β-BaCu₂Se₂", "BaSeO₃", "BaSe"))
REQUIRED_COLUMNS = (
    "sample_id", "status", "rwp", "omega_deg", "phase_count", "phase_labels",
    "diagnostic_codes", "fit_plot", "warning_geometry",
)
OUTPUT_COLUMNS = (
    "sample_id", "status", "rwp", "omega_deg", "phase_count", "phase_labels",
    "diagnostic_codes", "measurement_geometry", "fit_geometry_warning", "fit_mode",
    "fit_plot", "source_summary", "source_summary_sha256",
    "alpha_weight_fraction", "alpha_weight_fraction_availability_reason",
    "alpha_crystallite_size_nm", "alpha_crystallite_size_availability_reason",
    "alpha_microstrain", "alpha_microstrain_availability_reason",
)
WEIGHT_REASON = "Le Bail reflection intensities do not provide quantitative phase weight fractions."
SIZE_REASON = "No validated phase-specific crystallite size is reported in the source summary."
STRAIN_REASON = "No validated phase-specific microstrain is reported in the source summary."


def _validate_number(value: str, column: str, sample: str) -> None:
    """Allow missing observations but reject malformed or nonphysical numbers."""
    if not value.strip():
        return
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"{sample}: invalid {column}: {value!r}") from exc
    if not math.isfinite(number) or (column == "rwp" and number < 0):
        raise ValueError(f"{sample}: invalid {column}: {value!r}")


def _source_name(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def export_refined_metrics(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT) -> int:
    """Validate and export every summary row in source order; return row count.

    Inputs must use the standard summary schema and, when supplied, the
    standard four-phase set. Missing observations remain blank, including for
    failed fits; malformed values and duplicate/missing IDs are errors rather
    than reasons to drop rows. Diagnostic codes are copied verbatim.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must differ")
    source_bytes = input_path.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    # Decode and parse the exact bytes that were hashed for provenance.
    reader = csv.DictReader(io.StringIO(source_bytes.decode("utf-8-sig"), newline=""))
    headers = reader.fieldnames or []
    if len(set(headers)) != len(headers):
        raise ValueError("Duplicate input column names")
    missing = set(REQUIRED_COLUMNS) - set(headers)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    rows = []
    seen = set()
    for line, row in enumerate(reader, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Malformed CSV row at record {line}")
        sample = row["sample_id"].strip()
        if not sample:
            raise ValueError(f"Missing sample_id at record {line}")
        if sample in seen:
            raise ValueError(f"Duplicate sample_id: {sample}")
        seen.add(sample)
        if not row["status"].strip():
            raise ValueError(f"{sample}: missing status")
        for column in ("rwp", "omega_deg"):
            _validate_number(row[column], column, sample)
        if row["phase_count"].strip() and row["phase_count"].strip() != "4":
            raise ValueError(f"{sample}: expected four phases, got {row['phase_count']!r}")
        labels = row["phase_labels"].strip()
        if labels:
            parts = [part.strip() for part in labels.split(";")]
            if len(parts) != 4 or set(parts) != PHASES:
                raise ValueError(f"{sample}: unexpected four-phase labels: {labels!r}")
        rows.append({
            "sample_id": sample,
            "status": row["status"],
            "rwp": row["rwp"],
            "omega_deg": row["omega_deg"],
            "phase_count": row["phase_count"],
            "phase_labels": row["phase_labels"],
            "diagnostic_codes": row["diagnostic_codes"],
            "measurement_geometry": "GIXRD (source workflow label)",
            "fit_geometry_warning": row["warning_geometry"],
            "fit_mode": "Le Bail (exploratory; fixed cells)",
            "fit_plot": row["fit_plot"],
            "source_summary": _source_name(input_path),
            "source_summary_sha256": digest,
            "alpha_weight_fraction": "",
            "alpha_weight_fraction_availability_reason": WEIGHT_REASON,
            "alpha_crystallite_size_nm": "",
            "alpha_crystallite_size_availability_reason": SIZE_REASON,
            "alpha_microstrain": "",
            "alpha_microstrain_availability_reason": STRAIN_REASON,
        })
    if not rows:
        raise ValueError("Input summary contains no rows")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically so validation or an interrupted write cannot leave a partial CSV.
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=output_path.parent,
            prefix=f".{output_path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Four-phase Le Bail summary CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination CSV")
    args = parser.parse_args()
    count = export_refined_metrics(args.input, args.output)
    print(f"Exported {count} rows from {args.input} to {args.output}")


if __name__ == "__main__":
    main()