"""Local access helpers for NOMAD archive exports."""

import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def iter_archive_files(root: Path, pattern: str = "*.archive.json") -> Iterator[Path]:
    """Yield archive JSON files under a NOMAD volume or exported directory."""
    if not root.is_dir():
        return
    yield from sorted(root.rglob(pattern))


def read_archive(path: Path) -> dict[str, Any]:
    """Read one NOMAD archive JSON file and return its root object."""
    with path.open(encoding="utf-8") as archive_file:
        archive = json.load(archive_file)
    if not isinstance(archive, dict):
        raise ValueError(f"Expected a JSON object in NOMAD archive: {path}")
    return archive


def discover_bacuse_uploads(
    root: Path,
    keywords: tuple[str, ...] = ("bacuse", "bacu2se2", "cu2se"),
) -> dict[str, dict[str, Any]]:
    """Find NOMAD upload folders containing archive names that mention BaCuSe."""
    uploads: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "sample_names": set()}
    )

    for archive_path in iter_archive_files(root):
        archive_data = read_archive(archive_path).get("data", {})
        if not isinstance(archive_data, dict):
            continue

        relative_parts = archive_path.relative_to(root).parts
        if len(relative_parts) < 3:
            continue
        upload_id = "/".join(relative_parts[:2])
        samples = archive_data.get("samples", [])
        if not isinstance(samples, list):
            samples = []
        sample_names = [
            sample.get("name", "")
            for sample in samples
            if isinstance(sample, dict)
        ]
        searchable_text = " ".join(
            [str(archive_data.get("name", "")), *map(str, sample_names)]
        ).casefold()
        if any(keyword.casefold() in searchable_text for keyword in keywords):
            uploads[upload_id]["count"] += 1
            uploads[upload_id]["sample_names"].update(
                name for name in sample_names if name
            )

    return {
        upload_id: {
            "count": info["count"],
            "sample_names": sorted(info["sample_names"]),
        }
        for upload_id, info in sorted(uploads.items())
    }
