"""NOMAD Oasis REST API client for BaCuSe data extraction.

Reads credentials from the repository-root .env file (NOMAD_API_URL, NOMAD_PAT).
The local archive helpers in nomad_client.py remain the offline fallback; this
module adds the network path described in docs/NOMAD_HANDOFF.md.
"""

import io
import os
import zipfile
from pathlib import Path
from typing import Any, Iterator

import requests
from dotenv import load_dotenv

from .config import PROJECT_ROOT

DEFAULT_API_URL = "https://nomad-sadewasser.inl.local/nomad-oasis/api/v1"
REQUEST_TIMEOUT = 30

BACUSE_ELEMENTS = ("Ba", "Cu", "Se")
XRDML_SUFFIX = ".xrdml"


class NomadAPIError(RuntimeError):
    """Raised when the NOMAD Oasis API returns an unusable response."""


def load_credentials(env_path: Path | None = None) -> tuple[str, str]:
    """Load NOMAD_API_URL and NOMAD_PAT from .env and return (api_url, token)."""
    load_dotenv(env_path or PROJECT_ROOT / ".env")
    api_url = os.getenv("NOMAD_API_URL", DEFAULT_API_URL).rstrip("/")
    token = os.getenv("NOMAD_PAT", "").strip()
    if not token:
        raise NomadAPIError(
            "NOMAD_PAT is empty. Create a token in the Oasis GUI "
            "(Profile > Settings > Tokens) and paste it into .env."
        )
    return api_url, token


def _verify_ssl() -> bool:
    """Return False when NOMAD_VERIFY_SSL=0 (Oasis uses a self-signed cert)."""
    verify = os.getenv("NOMAD_VERIFY_SSL", "0").strip().lower() not in ("0", "false", "no")
    if not verify:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return verify


def make_session(token: str | None = None) -> requests.Session:
    """Build an authenticated session honoring NOMAD_VERIFY_SSL."""
    if token is None:
        _, token = load_credentials()
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/json"})
    session.verify = _verify_ssl()
    if not session.verify:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return session


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }


def query_entries(
    query: dict[str, Any],
    include: list[str],
    api_url: str | None = None,
    token: str | None = None,
    page_size: int = 100,
) -> Iterator[dict[str, Any]]:
    """Yield entries matching a query, following cursor-based pagination."""
    if api_url is None or token is None:
        api_url, token = load_credentials()

    request_body: dict[str, Any] = {
        "owner": "visible",
        "query": query,
        "pagination": {"page_size": page_size},
        "required": {"include": include},
    }

    while True:
        response = requests.post(
            f"{api_url}/entries/query",
            headers=_headers(token),
            json=request_body,
            timeout=REQUEST_TIMEOUT,
            verify=_verify_ssl(),
        )
        if response.status_code == 401:
            raise NomadAPIError("HTTP 401: token missing, expired, or revoked.")
        if response.status_code == 403:
            detail = response.json().get("detail", "")
            raise NomadAPIError(f"HTTP 403: insufficient token scopes. {detail}")
        response.raise_for_status()
        payload = response.json()

        yield from payload.get("data", [])

        next_value = payload.get("pagination", {}).get("next_page_after_value")
        if not next_value:
            break
        request_body["pagination"]["page_after_value"] = next_value


def query_bacuse_entries(
    api_url: str | None = None,
    token: str | None = None,
) -> list[dict[str, Any]]:
    """Return all entries whose normalized material contains Ba, Cu, and Se."""
    include = [
        "entry_id",
        "upload_id",
        "entry_name",
        "mainfile_key",
        "entry_type",
        "results.material.chemical_formula_hill",
        "results.material.elements",
    ]
    entries = list(
        query_entries(
            query={"results.material.elements": {"all": list(BACUSE_ELEMENTS)}},
            include=include,
            api_url=api_url,
            token=token,
        )
    )
    return entries


def fetch_entry_archive(
    entry_id: str,
    api_url: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Fetch the processed archive JSON for one entry."""
    if api_url is None or token is None:
        api_url, token = load_credentials()

    response = requests.get(
        f"{api_url}/entries/{entry_id}/archive",
        headers=_headers(token),
        timeout=REQUEST_TIMEOUT,
        verify=_verify_ssl(),
    )
    if response.status_code == 404:
        raise NomadAPIError(f"HTTP 404: no archive for entry {entry_id}.")
    response.raise_for_status()
    payload = response.json()
    archive = payload.get("data", {}).get("archive", payload.get("data", payload))
    if not isinstance(archive, dict):
        raise NomadAPIError(f"Unexpected archive payload for entry {entry_id}.")
    return archive


def download_raw_query_zip(
    query: dict[str, Any],
    api_url: str | None = None,
    token: str | None = None,
) -> bytes:
    """Download raw files for all entries matching a query as ZIP bytes."""
    if api_url is None or token is None:
        api_url, token = load_credentials()

    response = requests.post(
        f"{api_url}/entries/raw/query",
        headers=_headers(token),
        json={"query": query},
        timeout=120,
        verify=_verify_ssl(),
    )
    response.raise_for_status()
    return response.content


def download_upload_raw_file(
    upload_id: str,
    path: str,
    api_url: str | None = None,
    token: str | None = None,
) -> bytes:
    """Download one raw file from an upload by its path inside the upload."""
    if api_url is None or token is None:
        api_url, token = load_credentials()

    response = requests.get(
        f"{api_url}/uploads/{upload_id}/raw/{path}",
        headers=_headers(token),
        timeout=60,
        verify=_verify_ssl(),
    )
    response.raise_for_status()
    return response.content


def save_xrdml_from_zip(zip_bytes: bytes, output_dir: Path) -> list[Path]:
    """Extract .xrdml files from a raw-query ZIP into output_dir; return paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive_zip:
        for member in archive_zip.namelist():
            if not member.lower().endswith(XRDML_SUFFIX):
                continue
            target = output_dir / Path(member).name
            target.write_bytes(archive_zip.read(member))
            saved.append(target)
    return saved


# --- Upload-ID-driven extraction (primary path, per docs/NOMAD_HANDOFF.md) ---


def load_upload_ids(ids_path: Path | None = None) -> list[str]:
    """Read BaCuSe upload IDs from data/processed/bacuse_upload_ids.txt."""
    path = ids_path or (PROJECT_ROOT / "data" / "processed" / "bacuse_upload_ids.txt")
    if not path.is_file():
        raise NomadAPIError(f"Upload-ID list not found: {path}")
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        # Strip inline comments and whitespace; keep only the bare upload_id.
        token_line = line.split("#", 1)[0].strip()
        if token_line:
            ids.append(token_line.split()[0])
    if not ids:
        raise NomadAPIError(f"No upload IDs in {path}. Paste the BaCuSe upload IDs there.")
    return ids


def list_upload_entries(
    upload_id: str,
    api_url: str | None = None,
    token: str | None = None,
) -> list[dict[str, Any]]:
    """List all entries inside one upload (id, name, mainfile, type)."""
    include = ["entry_id", "upload_id", "entry_name", "mainfile_key", "entry_type"]
    return list(
        query_entries(
            query={"upload_id": upload_id},
            include=include,
            api_url=api_url,
            token=token,
        )
    )


def download_upload_archives(
    upload_id: str,
    output_dir: Path,
    api_url: str | None = None,
    token: str | None = None,
) -> list[Path]:
    """Download every entry archive in an upload as *.archive.json files."""
    if api_url is None or token is None:
        api_url, token = load_credentials()
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for entry in list_upload_entries(upload_id, api_url, token):
        entry_id = entry.get("entry_id")
        if not entry_id:
            continue
        archive = fetch_entry_archive(entry_id, api_url, token)
        name = (entry.get("entry_name") or entry_id).replace("/", "_")
        target = output_dir / upload_id / f"{name}.archive.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        import json

        target.write_text(json.dumps(archive, indent=2), encoding="utf-8")
        saved.append(target)
    return saved


def list_upload_raw_files(
    upload_id: str,
    api_url: str | None = None,
    token: str | None = None,
) -> list[dict[str, Any]]:
    """List raw files in an upload via the rawdir endpoint."""
    if api_url is None or token is None:
        api_url, token = load_credentials()

    response = requests.get(
        f"{api_url}/uploads/{upload_id}/rawdir/",
        headers=_headers(token),
        timeout=REQUEST_TIMEOUT,
        verify=_verify_ssl(),
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("data", []) if isinstance(payload, dict) else []


def download_upload_xrdml(
    upload_id: str,
    output_dir: Path,
    api_url: str | None = None,
    token: str | None = None,
) -> list[Path]:
    """Download all .xrdml raw files from an upload into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for raw_file in list_upload_raw_files(upload_id, api_url, token):
        path = raw_file.get("path", "") if isinstance(raw_file, dict) else ""
        if not path.lower().endswith(XRDML_SUFFIX):
            continue
        content = download_upload_raw_file(upload_id, path, api_url, token)
        target = output_dir / Path(path).name
        target.write_bytes(content)
        saved.append(target)
    return saved
