"""Download the public Figshare EEG/BIS dataset without requiring credentials."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

from ..config import FIGSHARE_API_URL


@dataclass(frozen=True)
class DataFile:
    name: str
    size: int
    download_url: str
    md5: str | None = None

    @property
    def case_id(self) -> str:
        return Path(self.name).stem


def fetch_manifest(api_url: str = FIGSHARE_API_URL) -> dict:
    """Fetch the official Figshare article metadata."""

    request = Request(api_url, headers={"User-Agent": "brainsniffer/0.1"})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def parse_data_files(manifest: dict) -> list[DataFile]:
    """Extract and validate case files from a Figshare response."""

    files: list[DataFile] = []
    for item in manifest.get("files", []):
        name = Path(str(item.get("name", ""))).name
        if not re.fullmatch(r"case\d+\.mat", name, flags=re.IGNORECASE):
            continue
        files.append(
            DataFile(
                name=name,
                size=int(item.get("size", 0)),
                download_url=str(item["download_url"]),
                md5=(item.get("computed_md5") or item.get("supplied_md5") or None),
            )
        )
    return sorted(files, key=lambda item: int(re.search(r"\d+", item.name).group()))


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _matches_integrity(path: Path, item: DataFile) -> bool:
    if item.size and path.stat().st_size != item.size:
        return False
    return item.md5 is None or _md5_file(path) == item.md5.lower()


def available_case_ids(manifest: dict | None = None) -> list[str]:
    """Return case identifiers in stable numeric order."""

    manifest = manifest if manifest is not None else fetch_manifest()
    return [item.case_id for item in parse_data_files(manifest)]


def download_dataset(
    destination: str | Path = "data/raw",
    cases: list[int] | None = None,
    overwrite: bool = False,
    progress: Callable[[str, int, int], None] | None = None,
) -> list[Path]:
    """Download selected cases atomically and return their local paths.

    ``progress`` receives ``(filename, bytes_downloaded, expected_bytes)``.
    Existing files are preserved unless ``overwrite=True`` is explicit.
    """

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    files = parse_data_files(fetch_manifest())

    requested = None if cases is None else {f"case{case}" for case in cases}
    selected = [item for item in files if requested is None or item.case_id in requested]
    if cases is not None and len(selected) != len(requested):
        known = ", ".join(item.case_id for item in files)
        missing = ", ".join(sorted(requested - {item.case_id for item in selected}))
        raise ValueError(f"Casos desconhecidos: {missing}. Casos disponíveis: {known}")

    downloaded: list[Path] = []
    for item in selected:
        target = destination / item.name
        if target.exists() and not overwrite:
            if not _matches_integrity(target, item):
                raise OSError(
                    f"Arquivo local inconsistente para {item.name}; "
                    "use --overwrite para baixar novamente"
                )
            downloaded.append(target)
            if progress:
                progress(item.name, item.size, item.size)
            continue

        partial = destination / f".{item.name}.part"
        request = Request(item.download_url, headers={"User-Agent": "brainsniffer/0.1"})
        bytes_downloaded = 0
        digest = hashlib.md5()
        with urlopen(request, timeout=120) as response, partial.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
                digest.update(chunk)
                bytes_downloaded += len(chunk)
                if progress:
                    progress(item.name, bytes_downloaded, item.size)
        if item.size and bytes_downloaded != item.size:
            raise OSError(
                f"Download incompleto para {item.name}: "
                f"{bytes_downloaded} bytes; esperado {item.size}"
            )
        if item.md5 and digest.hexdigest() != item.md5.lower():
            raise OSError(
                f"Checksum MD5 inválido para {item.name}: "
                f"{digest.hexdigest()}; esperado {item.md5}"
            )
        partial.replace(target)
        downloaded.append(target)
    return downloaded
