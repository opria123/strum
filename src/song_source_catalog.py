"""Read OCTAVE's canonical local song-source catalog without source locations.

OCTAVE owns import adapters and catalog materialization. STRUM owns task views:
this module validates the catalog boundary, verifies managed assets, and exposes
only approved, instrument-specific inputs to dataset builders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

CATALOG_FORMAT = "octave-song-source-catalog/v1"
CATALOG_FILENAME = "catalog.json"
TRAINING_ALLOWED = "allowed"
INSTRUMENTS = frozenset(
    {"drums", "guitar", "bass", "keys", "vocals", "pro_keys", "pro_guitar", "pro_bass"}
)
AUDIO_ROLES = frozenset({"mix", "drums", "guitar", "bass", "keys", "vocals", "other"})
DIFFICULTIES = frozenset({"easy", "medium", "hard", "expert"})
SOURCE_ID_PATTERN = re.compile(r"^octave-src-[a-z0-9][a-z0-9-]{7,127}$")
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
SAFE_FILENAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
WARNING_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
UNSAFE_TEXT_PATTERN = re.compile(
    r"(?:[A-Za-z][A-Za-z0-9+.-]*://|file:|(?:^|[\s=:(])/(?:[^/\s]+/?)+|[A-Za-z]:[\\/])",
    re.IGNORECASE,
)


class CatalogValidationError(ValueError):
    """Raised when a catalog cannot safely cross the OCTAVE/STRUM boundary."""


@dataclass(frozen=True)
class CatalogAsset:
    asset_id: str
    sha256: str
    path: Path
    byte_length: int
    media_type: str | None


@dataclass(frozen=True)
class InstrumentCoverage:
    status: str
    difficulties: frozenset[str]
    track_names: tuple[str, ...]


@dataclass(frozen=True)
class CatalogRecord:
    source_id: str
    training_use: str
    notes_midi: CatalogAsset
    instruments: dict[str, InstrumentCoverage]
    audio: dict[str, CatalogAsset]


@dataclass(frozen=True)
class SongSourceCatalog:
    catalog_id: str
    root: Path
    records: tuple[CatalogRecord, ...]


@dataclass(frozen=True)
class TrainingSource:
    """A STRUM-safe task input with no original package location or metadata."""

    source_id: str
    instrument: str
    notes_midi: Path
    audio: Path | None
    difficulties: frozenset[str]


def load_catalog(catalog_root: str | Path) -> SongSourceCatalog:
    """Load a complete OCTAVE v1 catalog and validate all selected asset references."""
    root = Path(catalog_root).expanduser().resolve()
    if not root.is_dir():
        raise CatalogValidationError("catalog root must be a directory")
    manifest = _read_json(root / CATALOG_FILENAME, "catalog manifest")
    if not isinstance(manifest, dict):
        raise CatalogValidationError("catalog manifest must be an object")
    if set(manifest) - {"schema_version", "format", "catalog_id", "records", "created_by"}:
        raise CatalogValidationError("catalog manifest contains unsupported fields")
    if manifest.get("schema_version") != 1 or manifest.get("format") != CATALOG_FORMAT:
        raise CatalogValidationError(f"catalog must use {CATALOG_FORMAT}")
    catalog_id = _safe_text(manifest.get("catalog_id"), "catalog_id")
    _parse_created_by(manifest.get("created_by"))
    records_value = manifest.get("records")
    records_path = _resolve_records_path(root, records_value)
    try:
        record_lines = records_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise CatalogValidationError("catalog records are missing or unreadable") from error
    records: list[CatalogRecord] = []
    source_ids: set[str] = set()
    for line_number, line in enumerate(record_lines, start=1):
        if not line.strip():
            continue
        try:
            raw_record = json.loads(line)
        except json.JSONDecodeError as error:
            raise CatalogValidationError(
                f"catalog record {line_number} is not valid JSON"
            ) from error
        record = _parse_record(raw_record, root)
        if record.source_id in source_ids:
            raise CatalogValidationError("catalog has duplicate source_id values")
        source_ids.add(record.source_id)
        records.append(record)
    if not records:
        raise CatalogValidationError("catalog has no source records")
    return SongSourceCatalog(catalog_id=catalog_id, root=root, records=tuple(records))


def select_training_sources(
    catalog: SongSourceCatalog,
    instrument: str,
    *,
    required_difficulties: Iterable[str] = (),
    audio_role: str | None = None,
) -> tuple[TrainingSource, ...]:
    """Select only rights-approved records with deterministic task inputs."""
    if instrument not in INSTRUMENTS:
        raise CatalogValidationError("requested instrument is unsupported by the catalog contract")
    required = frozenset(required_difficulties)
    if not required <= DIFFICULTIES:
        raise CatalogValidationError("requested difficulty is unsupported by the catalog contract")
    if audio_role is not None and audio_role not in AUDIO_ROLES:
        raise CatalogValidationError("requested audio role is unsupported by the catalog contract")
    selected: list[TrainingSource] = []
    for record in catalog.records:
        if record.training_use != TRAINING_ALLOWED:
            continue
        coverage = record.instruments.get(instrument)
        if (
            coverage is None
            or coverage.status != "present"
            or not required <= coverage.difficulties
        ):
            continue
        audio = record.audio.get(audio_role) if audio_role else None
        if audio_role and audio is None:
            continue
        selected.append(
            TrainingSource(
                source_id=record.source_id,
                instrument=instrument,
                notes_midi=record.notes_midi.path,
                audio=audio.path if audio else None,
                difficulties=coverage.difficulties,
            )
        )
    return tuple(sorted(selected, key=lambda source: source.source_id))


def _read_json(path: Path, label: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise CatalogValidationError(f"{label} is missing") from error
    except json.JSONDecodeError as error:
        raise CatalogValidationError(f"{label} is not valid JSON") from error


def _resolve_records_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value.endswith(".jsonl"):
        raise CatalogValidationError("catalog records must be a relative JSONL path")
    return _resolve_relative(root, value, required_prefix=None)


def _resolve_relative(root: Path, relative_path: str, *, required_prefix: str | None) -> Path:
    candidate = Path(relative_path)
    if "\\" in relative_path or candidate.is_absolute() or ".." in candidate.parts:
        raise CatalogValidationError("catalog contains an unsafe relative path")
    if required_prefix is not None and not relative_path.startswith(required_prefix):
        raise CatalogValidationError("catalog asset is outside the managed asset namespace")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise CatalogValidationError("catalog path escapes the catalog root") from error
    return resolved


def _parse_record(raw: object, root: Path) -> CatalogRecord:
    if not isinstance(raw, dict):
        raise CatalogValidationError("catalog record must be an object")
    if set(raw) - {"source_id", "import", "rights", "metadata", "chart", "audio"}:
        raise CatalogValidationError("catalog record contains unsupported fields")
    source_id = raw.get("source_id")
    if not isinstance(source_id, str) or not SOURCE_ID_PATTERN.fullmatch(source_id):
        raise CatalogValidationError("catalog record has an invalid source_id")
    _parse_import(raw.get("import"))
    training_use = _parse_rights(raw.get("rights"))
    _parse_metadata(raw.get("metadata"))
    chart = raw.get("chart")
    if not isinstance(chart, dict) or set(chart) - {"notes_midi", "instruments"}:
        raise CatalogValidationError("catalog record chart must be an object")
    notes_midi = _parse_asset(chart.get("notes_midi"), root)
    instruments = _parse_instruments(chart.get("instruments"))
    audio = _parse_audio(raw.get("audio"), root)
    return CatalogRecord(source_id, training_use, notes_midi, instruments, audio)


def _parse_import(raw: object) -> None:
    if not isinstance(raw, dict) or raw.get("kind") not in {"sng", "rb3con", "zip", "song_folder"}:
        raise CatalogValidationError("catalog record has an invalid import kind")
    if set(raw) - {"kind", "adapter_version", "container_sha256", "warnings"}:
        raise CatalogValidationError("catalog import contains unsupported fields")
    _safe_text(raw.get("adapter_version"), "adapter_version")
    container_sha256 = raw.get("container_sha256")
    if container_sha256 is not None and (
        not isinstance(container_sha256, str) or not SHA256_PATTERN.fullmatch(container_sha256)
    ):
        raise CatalogValidationError("catalog record has an invalid container hash")
    warnings = raw.get("warnings", [])
    if not isinstance(warnings, list):
        raise CatalogValidationError("catalog import warnings must be a list")
    for warning in warnings:
        if not isinstance(warning, dict) or set(warning) != {"code"}:
            raise CatalogValidationError("catalog import warnings must contain only safe codes")
        code = warning.get("code")
        if not isinstance(code, str) or not WARNING_CODE_PATTERN.fullmatch(code):
            raise CatalogValidationError("catalog import warning code is invalid")


def _parse_rights(raw: object) -> str:
    if not isinstance(raw, dict) or raw.get("training_use") not in {
        "allowed",
        "review_required",
        "prohibited",
    }:
        raise CatalogValidationError("catalog record has an invalid training rights decision")
    if set(raw) - {"training_use", "provenance", "license"}:
        raise CatalogValidationError("catalog rights contain unsupported fields")
    _safe_text(raw.get("provenance"), "provenance")
    _safe_text(raw.get("license"), "license")
    return raw["training_use"]


def _parse_metadata(raw: object) -> None:
    if not isinstance(raw, dict):
        raise CatalogValidationError("catalog metadata must be an object")
    if set(raw) - {"name", "artist", "album", "genre", "year", "charter"}:
        raise CatalogValidationError("catalog metadata contains unsupported fields")
    for value in raw.values():
        _safe_text(value, "metadata")


def _parse_asset(raw: object, root: Path) -> CatalogAsset:
    if not isinstance(raw, dict):
        raise CatalogValidationError("catalog asset must be an object")
    if set(raw) - {"asset_id", "sha256", "relative_path", "byte_length", "media_type"}:
        raise CatalogValidationError("catalog asset contains unsupported fields")
    asset_id, sha256, relative_path, byte_length = (
        raw.get("asset_id"),
        raw.get("sha256"),
        raw.get("relative_path"),
        raw.get("byte_length"),
    )
    if not isinstance(sha256, str) or not SHA256_PATTERN.fullmatch(sha256):
        raise CatalogValidationError("catalog asset has an invalid SHA-256")
    if asset_id != f"sha256:{sha256}" or not isinstance(relative_path, str):
        raise CatalogValidationError("catalog asset identity does not match its SHA-256")
    expected_prefix = f"assets/sha256/{sha256}/"
    filename = relative_path.removeprefix(expected_prefix)
    if not relative_path.startswith(expected_prefix) or not SAFE_FILENAME_PATTERN.fullmatch(
        filename
    ):
        raise CatalogValidationError("catalog asset path does not match its content address")
    if not isinstance(byte_length, int) or byte_length < 0:
        raise CatalogValidationError("catalog asset has an invalid byte length")
    media_type = raw.get("media_type")
    if media_type is not None:
        _safe_text(media_type, "media_type")
    path = _resolve_relative(root, relative_path, required_prefix="assets/sha256/")
    try:
        asset_is_valid_size = path.is_file() and path.stat().st_size == byte_length
    except OSError:
        asset_is_valid_size = False
    if not asset_is_valid_size:
        raise CatalogValidationError("catalog asset is missing or has an unexpected length")
    if _sha256(path) != sha256:
        raise CatalogValidationError("catalog asset hash verification failed")
    return CatalogAsset(asset_id, sha256, path, byte_length, media_type)


def _parse_instruments(raw: object) -> dict[str, InstrumentCoverage]:
    if not isinstance(raw, dict) or set(raw) - INSTRUMENTS:
        raise CatalogValidationError("catalog chart instruments are invalid")
    instruments: dict[str, InstrumentCoverage] = {}
    for instrument, coverage in raw.items():
        if not isinstance(coverage, dict):
            raise CatalogValidationError("catalog instrument coverage must be an object")
        if set(coverage) - {"status", "difficulties", "track_names"}:
            raise CatalogValidationError("catalog instrument coverage contains unsupported fields")
        status = coverage.get("status")
        difficulties = coverage.get("difficulties")
        track_names = coverage.get("track_names")
        if (
            status not in {"present", "absent", "unsupported"}
            or not isinstance(difficulties, list)
            or not isinstance(track_names, list)
        ):
            raise CatalogValidationError("catalog instrument coverage is invalid")
        if not all(
            isinstance(value, str) and value in DIFFICULTIES for value in difficulties
        ) or len(set(difficulties)) != len(difficulties):
            raise CatalogValidationError("catalog instrument difficulties are invalid")
        if not all(isinstance(value, str) and _is_safe_text(value) for value in track_names) or len(
            set(track_names)
        ) != len(track_names):
            raise CatalogValidationError("catalog instrument track names are invalid")
        if (status == "present") != bool(difficulties and track_names):
            raise CatalogValidationError("catalog instrument coverage status is inconsistent")
        instruments[instrument] = InstrumentCoverage(
            status, frozenset(difficulties), tuple(track_names)
        )
    return instruments


def _parse_audio(raw: object, root: Path) -> dict[str, CatalogAsset]:
    if raw is None:
        return {}
    if not isinstance(raw, dict) or set(raw) - AUDIO_ROLES:
        raise CatalogValidationError("catalog audio roles are invalid")
    return {role: _parse_asset(asset, root) for role, asset in raw.items()}


def _parse_created_by(raw: object) -> None:
    if raw is None:
        return
    if not isinstance(raw, dict) or set(raw) - {"product", "version", "source_revision"}:
        raise CatalogValidationError("catalog creator metadata is invalid")
    if raw.get("product") != "octave":
        raise CatalogValidationError("catalog creator product must be octave")
    _safe_text(raw.get("version"), "creator version")
    if raw.get("source_revision") is not None:
        _safe_text(raw["source_revision"], "creator source revision")


def _safe_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not _is_safe_text(value):
        raise CatalogValidationError(f"catalog {field} contains unsafe text")
    return value


def _is_safe_text(value: str) -> bool:
    return (
        bool(value)
        and len(value) <= 512
        and "\\" not in value
        and not any(ord(char) < 32 for char in value)
        and not UNSAFE_TEXT_PATTERN.search(value)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise CatalogValidationError("catalog asset could not be read") from error
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate or list an OCTAVE song-source catalog.")
    parser.add_argument("catalog", type=Path, help="catalog root directory")
    parser.add_argument(
        "--instrument",
        choices=sorted(INSTRUMENTS),
        help="list allowed task inputs for one instrument",
    )
    parser.add_argument(
        "--audio-role", choices=sorted(AUDIO_ROLES), help="require a local audio role when listing"
    )
    args = parser.parse_args()
    catalog = load_catalog(args.catalog)
    if args.instrument:
        records = select_training_sources(catalog, args.instrument, audio_role=args.audio_role)
        print(
            json.dumps(
                {
                    "catalog_id": catalog.catalog_id,
                    "source_ids": [record.source_id for record in records],
                }
            )
        )
    else:
        print(json.dumps({"catalog_id": catalog.catalog_id, "record_count": len(catalog.records)}))


if __name__ == "__main__":
    main()
