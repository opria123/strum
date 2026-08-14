"""Versioned STRUM model-bundle manifests and checkpoint resolution.

Bundles make a trained set of checkpoints portable without changing the
working-directory assumptions of the existing pipeline.  A bundle directory
contains ``strum-model-bundle.json`` and any checkpoint/configuration files it
references.  Paths in a manifest are always relative to that directory.

When no bundle is selected, :func:`get_active_bundle` exposes the repository's
historic ``checkpoints/`` layout as a virtual legacy bundle.  This is
intentional: existing commands and installations continue to work unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src import PROJECT_ROOT, __version__


MANIFEST_FILENAME = "strum-model-bundle.json"
MANIFEST_SCHEMA_VERSION = 1


class BundleValidationError(ValueError):
    """Raised when a model bundle manifest is invalid or incompatible."""


@dataclass(frozen=True)
class ModelComponent:
    """One named checkpoint-bearing component within a model bundle."""

    name: str
    root: Path
    checkpoint: Path | None = None
    config: Path | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class ModelBundle:
    """A parsed manifest with paths resolved relative to its bundle root."""

    root: Path
    model_id: str
    schema_version: int
    compatibility: dict[str, Any]
    components: dict[str, ModelComponent]
    manifest_path: Path | None = None
    legacy: bool = False

    def component(self, name: str) -> ModelComponent | None:
        """Return a declared component, or ``None`` for an optional override."""
        return self.components.get(name)

    def checkpoint(self, name: str, default: Path | None = None) -> Path | None:
        """Resolve a component checkpoint, falling back to ``default`` when absent."""
        component = self.component(name)
        return component.checkpoint if component and component.checkpoint else default

    def config(self, name: str, default: Path | None = None) -> Path | None:
        """Resolve a component configuration, falling back to ``default`` when absent."""
        component = self.component(name)
        return component.config if component and component.config else default

    def validate(self, *, check_files: bool = False, verify_hashes: bool = False) -> list[str]:
        """Return human-readable validation errors without loading any ML code."""
        errors: list[str] = []
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            errors.append(
                f"unsupported manifest schema {self.schema_version}; "
                f"expected {MANIFEST_SCHEMA_VERSION}"
            )
        required_schema = self.compatibility.get("manifest_schema")
        if required_schema is not None and required_schema != MANIFEST_SCHEMA_VERSION:
            errors.append(
                "bundle requires manifest schema "
                f"{required_schema}, but STRUM supports {MANIFEST_SCHEMA_VERSION}"
            )
        required_strum = self.compatibility.get("strum_version")
        if required_strum is not None and not _version_is_compatible(__version__, required_strum):
            errors.append(f"bundle requires STRUM {required_strum}; running {__version__}")
        required_revision = self.compatibility.get("strum_revision")
        if required_revision is not None:
            if not isinstance(required_revision, str) or not required_revision.strip():
                errors.append("bundle strum_revision must be a non-empty string")
            else:
                runtime_revision = get_runtime_revision()
                if runtime_revision is not None and runtime_revision != required_revision:
                    errors.append(
                        f"bundle requires STRUM source revision {required_revision}; "
                        f"running {runtime_revision}"
                    )

        if check_files:
            for component in self.components.values():
                for label, path in (("checkpoint", component.checkpoint), ("config", component.config)):
                    if path is not None and not path.is_file():
                        errors.append(f"{component.name}: {label} not found: {path}")
                if verify_hashes and component.sha256 and component.checkpoint and component.checkpoint.is_file():
                    actual = _sha256(component.checkpoint)
                    if actual != component.sha256:
                        errors.append(
                            f"{component.name}: checkpoint sha256 mismatch "
                            f"(expected {component.sha256}, got {actual})"
                        )
        return errors

    def compatibility_status(self) -> list[str]:
        """Describe compatibility declarations that cannot be verified locally."""
        required_revision = self.compatibility.get("strum_revision")
        if not isinstance(required_revision, str) or not required_revision.strip():
            return []
        runtime_revision = get_runtime_revision()
        if runtime_revision is None:
            return [
                f"STRUM source revision {required_revision}: declared, unverified "
                "(set STRUM_SOURCE_REVISION to verify)"
            ]
        return [f"STRUM source revision {required_revision}: verified"]


def _version_is_compatible(current: str, requirement: object) -> bool:
    """Check a simple exact or comma-separated semantic-version requirement.

    This intentionally supports the small subset needed by portable manifests
    without making model discovery depend on ``packaging`` at runtime.
    """
    if not isinstance(requirement, str) or not requirement.strip():
        return False
    if requirement == current:
        return True

    current_parts = _version_tuple(current)
    for term in (part.strip() for part in requirement.split(",")):
        if term.startswith(">="):
            if current_parts < _version_tuple(term[2:]):
                return False
        elif term.startswith(">"):
            if current_parts <= _version_tuple(term[1:]):
                return False
        elif term.startswith("<="):
            if current_parts > _version_tuple(term[2:]):
                return False
        elif term.startswith("<"):
            if current_parts >= _version_tuple(term[1:]):
                return False
        elif term.startswith("=="):
            if current != term[2:]:
                return False
        else:
            return False
    return True


def get_runtime_revision() -> str | None:
    """Return an explicitly supplied STRUM source revision, if one is known.

    Wheels and copied source trees do not reliably contain Git metadata.  The
    caller that pins STRUM (for example, an editor integration) can therefore
    set ``STRUM_SOURCE_REVISION`` to make a bundle's revision requirement
    enforceable.  Absence means "declared but unverified", not incompatibility.
    """
    revision = os.environ.get("STRUM_SOURCE_REVISION")
    return revision.strip() if revision and revision.strip() else None


def _version_tuple(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in value.strip().split("."))
    except ValueError as error:
        raise BundleValidationError(f"invalid STRUM version requirement: {value!r}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_relative_path(root: Path, value: object, field: str, component: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise BundleValidationError(f"{component}.{field} must be a non-empty relative path")
    candidate = Path(value)
    if candidate.is_absolute():
        raise BundleValidationError(f"{component}.{field} must be relative to the bundle root")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise BundleValidationError(f"{component}.{field} escapes the bundle root") from error
    return resolved


def _parse_component(root: Path, name: str, value: object) -> ModelComponent:
    if not isinstance(value, dict):
        raise BundleValidationError(f"components.{name} must be an object")
    allowed = {"checkpoint", "config", "sha256"}
    unknown = set(value) - allowed
    if unknown:
        raise BundleValidationError(f"components.{name} has unknown field(s): {', '.join(sorted(unknown))}")
    sha256 = value.get("sha256")
    if sha256 is not None and (
        not isinstance(sha256, str) or len(sha256) != 64 or any(ch not in "0123456789abcdef" for ch in sha256)
    ):
        raise BundleValidationError(f"components.{name}.sha256 must be a lowercase SHA-256 hex digest")
    if sha256 is not None and value.get("checkpoint") is None:
        raise BundleValidationError(f"components.{name}.sha256 requires a checkpoint path")
    return ModelComponent(
        name=name,
        root=root,
        checkpoint=_resolve_relative_path(root, value.get("checkpoint"), "checkpoint", name),
        config=_resolve_relative_path(root, value.get("config"), "config", name),
        sha256=sha256,
    )


def load_model_bundle(path: str | Path, *, check_files: bool = False) -> ModelBundle:
    """Load a manifest file or a directory containing one.

    ``check_files`` verifies declared paths but does not deserialize model
    weights, so it is safe for installation and editor discovery flows.
    """
    candidate = Path(path).expanduser().resolve()
    manifest_path = candidate if candidate.name == MANIFEST_FILENAME else candidate / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise BundleValidationError(f"model bundle manifest not found: {manifest_path}")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise BundleValidationError(f"invalid JSON in {manifest_path}: {error}") from error
    if not isinstance(raw, dict):
        raise BundleValidationError("bundle manifest must be a JSON object")

    allowed = {"schema_version", "model_id", "compatibility", "components"}
    unknown = set(raw) - allowed
    missing = {"schema_version", "model_id", "compatibility", "components"} - set(raw)
    if unknown:
        raise BundleValidationError(f"bundle manifest has unknown field(s): {', '.join(sorted(unknown))}")
    if missing:
        raise BundleValidationError(f"bundle manifest missing field(s): {', '.join(sorted(missing))}")
    if not isinstance(raw["schema_version"], int):
        raise BundleValidationError("schema_version must be an integer")
    if not isinstance(raw["model_id"], str) or not raw["model_id"].strip():
        raise BundleValidationError("model_id must be a non-empty string")
    if not isinstance(raw["compatibility"], dict):
        raise BundleValidationError("compatibility must be an object")
    if not isinstance(raw["components"], dict) or not raw["components"]:
        raise BundleValidationError("components must be a non-empty object")

    root = manifest_path.parent.resolve()
    components = {name: _parse_component(root, name, value) for name, value in raw["components"].items()}
    bundle = ModelBundle(
        root=root,
        model_id=raw["model_id"],
        schema_version=raw["schema_version"],
        compatibility=raw["compatibility"],
        components=components,
        manifest_path=manifest_path,
    )
    errors = bundle.validate(check_files=check_files)
    if errors:
        raise BundleValidationError("; ".join(errors))
    return bundle


def legacy_model_bundle(root: Path = PROJECT_ROOT) -> ModelBundle:
    """Expose the established repository layout through the new resolver API."""
    root = root.resolve()
    component_paths = {
        "drums.v14_onset": {"checkpoint": "checkpoints/drums_v14/best.pt"},
        "drums.ensemble.v2": {
            "checkpoint": "checkpoints/onset_classifier/best_f1.pt",
            "config": "configs/onset_classifier.yaml",
        },
        "drums.ensemble.v4": {
            "checkpoint": "checkpoints/onset_classifier_v4/best_f1.pt",
            "config": "configs/onset_classifier_v4.yaml",
        },
        "drums.ensemble.v6": {
            "checkpoint": "checkpoints/onset_classifier_v6/best_f1.pt",
            "config": "configs/onset_classifier_v6.yaml",
        },
        "drums.ensemble.v12c": {
            "checkpoint": "checkpoints/onset_classifier_v12_clean/best_f1.pt",
            "config": "configs/onset_classifier_v12_clean.yaml",
        },
        "drums.ensemble.v15": {
            "checkpoint": "checkpoints/onset_classifier_v15/best_f1.pt",
            "config": "configs/onset_classifier_v15.yaml",
        },
        "drums.ensemble.v16": {
            "checkpoint": "checkpoints/onset_classifier_v16/best_f1.pt",
            "config": "configs/onset_classifier_v16.yaml",
        },
        "drums.ensemble.v17": {
            "checkpoint": "checkpoints/onset_classifier_v17/best_f1.pt",
            "config": "configs/onset_classifier_v17.yaml",
        },
        "guitar.onset": {"checkpoint": "checkpoints/guitar_v2/guitar_v2_onset/best.pt"},
    }
    components = {name: _parse_component(root, name, value) for name, value in component_paths.items()}
    return ModelBundle(
        root=root,
        model_id="legacy-repository-layout",
        schema_version=MANIFEST_SCHEMA_VERSION,
        compatibility={"manifest_schema": MANIFEST_SCHEMA_VERSION, "strum_version": f">={__version__}"},
        components=components,
        legacy=True,
    )


def get_active_bundle() -> ModelBundle:
    """Return the explicitly selected bundle or the backwards-compatible default.

    Set ``STRUM_MODEL_BUNDLE`` to either a bundle directory or its manifest
    file.  An explicit invalid selection fails early instead of silently using
    a different model set.
    """
    selected = os.environ.get("STRUM_MODEL_BUNDLE")
    return load_model_bundle(selected) if selected else legacy_model_bundle()


def discover_model_bundles(root: str | Path) -> list[ModelBundle]:
    """Find and validate manifests below a user-selected model directory.

    Invalid manifests are intentionally omitted: callers can independently
    call :func:`load_model_bundle` to display their validation error.
    """
    base = Path(root).expanduser()
    manifests: Iterable[Path]
    if base.name == MANIFEST_FILENAME:
        manifests = (base,)
    else:
        manifests = base.glob(f"*/{MANIFEST_FILENAME}")
    bundles: list[ModelBundle] = []
    for manifest in sorted(manifests):
        try:
            bundles.append(load_model_bundle(manifest))
        except BundleValidationError:
            continue
    return bundles


def _main() -> int:
    parser = argparse.ArgumentParser(description="Inspect STRUM model bundles without loading weights.")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate one bundle manifest")
    validate.add_argument("path", help="bundle directory or manifest path")
    validate.add_argument("--check-files", action="store_true", help="require declared checkpoint/config files")
    validate.add_argument("--verify-hashes", action="store_true", help="SHA-256 declared checkpoint files")
    listing = commands.add_parser("list", help="list valid child bundles in a directory")
    listing.add_argument("path", help="directory containing bundle directories")
    args = parser.parse_args()
    try:
        if args.command == "validate":
            check_files = args.check_files or args.verify_hashes
            bundle = load_model_bundle(args.path, check_files=check_files)
            errors = bundle.validate(check_files=check_files, verify_hashes=args.verify_hashes)
            if errors:
                raise BundleValidationError("; ".join(errors))
            print(f"valid: {bundle.model_id} ({bundle.manifest_path})")
            for status in bundle.compatibility_status():
                print(f"status: {status}")
        else:
            for bundle in discover_model_bundles(args.path):
                print(f"{bundle.model_id}\t{bundle.manifest_path}")
    except BundleValidationError as error:
        print(f"invalid: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
