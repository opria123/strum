import hashlib
import json
from pathlib import Path

import pytest

from src.model_bundle import (
    MANIFEST_FILENAME,
    BundleValidationError,
    discover_model_bundles,
    legacy_model_bundle,
    load_model_bundle,
)


def write_manifest(root: Path, components: dict, **overrides: object) -> Path:
    manifest = {
        "schema_version": 1,
        "model_id": "test-model",
        "compatibility": {"manifest_schema": 1, "strum_version": ">=0.1.0"},
        "components": components,
    }
    manifest.update(overrides)
    path = root / MANIFEST_FILENAME
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_load_resolves_component_paths_and_checks_hashes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights" / "best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"test checkpoint")
    config = tmp_path / "configs" / "model.yaml"
    config.parent.mkdir()
    config.write_text("model: {}\n", encoding="utf-8")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    write_manifest(
        tmp_path,
        {"drums.v14_onset": {"checkpoint": "weights/best.pt", "config": "configs/model.yaml", "sha256": digest}},
    )

    bundle = load_model_bundle(tmp_path, check_files=True)

    component = bundle.component("drums.v14_onset")
    assert component is not None
    assert component.checkpoint == checkpoint
    assert component.config == config
    assert bundle.validate(check_files=True, verify_hashes=True) == []


def test_manifest_rejects_path_escape(tmp_path: Path) -> None:
    write_manifest(tmp_path, {"guitar.onset": {"checkpoint": "../outside.pt"}})

    with pytest.raises(BundleValidationError, match="escapes the bundle root"):
        load_model_bundle(tmp_path)


def test_manifest_rejects_checksum_without_checkpoint(tmp_path: Path) -> None:
    write_manifest(tmp_path, {"guitar.onset": {"sha256": "a" * 64}})

    with pytest.raises(BundleValidationError, match="requires a checkpoint"):
        load_model_bundle(tmp_path)


def test_missing_files_are_optional_until_explicitly_checked(tmp_path: Path) -> None:
    write_manifest(tmp_path, {"guitar.onset": {"checkpoint": "weights/missing.pt"}})

    bundle = load_model_bundle(tmp_path)
    assert bundle.validate(check_files=True) == [
        f"guitar.onset: checkpoint not found: {tmp_path / 'weights' / 'missing.pt'}"
    ]
    with pytest.raises(BundleValidationError, match="checkpoint not found"):
        load_model_bundle(tmp_path, check_files=True)


def test_declared_source_revision_is_unverified_without_runtime_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("STRUM_SOURCE_REVISION", raising=False)
    write_manifest(
        tmp_path,
        {"guitar.onset": {"checkpoint": "weights/best.pt"}},
        compatibility={"manifest_schema": 1, "strum_version": ">=0.1.0", "strum_revision": "abc123"},
    )

    bundle = load_model_bundle(tmp_path)

    assert bundle.validate() == []
    assert bundle.compatibility_status() == [
        "STRUM source revision abc123: declared, unverified (set STRUM_SOURCE_REVISION to verify)"
    ]


def test_source_revision_mismatch_is_rejected_when_runtime_revision_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRUM_SOURCE_REVISION", "different")
    write_manifest(
        tmp_path,
        {"guitar.onset": {"checkpoint": "weights/best.pt"}},
        compatibility={"manifest_schema": 1, "strum_version": ">=0.1.0", "strum_revision": "abc123"},
    )

    with pytest.raises(BundleValidationError, match="requires STRUM source revision abc123"):
        load_model_bundle(tmp_path)


def test_discovery_lists_valid_child_bundles_only(tmp_path: Path) -> None:
    valid = tmp_path / "valid"
    valid.mkdir()
    write_manifest(valid, {"guitar.onset": {"checkpoint": "weights/best.pt"}}, model_id="good")
    invalid = tmp_path / "invalid"
    invalid.mkdir()
    (invalid / MANIFEST_FILENAME).write_text("not json", encoding="utf-8")

    bundles = discover_model_bundles(tmp_path)

    assert [bundle.model_id for bundle in bundles] == ["good"]


def test_legacy_bundle_preserves_current_checkpoint_layout(tmp_path: Path) -> None:
    bundle = legacy_model_bundle(tmp_path)

    assert bundle.legacy
    assert bundle.checkpoint("drums.v14_onset") == tmp_path / "checkpoints/drums_v14/best.pt"
    assert bundle.config("drums.ensemble.v17") == tmp_path / "configs/onset_classifier_v17.yaml"
