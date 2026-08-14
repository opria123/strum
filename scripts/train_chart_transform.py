#!/usr/bin/env python3
"""Fine-tune a chart transform from paired charts and optional local songs.

Records are split by ``song_id`` before event expansion, preventing the same
song from leaking between train and validation.  An optional local-only audio
manifest supplies song-aligned features without copying audio paths or source
music into the resulting model bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from src import __version__
from src.model_bundle import MANIFEST_FILENAME, MANIFEST_SCHEMA_VERSION
from src.models.chart_audio import (
    AUDIO_FEATURE_DIM,
    AUDIO_FEATURE_MODE,
    MAX_AUDIO_DURATION_SECONDS,
    MAX_AUDIO_PCM_BYTES,
    MAX_AUDIO_SAMPLE_RATE,
    AudioFeatureError,
    event_audio_features,
)
from src.models.chart_transform import EventTransformMLP

DATASET_SCHEMA = "strum-chart-pairs/v1"
AUDIO_ASSET_SCHEMA = "strum-local-audio-assets/v1"


class DatasetValidationError(ValueError):
    """Raised when a local chart-pair dataset is incomplete or malformed."""


@dataclass(frozen=True)
class TrainingConfig:
    dataset_manifest: str
    output_dir: str
    model_id: str
    source_difficulty: str
    target_difficulty: str
    seed: int = 20260813
    validation_fraction: float = 0.2
    lane_count: int = 5
    alignment_tolerance_ms: float = 50.0
    hidden_dim: int = 32
    learning_rate: float = 0.001
    epochs: int = 20
    device: str = "auto"
    audio_feature_mode: str = "none"
    audio_manifest: str | None = None
    audio_sample_rate: int = 16_000
    audio_window_ms: float = 50.0
    audio_max_duration_seconds: float = 900.0
    init_checkpoint: str | None = None
    strum_revision: str | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> TrainingConfig:
        fields = set(cls.__dataclass_fields__)
        unknown = set(raw) - fields
        missing = {
            "dataset_manifest",
            "output_dir",
            "model_id",
            "source_difficulty",
            "target_difficulty",
        } - set(raw)
        if unknown:
            raise DatasetValidationError(
                f"unknown training config field(s): {', '.join(sorted(unknown))}"
            )
        if missing:
            raise DatasetValidationError(
                f"missing training config field(s): {', '.join(sorted(missing))}"
            )
        config = cls(**raw)
        if not config.dataset_manifest or not config.output_dir or not config.model_id:
            raise DatasetValidationError(
                "dataset_manifest, output_dir, and model_id must be non-empty"
            )
        if not 0 < config.validation_fraction < 1:
            raise DatasetValidationError("validation_fraction must be between 0 and 1")
        if config.lane_count < 1 or config.hidden_dim < 1 or config.epochs < 1:
            raise DatasetValidationError("lane_count, hidden_dim, and epochs must be positive")
        if config.learning_rate <= 0 or config.alignment_tolerance_ms < 0:
            raise DatasetValidationError(
                "learning_rate must be positive and alignment_tolerance_ms non-negative"
            )
        if config.audio_feature_mode not in {"none", AUDIO_FEATURE_MODE}:
            raise DatasetValidationError(f"audio_feature_mode must be none or {AUDIO_FEATURE_MODE}")
        if config.audio_feature_mode == "none" and config.audio_manifest is not None:
            raise DatasetValidationError("audio_manifest requires an audio_feature_mode")
        if config.audio_feature_mode != "none" and (
            config.audio_manifest is None or not config.audio_manifest.strip()
        ):
            raise DatasetValidationError("audio_feature_mode requires a non-empty audio_manifest")
        if (
            config.audio_sample_rate < 1
            or config.audio_window_ms <= 0
            or config.audio_max_duration_seconds <= 0
        ):
            raise DatasetValidationError(
                "audio sample rate, window, and duration limit must be positive"
            )
        if (
            config.audio_sample_rate > MAX_AUDIO_SAMPLE_RATE
            or config.audio_max_duration_seconds > MAX_AUDIO_DURATION_SECONDS
            or config.audio_sample_rate * config.audio_max_duration_seconds * 4
            > MAX_AUDIO_PCM_BYTES
        ):
            raise DatasetValidationError(
                "audio decode configuration exceeds the supported memory limits"
            )
        _resolve_device(config.device)
        if config.strum_revision is not None and not config.strum_revision.strip():
            raise DatasetValidationError("strum_revision must be non-empty when provided")
        if config.init_checkpoint is not None and not config.init_checkpoint.strip():
            raise DatasetValidationError("init_checkpoint must be non-empty when provided")
        return config


@dataclass(frozen=True)
class ChartEvent:
    time_ms: float
    lanes: tuple[int, ...]


@dataclass(frozen=True)
class ChartPair:
    song_id: str
    source_events: tuple[ChartEvent, ...]
    target_events: tuple[ChartEvent, ...]
    audio_path: Path | None = None


def load_config(path: str | Path) -> TrainingConfig:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise DatasetValidationError(f"invalid YAML config: {error}") from error
    if not isinstance(raw, dict):
        raise DatasetValidationError("training config must be a YAML object")
    return TrainingConfig.from_mapping(raw)


def load_dataset(config: TrainingConfig) -> tuple[list[ChartPair], dict[str, Any]]:
    manifest_path = Path(config.dataset_manifest).expanduser().resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DatasetValidationError(f"invalid dataset manifest JSON: {error}") from error
    if not isinstance(manifest, dict):
        raise DatasetValidationError("dataset manifest must be a JSON object")
    for field in ("dataset_id", "records", "provenance", "license"):
        if not manifest.get(field):
            raise DatasetValidationError(f"dataset manifest requires non-empty {field}")
    if manifest.get("schema_version") != 1 or manifest.get("format") != DATASET_SCHEMA:
        raise DatasetValidationError(f"dataset manifest must use format {DATASET_SCHEMA} schema 1")
    records_value = manifest["records"]
    if not isinstance(records_value, str):
        raise DatasetValidationError("dataset manifest records must be a relative JSONL path")
    records_path = _resolve_dataset_path(manifest_path.parent, records_value)
    pairs: list[ChartPair] = []
    for line_number, line in enumerate(
        records_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            raw_pair = json.loads(line)
        except json.JSONDecodeError as error:
            raise DatasetValidationError(
                f"invalid JSONL record at line {line_number}: {error}"
            ) from error
        pairs.append(_parse_pair(raw_pair, config, line_number))
    if not pairs:
        raise DatasetValidationError("dataset has no chart pairs")
    if len({pair.song_id for pair in pairs}) < 2:
        raise DatasetValidationError(
            "dataset requires at least two song_id values for song-level validation"
        )
    return pairs, manifest


def _resolve_dataset_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        raise DatasetValidationError("dataset records path must be relative to the manifest")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise DatasetValidationError(
            "dataset records path escapes the manifest directory"
        ) from error
    if not resolved.is_file():
        raise DatasetValidationError(f"dataset records file not found: {resolved}")
    return resolved


def _load_audio_assets(
    config: TrainingConfig, pairs: list[ChartPair]
) -> tuple[dict[str, Path], str | None]:
    if config.audio_feature_mode == "none":
        return {}, None
    assert config.audio_manifest is not None
    manifest_path = Path(config.audio_manifest).expanduser().resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DatasetValidationError(f"invalid audio manifest JSON: {error}") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("format") != AUDIO_ASSET_SCHEMA
    ):
        raise DatasetValidationError(
            f"audio manifest must use format {AUDIO_ASSET_SCHEMA} schema 1"
        )
    raw_assets = manifest.get("assets")
    if not isinstance(raw_assets, list):
        raise DatasetValidationError("audio manifest assets must be a list")
    assets: dict[str, Path] = {}
    for index, raw_asset in enumerate(raw_assets):
        if not isinstance(raw_asset, dict):
            raise DatasetValidationError(f"audio manifest asset {index} must be an object")
        song_id = raw_asset.get("song_id")
        relative_path = raw_asset.get("audio")
        if not isinstance(song_id, str) or not song_id or not isinstance(relative_path, str):
            raise DatasetValidationError(
                f"audio manifest asset {index} requires non-empty song_id and audio"
            )
        if song_id in assets:
            raise DatasetValidationError(f"audio manifest has duplicate song_id: {song_id}")
        assets[song_id] = _resolve_dataset_path(manifest_path.parent, relative_path)
    if {pair.song_id for pair in pairs} - set(assets):
        raise DatasetValidationError("audio manifest is missing one or more dataset song_id values")
    return assets, _sha256(manifest_path)


def _parse_pair(raw: object, config: TrainingConfig, line_number: int) -> ChartPair:
    if not isinstance(raw, dict):
        raise DatasetValidationError(f"record {line_number} must be an object")
    song_id = raw.get("song_id")
    if not isinstance(song_id, str) or not song_id:
        raise DatasetValidationError(f"record {line_number} requires a non-empty song_id")
    if raw.get("source_difficulty") != config.source_difficulty:
        raise DatasetValidationError(
            f"record {line_number} source_difficulty does not match config"
        )
    if raw.get("target_difficulty") != config.target_difficulty:
        raise DatasetValidationError(
            f"record {line_number} target_difficulty does not match config"
        )
    return ChartPair(
        song_id=song_id,
        source_events=_parse_events(
            raw.get("source_events"), config.lane_count, line_number, "source_events"
        ),
        target_events=_parse_events(
            raw.get("target_events"),
            config.lane_count,
            line_number,
            "target_events",
            allow_empty=True,
        ),
    )


def _parse_events(
    raw: object, lane_count: int, line_number: int, field: str, *, allow_empty: bool = False
) -> tuple[ChartEvent, ...]:
    if not isinstance(raw, list) or (not raw and not allow_empty):
        expectation = "a list" if allow_empty else "a non-empty list"
        raise DatasetValidationError(f"record {line_number} requires {expectation} for {field}")
    events: list[ChartEvent] = []
    for index, event in enumerate(raw):
        if (
            not isinstance(event, dict)
            or not isinstance(event.get("time_ms"), (int, float))
            or not math.isfinite(float(event["time_ms"]))
            or event["time_ms"] < 0
        ):
            raise DatasetValidationError(
                f"record {line_number} {field}[{index}] requires numeric time_ms"
            )
        lanes = event.get("lanes")
        if not isinstance(lanes, list) or not lanes:
            raise DatasetValidationError(
                f"record {line_number} {field}[{index}] requires non-empty lanes"
            )
        if any(not isinstance(lane, int) or lane < 0 or lane >= lane_count for lane in lanes):
            raise DatasetValidationError(
                f"record {line_number} {field}[{index}] has an invalid lane"
            )
        events.append(ChartEvent(float(event["time_ms"]), tuple(sorted(set(lanes)))))
    return tuple(sorted(events, key=lambda event: event.time_ms))


def _resolve_device(requested: str) -> torch.device:
    """Resolve a CPU or CUDA device without silently ignoring an explicit request."""
    if not isinstance(requested, str) or not requested.strip():
        raise DatasetValidationError("device must be auto, cpu, cuda, or cuda:<index>")
    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(normalized)
    except RuntimeError as error:
        raise DatasetValidationError("device must be auto, cpu, cuda, or cuda:<index>") from error
    if device.type not in {"cpu", "cuda"}:
        raise DatasetValidationError("device must be auto, cpu, cuda, or cuda:<index>")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise DatasetValidationError("CUDA was requested but is not available")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise DatasetValidationError(
                f"CUDA device index {index} is unavailable; found {torch.cuda.device_count()} device(s)"
            )
        return torch.device(f"cuda:{index}")
    return device


def _device_metadata(requested: str, device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "requested": requested,
        "resolved": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        metadata["cuda_device_count"] = torch.cuda.device_count()
        metadata["cuda_device_name"] = torch.cuda.get_device_name(device)
    return metadata


def split_by_song(
    pairs: list[ChartPair], seed: int, validation_fraction: float
) -> tuple[list[ChartPair], list[ChartPair]]:
    song_ids = sorted({pair.song_id for pair in pairs})
    random.Random(seed).shuffle(song_ids)
    validation_count = min(len(song_ids) - 1, max(1, round(len(song_ids) * validation_fraction)))
    validation_song_ids = set(song_ids[:validation_count])
    train = [pair for pair in pairs if pair.song_id not in validation_song_ids]
    validation = [pair for pair in pairs if pair.song_id in validation_song_ids]
    return train, validation


def _examples_for_pair(
    pair: ChartPair, config: TrainingConfig
) -> tuple[list[list[float]], list[list[float]], int]:
    target_labels = [[0.0] * config.lane_count for _ in pair.source_events]
    unmatched_target_events = 0
    for target in pair.target_events:
        candidates = [
            (abs(source.time_ms - target.time_ms), index)
            for index, source in enumerate(pair.source_events)
            if abs(source.time_ms - target.time_ms) <= config.alignment_tolerance_ms
        ]
        if not candidates:
            unmatched_target_events += 1
            continue
        _, source_index = min(candidates)
        for lane in target.lanes:
            target_labels[source_index][lane] = 1.0

    song_duration = max(pair.source_events[-1].time_ms, 1.0)
    if config.audio_feature_mode == "none":
        audio_features = [[] for _ in pair.source_events]
    else:
        if pair.audio_path is None:
            raise DatasetValidationError(
                "audio-conditioned training requires every chart pair to have a song asset"
            )
        try:
            audio_features = event_audio_features(
                pair.audio_path,
                [event.time_ms for event in pair.source_events],
                sample_rate=config.audio_sample_rate,
                window_ms=config.audio_window_ms,
                max_duration_seconds=config.audio_max_duration_seconds,
            )
        except AudioFeatureError as error:
            raise DatasetValidationError("could not extract local audio features") from error
    features: list[list[float]] = []
    for index, source in enumerate(pair.source_events):
        lane_vector = [1.0 if lane in source.lanes else 0.0 for lane in range(config.lane_count)]
        previous_time = pair.source_events[index - 1].time_ms if index else source.time_ms
        previous_gap = min((source.time_ms - previous_time) / 2000.0, 1.0)
        song_position = min(source.time_ms / song_duration, 1.0)
        features.append([*lane_vector, previous_gap, song_position, *audio_features[index]])
    return features, target_labels, unmatched_target_events


def _make_tensors(
    pairs: list[ChartPair], config: TrainingConfig
) -> tuple[torch.Tensor, torch.Tensor, int]:
    features: list[list[float]] = []
    targets: list[list[float]] = []
    unmatched = 0
    for pair in pairs:
        pair_features, pair_targets, pair_unmatched = _examples_for_pair(pair, config)
        features.extend(pair_features)
        targets.extend(pair_targets)
        unmatched += pair_unmatched
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
        unmatched,
    )


def _metrics(
    model: EventTransformMLP, features: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    model.eval()
    with torch.inference_mode():
        loss = F.binary_cross_entropy_with_logits(model(features), targets).item()
        predictions = (torch.sigmoid(model(features)) >= 0.5).to(torch.int64)
    expected = targets.to(torch.int64)
    true_positive = int(((predictions == 1) & (expected == 1)).sum())
    false_positive = int(((predictions == 1) & (expected == 0)).sum())
    false_negative = int(((predictions == 0) & (expected == 1)).sum())
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"loss": loss, "lane_precision": precision, "lane_recall": recall, "lane_f1": f1}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def train(config: TrainingConfig) -> dict[str, Any]:
    """Train on local pairs and write a self-describing registry-compatible bundle."""
    config = TrainingConfig.from_mapping(asdict(config))
    _seed_everything(config.seed)
    device = _resolve_device(config.device)
    pairs, dataset_manifest = load_dataset(config)
    audio_assets, audio_manifest_sha256 = _load_audio_assets(config, pairs)
    pairs = [replace(pair, audio_path=audio_assets.get(pair.song_id)) for pair in pairs]
    train_pairs, validation_pairs = split_by_song(pairs, config.seed, config.validation_fraction)
    train_features, train_targets, train_unmatched = _make_tensors(train_pairs, config)
    validation_features, validation_targets, validation_unmatched = _make_tensors(
        validation_pairs, config
    )
    train_features = train_features.to(device)
    train_targets = train_targets.to(device)
    validation_features = validation_features.to(device)
    validation_targets = validation_targets.to(device)

    audio_feature_dim = AUDIO_FEATURE_DIM if config.audio_feature_mode != "none" else 0
    model = EventTransformMLP(
        lane_count=config.lane_count,
        hidden_dim=config.hidden_dim,
        audio_feature_dim=audio_feature_dim,
    ).to(device)
    initialization: dict[str, str] | None = None
    if config.init_checkpoint:
        initial_path = Path(config.init_checkpoint).expanduser().resolve()
        if not initial_path.is_file():
            raise DatasetValidationError(f"init_checkpoint not found: {initial_path}")
        initial = torch.load(initial_path, map_location="cpu", weights_only=False)
        if not isinstance(initial, dict) or initial.get("model_type") != "EventTransformMLP":
            raise DatasetValidationError("init_checkpoint is not an EventTransformMLP checkpoint")
        if (
            initial.get("lane_count") != config.lane_count
            or initial.get("hidden_dim") != config.hidden_dim
            or initial.get("audio_feature_dim", 0) != audio_feature_dim
        ):
            raise DatasetValidationError(
                "init_checkpoint model shape does not match lane_count/hidden_dim"
            )
        model.load_state_dict(initial["model_state_dict"])
        initialization = {"checkpoint_sha256": _sha256(initial_path)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    model.train()
    for _ in range(config.epochs):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(train_features), train_targets)
        loss.backward()
        optimizer.step()

    metrics = {
        "train": _metrics(model, train_features, train_targets),
        "validation": _metrics(model, validation_features, validation_targets),
    }
    output_dir = Path(config.output_dir).expanduser().resolve()
    weights_path = output_dir / "weights" / "chart_transform.pt"
    model_config_path = output_dir / "configs" / "training-config.json"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    model_config_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {
                name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
            },
            "model_type": "EventTransformMLP",
            "lane_count": config.lane_count,
            "hidden_dim": config.hidden_dim,
            "audio_feature_dim": audio_feature_dim,
            "audio_feature_mode": config.audio_feature_mode,
            "audio_sample_rate": config.audio_sample_rate,
            "audio_window_ms": config.audio_window_ms,
            "audio_max_duration_seconds": config.audio_max_duration_seconds,
            "source_difficulty": config.source_difficulty,
            "target_difficulty": config.target_difficulty,
        },
        weights_path,
    )
    portable_config = asdict(config)
    for local_field in ("dataset_manifest", "output_dir", "audio_manifest", "init_checkpoint"):
        portable_config[local_field] = None
    model_config_path.write_text(
        json.dumps(portable_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    component_name = (
        f"chart_transform.{_slug(config.source_difficulty)}_to_{_slug(config.target_difficulty)}"
    )
    compatibility: dict[str, Any] = {
        "manifest_schema": MANIFEST_SCHEMA_VERSION,
        "strum_version": f">={__version__}",
    }
    if config.strum_revision:
        compatibility["strum_revision"] = config.strum_revision
    bundle_manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "model_id": config.model_id,
        "compatibility": compatibility,
        "components": {
            component_name: {
                "checkpoint": "weights/chart_transform.pt",
                "config": "configs/training-config.json",
                "sha256": _sha256(weights_path),
            }
        },
    }
    (output_dir / MANIFEST_FILENAME).write_text(
        json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata = {
        "dataset": {
            "dataset_id": dataset_manifest["dataset_id"],
            "format": dataset_manifest["format"],
            "provenance": dataset_manifest["provenance"],
            "license": dataset_manifest["license"],
        },
        "split": {
            "unit": "song_id",
            "seed": config.seed,
            "train_song_ids": sorted({pair.song_id for pair in train_pairs}),
            "validation_song_ids": sorted({pair.song_id for pair in validation_pairs}),
        },
        "alignment": {
            "tolerance_ms": config.alignment_tolerance_ms,
            "unmatched_target_events": {
                "train": train_unmatched,
                "validation": validation_unmatched,
            },
        },
        "runtime": {"device": _device_metadata(config.device, device)},
        "audio_conditioning": {
            "mode": config.audio_feature_mode,
            "feature_dim": audio_feature_dim,
            "audio_manifest_sha256": audio_manifest_sha256,
            "sample_rate": config.audio_sample_rate if audio_feature_dim else None,
            "window_ms": config.audio_window_ms if audio_feature_dim else None,
        },
        "metrics": metrics,
    }
    if initialization:
        metadata["initialization"] = initialization
    (output_dir / "training-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"bundle_dir": output_dir, "metrics": metrics, "metadata": metadata}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip(
        "_"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a local chart-pair transform model on CPU or CUDA."
    )
    parser.add_argument("--config", required=True, help="YAML training configuration")
    parser.add_argument("--dataset-manifest", help="override dataset_manifest in the YAML config")
    parser.add_argument("--output-dir", help="override output_dir in the YAML config")
    parser.add_argument("--epochs", type=int, help="override epochs in the YAML config")
    parser.add_argument("--seed", type=int, help="override seed in the YAML config")
    parser.add_argument("--device", help="override device: auto, cpu, cuda, or cuda:<index>")
    parser.add_argument(
        "--audio-manifest", help="local-only audio asset manifest for audio-conditioned training"
    )
    parser.add_argument(
        "--audio-feature-mode",
        choices=["none", AUDIO_FEATURE_MODE],
        help="override audio_feature_mode; required with --audio-manifest",
    )
    parser.add_argument("--init-checkpoint", help="optional compatible checkpoint to fine-tune")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    overrides = {
        key: value
        for key, value in {
            "dataset_manifest": args.dataset_manifest,
            "output_dir": args.output_dir,
            "epochs": args.epochs,
            "seed": args.seed,
            "device": args.device,
            "audio_feature_mode": args.audio_feature_mode,
            "audio_manifest": args.audio_manifest,
            "init_checkpoint": args.init_checkpoint,
        }.items()
        if value is not None
    }
    result = train(replace(config, **overrides))
    print(
        json.dumps(
            {
                "bundle_dir": str(result["bundle_dir"]),
                "validation": result["metrics"]["validation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
