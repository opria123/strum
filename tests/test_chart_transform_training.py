import json
import math
import shutil
import wave
from array import array
from pathlib import Path

import pytest
import torch

from scripts.infer_chart_transform import load_source_events, predict
from scripts.train_chart_transform import (
    ChartEvent,
    DatasetValidationError,
    TrainingConfig,
    _parse_events,
    train,
)
from src.model_bundle import load_model_bundle
from src.models.chart_audio import AudioFeatureError, event_audio_features
from src.models.chart_transform import EventTransformMLP


def _write_test_song(path: Path, frequency_hz: float) -> None:
    sample_rate = 16_000
    samples = array(
        "h",
        (
            round(12_000 * math.sin(2 * math.pi * frequency_hz * index / sample_rate))
            for index in range(sample_rate)
        ),
    )
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())


@pytest.mark.parametrize("time_ms", [-1, float("nan"), float("inf")])
def test_chart_event_parsers_reject_invalid_timestamps(tmp_path: Path, time_ms: float) -> None:
    with pytest.raises(DatasetValidationError, match="numeric time_ms"):
        _parse_events([{"time_ms": time_ms, "lanes": [0]}], 5, 1, "source_events")

    source_events = tmp_path / "events.json"
    source_events.write_text(json.dumps({"source_events": [{"time_ms": time_ms, "lanes": [0]}]}))
    with pytest.raises(DatasetValidationError, match="numeric time_ms"):
        load_source_events(source_events, 5)


def test_train_revalidates_audio_overrides_before_reading_the_dataset(tmp_path: Path) -> None:
    config = TrainingConfig(
        dataset_manifest=str(tmp_path / "not-read.json"),
        output_dir=str(tmp_path / "output"),
        model_id="invalid-audio-config",
        source_difficulty="Expert",
        target_difficulty="Hard",
        audio_manifest=str(tmp_path / "audio-manifest.json"),
    )
    with pytest.raises(DatasetValidationError, match="audio_manifest requires"):
        train(config)


def test_cpu_chart_pair_training_writes_valid_model_bundle(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    records = [
        {
            "song_id": "song-a",
            "instrument": "bass",
            "source_difficulty": "Expert",
            "target_difficulty": "Hard",
            "source_events": [{"time_ms": 0, "lanes": [0]}, {"time_ms": 500, "lanes": [1, 2]}],
            "target_events": [{"time_ms": 0, "lanes": [0]}, {"time_ms": 500, "lanes": [1]}],
        },
        {
            "song_id": "song-b",
            "instrument": "bass",
            "source_difficulty": "Expert",
            "target_difficulty": "Hard",
            "source_events": [{"time_ms": 0, "lanes": [3]}, {"time_ms": 400, "lanes": [4]}],
            "target_events": [{"time_ms": 0, "lanes": [3]}],
        },
    ]
    (dataset_dir / "pairs.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n"
    )
    (dataset_dir / "dataset-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "strum-chart-pairs/v1",
                "dataset_id": "local-test-pairs",
                "records": "pairs.jsonl",
                "provenance": "synthetic test fixture; replace with documented local chart provenance",
                "license": "test-only",
                "instrument": "bass",
            }
        )
    )
    output_dir = tmp_path / "bundle"
    result = train(
        TrainingConfig(
            dataset_manifest=str(dataset_dir / "dataset-manifest.json"),
            output_dir=str(output_dir),
            model_id="test-expert-to-hard",
            source_difficulty="Expert",
            target_difficulty="Hard",
            seed=7,
            validation_fraction=0.5,
            hidden_dim=4,
            epochs=1,
            device="cpu",
            strum_revision="test-revision",
        )
    )

    bundle = load_model_bundle(output_dir, check_files=True)
    metadata = json.loads((output_dir / "training-metadata.json").read_text())

    assert bundle.model_id == "test-expert-to-hard"
    assert bundle.component("chart_transform.bass.expert_to_hard") is not None
    assert bundle.validate(check_files=True, verify_hashes=True) == []
    assert metadata["dataset"]["provenance"].startswith("synthetic")
    assert metadata["dataset"]["license"] == "test-only"
    assert metadata["dataset"]["instrument"] == "bass"
    assert set(metadata["split"]["train_song_ids"]).isdisjoint(
        metadata["split"]["validation_song_ids"]
    )
    assert result["metrics"]["validation"]["loss"] >= 0

    fine_tune_output = tmp_path / "fine-tuned-bundle"
    fine_tuned = train(
        TrainingConfig(
            dataset_manifest=str(dataset_dir / "dataset-manifest.json"),
            output_dir=str(fine_tune_output),
            model_id="test-expert-to-hard-fine-tuned",
            source_difficulty="Expert",
            target_difficulty="Hard",
            seed=7,
            validation_fraction=0.5,
            hidden_dim=4,
            epochs=1,
            device="cpu",
            init_checkpoint=str(output_dir / "weights" / "chart_transform.pt"),
        )
    )
    fine_tuned_metadata = json.loads((fine_tune_output / "training-metadata.json").read_text())

    assert fine_tuned["metrics"]["validation"]["loss"] >= 0
    assert "checkpoint_sha256" in fine_tuned_metadata["initialization"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is unavailable")
def test_audio_conditioned_training_and_inference_keep_song_paths_local(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    records = [
        {
            "song_id": "song-a",
            "source_difficulty": "Expert",
            "target_difficulty": "Hard",
            "source_events": [{"time_ms": 100, "lanes": [0]}, {"time_ms": 500, "lanes": [1]}],
            "target_events": [{"time_ms": 100, "lanes": [0]}],
        },
        {
            "song_id": "song-b",
            "source_difficulty": "Expert",
            "target_difficulty": "Hard",
            "source_events": [{"time_ms": 200, "lanes": [2]}, {"time_ms": 600, "lanes": [3]}],
            "target_events": [{"time_ms": 200, "lanes": [2]}],
        },
    ]
    (dataset_dir / "pairs.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n"
    )
    (dataset_dir / "dataset-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "strum-chart-pairs/v1",
                "dataset_id": "audio-test-pairs",
                "records": "pairs.jsonl",
                "provenance": "synthetic audio fixture",
                "license": "test-only",
            }
        )
    )
    _write_test_song(dataset_dir / "song-a.wav", 220.0)
    _write_test_song(dataset_dir / "song-b.wav", 440.0)
    audio_manifest = dataset_dir / "audio-manifest.json"
    audio_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "strum-local-audio-assets/v1",
                "assets": [
                    {"song_id": "song-a", "audio": "song-a.wav"},
                    {"song_id": "song-b", "audio": "song-b.wav"},
                ],
            }
        )
    )
    output_dir = tmp_path / "audio-bundle"
    train(
        TrainingConfig(
            dataset_manifest=str(dataset_dir / "dataset-manifest.json"),
            output_dir=str(output_dir),
            model_id="audio-test-expert-to-hard",
            source_difficulty="Expert",
            target_difficulty="Hard",
            validation_fraction=0.5,
            hidden_dim=4,
            epochs=1,
            device="cpu",
            audio_feature_mode="rms_onset_v1",
            audio_manifest=str(audio_manifest),
        )
    )

    checkpoint_path = output_dir / "weights" / "chart_transform.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = json.loads((output_dir / "training-metadata.json").read_text())
    portable_config = (output_dir / "configs" / "training-config.json").read_text()
    source_events = (ChartEvent(100.0, (0,)), ChartEvent(500.0, (1,)))

    assert checkpoint["audio_feature_dim"] == 2
    assert checkpoint["instrument"] == "guitar"
    assert metadata["audio_conditioning"]["audio_manifest_sha256"]
    assert str(audio_manifest) not in portable_config
    with pytest.raises(ValueError, match="requires --song"):
        predict(checkpoint_path, source_events, song_path=None, device_name="cpu", threshold=0.5)
    assert isinstance(
        predict(
            checkpoint_path,
            source_events,
            song_path=dataset_dir / "song-a.wav",
            device_name="cpu",
            threshold=0.5,
        ),
        list,
    )
    with pytest.raises(AudioFeatureError, match="duration limit"):
        event_audio_features(
            dataset_dir / "song-a.wav",
            [1_201_000.0],
            sample_rate=16_000,
            window_ms=50.0,
            max_duration_seconds=1_200.0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_chart_pair_training_uses_cuda_and_saves_portable_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    records = [
        {
            "song_id": "song-a",
            "source_difficulty": "Expert",
            "target_difficulty": "Hard",
            "source_events": [{"time_ms": 0, "lanes": [0]}],
            "target_events": [{"time_ms": 0, "lanes": [0]}],
        },
        {
            "song_id": "song-b",
            "source_difficulty": "Expert",
            "target_difficulty": "Hard",
            "source_events": [{"time_ms": 0, "lanes": [1]}],
            "target_events": [{"time_ms": 0, "lanes": [1]}],
        },
    ]
    (dataset_dir / "pairs.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n"
    )
    (dataset_dir / "dataset-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "strum-chart-pairs/v1",
                "dataset_id": "cuda-test-pairs",
                "records": "pairs.jsonl",
                "provenance": "synthetic CUDA test fixture",
                "license": "test-only",
            }
        )
    )
    output_dir = tmp_path / "bundle"
    observed_devices: list[tuple[torch.device, torch.device]] = []
    original_forward = EventTransformMLP.forward

    def record_forward(self: EventTransformMLP, features: torch.Tensor) -> torch.Tensor:
        observed_devices.append((next(self.parameters()).device, features.device))
        return original_forward(self, features)

    monkeypatch.setattr(EventTransformMLP, "forward", record_forward)

    train(
        TrainingConfig(
            dataset_manifest=str(dataset_dir / "dataset-manifest.json"),
            output_dir=str(output_dir),
            model_id="cuda-test-expert-to-hard",
            source_difficulty="Expert",
            target_difficulty="Hard",
            validation_fraction=0.5,
            hidden_dim=4,
            epochs=1,
            device="cuda:0",
        )
    )

    metadata = json.loads((output_dir / "training-metadata.json").read_text())
    checkpoint = torch.load(
        output_dir / "weights" / "chart_transform.pt", map_location="cpu", weights_only=False
    )

    assert metadata["runtime"]["device"]["requested"] == "cuda:0"
    assert metadata["runtime"]["device"]["resolved"] == "cuda:0"
    assert metadata["runtime"]["device"]["cuda_device_name"]
    assert observed_devices
    assert all(
        model_device.type == features_device.type == "cuda"
        for model_device, features_device in observed_devices
    )
    assert all(tensor.device.type == "cpu" for tensor in checkpoint["model_state_dict"].values())
