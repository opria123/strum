import json
from pathlib import Path

import pytest
import torch

from scripts.train_chart_transform import TrainingConfig, train
from src.model_bundle import load_model_bundle
from src.models.chart_transform import EventTransformMLP


def test_cpu_chart_pair_training_writes_valid_model_bundle(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    records = [
        {
            "song_id": "song-a",
            "source_difficulty": "Expert",
            "target_difficulty": "Hard",
            "source_events": [{"time_ms": 0, "lanes": [0]}, {"time_ms": 500, "lanes": [1, 2]}],
            "target_events": [{"time_ms": 0, "lanes": [0]}, {"time_ms": 500, "lanes": [1]}],
        },
        {
            "song_id": "song-b",
            "source_difficulty": "Expert",
            "target_difficulty": "Hard",
            "source_events": [{"time_ms": 0, "lanes": [3]}, {"time_ms": 400, "lanes": [4]}],
            "target_events": [{"time_ms": 0, "lanes": [3]}],
        },
    ]
    (dataset_dir / "pairs.jsonl").write_text("\n".join(json.dumps(record) for record in records) + "\n")
    (dataset_dir / "dataset-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "strum-chart-pairs/v1",
                "dataset_id": "local-test-pairs",
                "records": "pairs.jsonl",
                "provenance": "synthetic test fixture; replace with documented local chart provenance",
                "license": "test-only",
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
    assert bundle.component("chart_transform.expert_to_hard") is not None
    assert bundle.validate(check_files=True, verify_hashes=True) == []
    assert metadata["dataset"]["provenance"].startswith("synthetic")
    assert metadata["dataset"]["license"] == "test-only"
    assert set(metadata["split"]["train_song_ids"]).isdisjoint(metadata["split"]["validation_song_ids"])
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
    (dataset_dir / "pairs.jsonl").write_text("\n".join(json.dumps(record) for record in records) + "\n")
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
    checkpoint = torch.load(output_dir / "weights" / "chart_transform.pt", map_location="cpu", weights_only=False)

    assert metadata["runtime"]["device"]["requested"] == "cuda:0"
    assert metadata["runtime"]["device"]["resolved"] == "cuda:0"
    assert metadata["runtime"]["device"]["cuda_device_name"]
    assert observed_devices
    assert all(model_device.type == features_device.type == "cuda" for model_device, features_device in observed_devices)
    assert all(tensor.device.type == "cpu" for tensor in checkpoint["model_state_dict"].values())
