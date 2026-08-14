#!/usr/bin/env python3
"""Apply a chart-transform checkpoint to source events, optionally with a song."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

try:
    from scripts.train_chart_transform import ChartEvent, DatasetValidationError, _resolve_device
except ModuleNotFoundError:  # Support ``python scripts/infer_chart_transform.py``.
    from train_chart_transform import ChartEvent, DatasetValidationError, _resolve_device
from src.models.chart_audio import (
    AUDIO_FEATURE_DIM,
    AUDIO_FEATURE_MODE,
    AudioFeatureError,
    event_audio_features,
)
from src.models.chart_transform import EventTransformMLP


def load_source_events(path: Path, lane_count: int) -> tuple[ChartEvent, ...]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DatasetValidationError(f"invalid source events JSON: {error}") from error
    events_value = raw.get("source_events") if isinstance(raw, dict) else raw
    if not isinstance(events_value, list) or not events_value:
        raise DatasetValidationError(
            "source events must be a non-empty array or an object with source_events"
        )
    events: list[ChartEvent] = []
    for index, raw_event in enumerate(events_value):
        if (
            not isinstance(raw_event, dict)
            or not isinstance(raw_event.get("time_ms"), (int, float))
            or not math.isfinite(float(raw_event["time_ms"]))
            or raw_event["time_ms"] < 0
        ):
            raise DatasetValidationError(f"source event {index} requires numeric time_ms")
        lanes = raw_event.get("lanes")
        if not isinstance(lanes, list) or not lanes:
            raise DatasetValidationError(f"source event {index} requires non-empty lanes")
        if any(not isinstance(lane, int) or lane < 0 or lane >= lane_count for lane in lanes):
            raise DatasetValidationError(f"source event {index} has an invalid lane")
        events.append(ChartEvent(float(raw_event["time_ms"]), tuple(sorted(set(lanes)))))
    return tuple(sorted(events, key=lambda event: event.time_ms))


def predict(
    checkpoint_path: Path,
    source_events: tuple[ChartEvent, ...],
    *,
    song_path: Path | None,
    device_name: str,
    threshold: float,
) -> list[dict[str, object]]:
    if not 0 < threshold < 1:
        raise DatasetValidationError("threshold must be between 0 and 1")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("model_type") != "EventTransformMLP":
        raise DatasetValidationError("checkpoint is not an EventTransformMLP checkpoint")
    lane_count = checkpoint.get("lane_count")
    hidden_dim = checkpoint.get("hidden_dim")
    audio_feature_dim = checkpoint.get("audio_feature_dim", 0)
    if (
        not isinstance(lane_count, int)
        or not isinstance(hidden_dim, int)
        or audio_feature_dim not in {0, AUDIO_FEATURE_DIM}
    ):
        raise DatasetValidationError("checkpoint has an unsupported chart-transform shape")
    audio_features: list[list[float]]
    if audio_feature_dim:
        if checkpoint.get("audio_feature_mode") != AUDIO_FEATURE_MODE or song_path is None:
            raise DatasetValidationError(
                "this checkpoint requires --song for audio-conditioned inference"
            )
        try:
            audio_features = event_audio_features(
                song_path,
                [event.time_ms for event in source_events],
                sample_rate=checkpoint.get("audio_sample_rate", 16_000),
                window_ms=checkpoint.get("audio_window_ms", 50.0),
                max_duration_seconds=checkpoint.get("audio_max_duration_seconds", 900.0),
            )
        except AudioFeatureError as error:
            raise DatasetValidationError("could not extract local audio features") from error
    else:
        audio_features = [[] for _ in source_events]

    duration = max(source_events[-1].time_ms, 1.0)
    feature_rows: list[list[float]] = []
    for index, event in enumerate(source_events):
        lanes = [1.0 if lane in event.lanes else 0.0 for lane in range(lane_count)]
        previous = source_events[index - 1].time_ms if index else event.time_ms
        gap = min((event.time_ms - previous) / 2000.0, 1.0)
        position = min(event.time_ms / duration, 1.0)
        feature_rows.append([*lanes, gap, position, *audio_features[index]])

    device = _resolve_device(device_name)
    model = EventTransformMLP(lane_count, hidden_dim, audio_feature_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.inference_mode():
        probabilities = torch.sigmoid(
            model(torch.tensor(feature_rows, dtype=torch.float32, device=device))
        ).cpu()
    return [
        {"time_ms": event.time_ms, "lanes": torch.where(row >= threshold)[0].tolist()}
        for event, row in zip(source_events, probabilities, strict=True)
        if bool(torch.any(row >= threshold))
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a local chart-transform checkpoint to source chart events."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--source-events", type=Path, required=True, help="JSON array or object with source_events"
    )
    parser.add_argument(
        "--song",
        type=Path,
        help="required for audio-conditioned checkpoints; never copied to output",
    )
    parser.add_argument("--output", type=Path, help="optional JSON output path; defaults to stdout")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    lane_count = checkpoint.get("lane_count") if isinstance(checkpoint, dict) else None
    if not isinstance(lane_count, int):
        raise SystemExit("checkpoint has no valid lane_count")
    events = load_source_events(args.source_events, lane_count)
    result: dict[str, Any] = {
        "source_difficulty": checkpoint.get("source_difficulty"),
        "target_difficulty": checkpoint.get("target_difficulty"),
        "events": predict(
            args.checkpoint,
            events,
            song_path=args.song,
            device_name=args.device,
            threshold=args.threshold,
        ),
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
