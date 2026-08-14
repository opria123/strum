import json
from pathlib import Path

import pytest

mido = pytest.importorskip("mido")

from scripts.prepare_guitar_chart_pairs import (
    PreparationOptions,
    discover_notes_mid,
    prepare_dataset,
)


def test_prepare_guitar_chart_pairs_from_synthetic_notes_mid(tmp_path: Path) -> None:
    song_dir = tmp_path / "A Safe Song"
    song_dir.mkdir()
    midi_path = song_dir / "notes.mid"
    midi = mido.MidiFile(ticks_per_beat=480)
    tempo = mido.MidiTrack()
    tempo.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    midi.tracks.append(tempo)
    guitar = mido.MidiTrack()
    guitar.append(mido.MetaMessage("track_name", name="PART GUITAR", time=0))
    guitar.append(mido.Message("note_on", note=96, velocity=100, time=0))
    guitar.append(mido.Message("note_on", note=98, velocity=100, time=0))
    guitar.append(mido.Message("note_on", note=84, velocity=100, time=0))
    guitar.append(mido.Message("note_off", note=96, velocity=0, time=480))
    guitar.append(mido.Message("note_off", note=98, velocity=0, time=0))
    guitar.append(mido.Message("note_off", note=84, velocity=0, time=0))
    guitar.append(mido.Message("note_on", note=97, velocity=100, time=0))
    midi.tracks.append(guitar)
    midi.save(midi_path)

    result = prepare_dataset(
        discover_notes_mid([tmp_path]),
        tmp_path / "prepared",
        PreparationOptions(
            target_difficulty="Hard",
            dataset_id="synthetic-guitar-pairs",
            provenance="synthetic MIDI fixture",
            license="test-only",
        ),
    )

    manifest = json.loads((tmp_path / "prepared" / "dataset-manifest.json").read_text())
    records = [json.loads(line) for line in (tmp_path / "prepared" / "pairs.jsonl").read_text().splitlines()]

    assert result["record_count"] == 1
    assert manifest["format"] == "strum-chart-pairs/v1"
    assert manifest["provenance"] == "synthetic MIDI fixture"
    assert manifest["license"] == "test-only"
    assert records == [
        {
            "instrument": "guitar",
            "song_id": records[0]["song_id"],
            "source_difficulty": "Expert",
            "target_difficulty": "Hard",
            "source_events": [
                {"lanes": [0, 2], "time_ms": 0.0},
                {"lanes": [1], "time_ms": 500.0},
            ],
            "target_events": [{"lanes": [0], "time_ms": 0.0}],
        }
    ]
    assert records[0]["song_id"].startswith("a-safe-song-")
