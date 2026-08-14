#!/usr/bin/env python3
"""Prepare local five-lane Clone Hero ``notes.mid`` chart pairs.

This offline bridge supports the standard five-lane ``PART GUITAR``,
``PART BASS``, ``PART KEYS``, and ``PART DRUMS`` tracks. It writes the
``strum-chart-pairs/v1`` JSONL consumed by ``train_chart_transform.py``.
Open notes and modifiers are intentionally excluded: each dataset/model is
for one instrument and predicts its five base lanes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import mido

DIFFICULTY_BASE_NOTES = {"Expert": 96, "Hard": 84, "Medium": 72, "Easy": 60}
PAIR_DATASET_FORMAT = "strum-chart-pairs/v1"
FIVE_LANE_INSTRUMENT_TRACKS = {
    "guitar": "PART GUITAR",
    "bass": "PART BASS",
    "keys": "PART KEYS",
    "drums": "PART DRUMS",
}


class PreparationError(ValueError):
    """Raised when the selected inputs cannot produce a valid local dataset."""


@dataclass(frozen=True)
class PreparationOptions:
    target_difficulty: str
    dataset_id: str
    provenance: str
    license: str
    instrument: str = "guitar"

    def __post_init__(self) -> None:
        if self.target_difficulty not in {"Hard", "Medium", "Easy"}:
            raise PreparationError("target_difficulty must be Hard, Medium, or Easy")
        if self.instrument not in FIVE_LANE_INSTRUMENT_TRACKS:
            raise PreparationError(
                f"instrument must be one of: {', '.join(sorted(FIVE_LANE_INSTRUMENT_TRACKS))}"
            )
        for field in ("dataset_id", "provenance", "license"):
            if not getattr(self, field).strip():
                raise PreparationError(f"{field} must be non-empty")


def discover_notes_mid(inputs: Iterable[Path], list_files: Iterable[Path] = ()) -> list[Path]:
    """Collect unique ``notes.mid`` paths from explicit paths/directories/lists."""
    candidates = [Path(path).expanduser() for path in inputs]
    for list_file in list_files:
        list_path = Path(list_file).expanduser()
        if not list_path.is_file():
            raise PreparationError(f"input list not found: {list_path}")
        for line in list_path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                item = Path(line.strip()).expanduser()
                candidates.append(item if item.is_absolute() else list_path.parent / item)

    discovered: set[Path] = set()
    for candidate in candidates:
        if candidate.is_dir():
            discovered.update(
                path.resolve() for path in candidate.rglob("notes.mid") if path.is_file()
            )
        elif candidate.is_file() and candidate.name.lower() == "notes.mid":
            discovered.add(candidate.resolve())
        else:
            raise PreparationError(f"expected a notes.mid file or directory: {candidate}")
    if not discovered:
        raise PreparationError("no notes.mid files found")
    return sorted(discovered)


def parse_instrument_difficulties(
    midi_path: Path, instrument: str
) -> dict[str, list[dict[str, object]]]:
    """Extract grouped standard five-lane events for one supported instrument."""
    track_name = FIVE_LANE_INSTRUMENT_TRACKS.get(instrument)
    if track_name is None:
        raise PreparationError(f"unsupported five-lane instrument: {instrument}")
    midi = mido.MidiFile(midi_path)
    instrument_track = next((track for track in midi.tracks if track.name == track_name), None)
    if instrument_track is None:
        raise PreparationError(f"{track_name} track not found: {midi_path}")
    tempo_map = _tempo_map(midi)
    events_by_difficulty: dict[str, dict[int, set[int]]] = {
        difficulty: defaultdict(set) for difficulty in DIFFICULTY_BASE_NOTES
    }
    tick = 0
    for message in instrument_track:
        tick += message.time
        if message.type != "note_on" or message.velocity <= 0:
            continue
        for difficulty, base_note in DIFFICULTY_BASE_NOTES.items():
            if base_note <= message.note <= base_note + 4:
                events_by_difficulty[difficulty][tick].add(message.note - base_note)
                break
    return {
        difficulty: [
            {
                "time_ms": round(_tick_to_ms(tick, midi.ticks_per_beat, tempo_map), 3),
                "lanes": sorted(lanes),
            }
            for tick, lanes in sorted(events.items())
        ]
        for difficulty, events in events_by_difficulty.items()
    }


def parse_guitar_difficulties(midi_path: Path) -> dict[str, list[dict[str, object]]]:
    """Backward-compatible guitar alias for callers of the original bridge."""
    return parse_instrument_difficulties(midi_path, "guitar")


def prepare_dataset(
    midi_paths: Iterable[Path],
    output_dir: Path,
    options: PreparationOptions,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Write local chart-pair JSONL and its provenance/license-bearing manifest."""
    records: list[dict[str, object]] = []
    skipped: list[str] = []
    seen_song_ids: set[str] = set()
    for midi_path in sorted(Path(path).resolve() for path in midi_paths):
        try:
            difficulties = parse_instrument_difficulties(midi_path, options.instrument)
        except (OSError, PreparationError, ValueError) as error:
            skipped.append(f"{midi_path.name}: {error}")
            continue
        if not difficulties["Expert"]:
            skipped.append(
                f"{midi_path.name}: no Expert {FIVE_LANE_INSTRUMENT_TRACKS[options.instrument]} lanes"
            )
            continue
        song_id = _safe_song_id(midi_path)
        if song_id in seen_song_ids:
            skipped.append(f"{midi_path.name}: duplicate chart content")
            continue
        seen_song_ids.add(song_id)
        records.append(
            {
                "song_id": song_id,
                "instrument": options.instrument,
                "source_difficulty": "Expert",
                "target_difficulty": options.target_difficulty,
                "source_events": difficulties["Expert"],
                "target_events": difficulties[options.target_difficulty],
            }
        )
    if not records:
        details = "; ".join(skipped) if skipped else "no usable charts"
        raise PreparationError(
            f"no usable Expert {FIVE_LANE_INSTRUMENT_TRACKS[options.instrument]} charts: {details}"
        )

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "pairs.jsonl"
    manifest_path = output_dir / "dataset-manifest.json"
    if not overwrite and (records_path.exists() or manifest_path.exists()):
        raise PreparationError(
            f"output already contains {records_path.name} or {manifest_path.name}; pass --overwrite"
        )
    records_path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "format": PAIR_DATASET_FORMAT,
        "dataset_id": options.dataset_id,
        "records": records_path.name,
        "provenance": options.provenance,
        "license": options.license,
        "instrument": options.instrument,
        "source_difficulty": "Expert",
        "target_difficulty": options.target_difficulty,
        "record_count": len(records),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"manifest_path": manifest_path, "record_count": len(records), "skipped": skipped}


def _tempo_map(midi: mido.MidiFile) -> list[tuple[int, int]]:
    changes: list[tuple[int, int]] = [(0, 500_000)]
    for track in midi.tracks:
        tick = 0
        for message in track:
            tick += message.time
            if message.type == "set_tempo":
                changes.append((tick, message.tempo))
    changes.sort()
    deduplicated: list[tuple[int, int]] = []
    for change in changes:
        if deduplicated and change[0] == deduplicated[-1][0]:
            deduplicated[-1] = change
        else:
            deduplicated.append(change)
    return deduplicated


def _tick_to_ms(target_tick: int, ticks_per_beat: int, tempo_map: list[tuple[int, int]]) -> float:
    elapsed_ms = 0.0
    current_tick = 0
    current_tempo = tempo_map[0][1]
    for next_tick, next_tempo in tempo_map[1:]:
        if next_tick >= target_tick:
            break
        elapsed_ms += (next_tick - current_tick) * current_tempo / ticks_per_beat / 1000.0
        current_tick = next_tick
        current_tempo = next_tempo
    return elapsed_ms + (target_tick - current_tick) * current_tempo / ticks_per_beat / 1000.0


def _safe_song_id(midi_path: Path) -> str:
    """Use a filesystem-safe label plus content hash; never expose absolute paths."""
    parent_label = re.sub(r"[^a-z0-9]+", "-", midi_path.parent.name.lower()).strip("-") or "song"
    digest = hashlib.sha256(midi_path.read_bytes()).hexdigest()[:12]
    return f"{parent_label}-{digest}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare local five-lane instrument chart pairs from notes.mid files."
    )
    parser.add_argument(
        "--input", nargs="*", type=Path, default=[], help="notes.mid file(s) or directories to scan"
    )
    parser.add_argument(
        "--list-file",
        action="append",
        type=Path,
        default=[],
        help="UTF-8 list of notes.mid files/directories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for pairs.jsonl and dataset-manifest.json",
    )
    parser.add_argument("--target-difficulty", choices=["Hard", "Medium", "Easy"], required=True)
    parser.add_argument(
        "--instrument", choices=sorted(FIVE_LANE_INSTRUMENT_TRACKS), default="guitar"
    )
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument(
        "--provenance", required=True, help="source and collection method for authorized charts"
    )
    parser.add_argument(
        "--license", required=True, help="license/permission covering model-training use"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing pairs.jsonl/dataset-manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input and not args.list_file:
        raise SystemExit("provide at least one --input or --list-file")
    result = prepare_dataset(
        discover_notes_mid(args.input, args.list_file),
        args.output_dir,
        PreparationOptions(
            target_difficulty=args.target_difficulty,
            dataset_id=args.dataset_id,
            provenance=args.provenance,
            license=args.license,
            instrument=args.instrument,
        ),
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "manifest_path": str(result["manifest_path"]),
                "record_count": result["record_count"],
                "skipped": result["skipped"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
