"""Regression test for the apply_difficulty_reduction() idempotency guard.

Before this fix, apply_difficulty_reduction() (guitar/bass) had no check for
whether Hard/Medium/Easy were already populated, unlike its drums equivalent
(apply_drums_difficulty_reduction, which has always had a has_hard/has_medium/
has_easy skip). In the real batch_pipeline.py flow, create_combined_midi()
already reduces guitar/bass difficulties via reduce_to_difficulty() before
ChartEnhancer.enhance_chart() runs on the same notes.mid path, so this
function was silently re-thinning and re-applying a second, worse chord-shape
pass on top of already-C3-legal output.
"""
import sys
from pathlib import Path

import mido

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from chart_enhancer import ChartEnhancer  # noqa: E402


def _note_pair(tick: int, note: int, duration: int = 40) -> list[mido.Message]:
    return [
        mido.Message("note_on", note=note, velocity=100, time=tick),
        mido.Message("note_off", note=note, velocity=0, time=tick + duration),
    ]


def _build_track(events: list[tuple[int, mido.Message]]) -> mido.MidiTrack:
    """events: list of (absolute_tick, message) pairs, converted to delta-time."""
    events = sorted(events, key=lambda e: e[0])
    track = mido.MidiTrack()
    prev = 0
    for tick, msg in events:
        msg = msg.copy(time=tick - prev)
        track.append(msg)
        prev = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


def _notes_in_range(track: mido.MidiTrack, lo: int, hi: int) -> int:
    return sum(
        1 for msg in track if msg.type == "note_on" and msg.velocity > 0 and lo <= msg.note <= hi
    )


def test_skips_when_all_difficulties_already_populated():
    """A track with real notes in Expert/Hard/Medium/Easy must come back unchanged."""
    events = []
    for beat, tick in enumerate([0, 480, 960, 1440]):
        events += [(tick, m) for m in _note_pair(tick, 96 + (beat % 5))]  # Expert
        events += [(tick, m) for m in _note_pair(tick, 84 + (beat % 5))]  # Hard
        events += [(tick, m) for m in _note_pair(tick, 72 + (beat % 5))]  # Medium
        events += [(tick, m) for m in _note_pair(tick, 60 + min(beat % 5, 2))]  # Easy

    track = _build_track(events)
    before_msgs = [(m.type, getattr(m, "note", None), getattr(m, "time", None)) for m in track]

    enhancer = ChartEnhancer(tempo_bpm=120, ticks_per_beat=480)
    result = enhancer.apply_difficulty_reduction(track)

    after_msgs = [(m.type, getattr(m, "note", None), getattr(m, "time", None)) for m in result]
    assert after_msgs == before_msgs, "already-populated track must be returned unmodified"


def test_does_not_skip_when_a_lower_difficulty_is_missing():
    """The guard requires ALL of Hard/Medium/Easy to be present to skip.

    apply_difficulty_reduction() only thins note events that already exist in
    the Hard/Medium/Easy ranges -- it never fabricates new ones (that's
    reduce_to_difficulty()'s job in src/inference/guitar_bass.py, called
    earlier in _create_guitar_track()). So an Expert-only track (the shape a
    standalone `python chart_enhancer.py <expert-only-midi>` CLI run would
    hand in) must fall through the guard rather than being skipped, and must
    come out with its Expert notes untouched and no crash, even though there
    is nothing in the lower ranges for it to thin.
    """
    events = []
    for tick in [0, 480, 960, 1440, 1920, 2400, 2880, 3360]:
        events += [(t, m) for t, m in zip([tick] * 2, _note_pair(tick, 96))]

    track = _build_track(events)
    enhancer = ChartEnhancer(tempo_bpm=120, ticks_per_beat=480)
    result = enhancer.apply_difficulty_reduction(track)

    assert _notes_in_range(track, 96, 100) == 8
    assert _notes_in_range(result, 96, 100) == 8, "Expert notes must survive untouched"
    # Nothing to thin in the lower ranges yet -- that's expected, not a bug
    # apply_difficulty_reduction() is introducing; it documents an existing
    # gap (this function thins, it doesn't generate) that Fase 1 addresses.
    assert _notes_in_range(result, 84, 88) == 0
    assert _notes_in_range(result, 72, 76) == 0
    assert _notes_in_range(result, 60, 64) == 0
