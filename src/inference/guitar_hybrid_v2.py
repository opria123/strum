"""Hybrid V2 guitar transcription pipeline.

Combines best-in-class components for each subtask:

  1. ONSET TIMING        — V2 GuitarOnsetCRNN          (val F1 0.814)
  2. PITCH TRANSCRIPTION — Spotify basic-pitch (ONNX)  (polyphonic, MIT)
  3. PITCH → FRET MAP    — rule-based, music-aware     (this module)
  4. SUSTAIN DETECTION   — basic-pitch note durations
  5. (later) HOPO/post-process — chart_postprocess.py

Each subtask uses the best tool. The neural fret head trained in V2 is
abandoned because pitch transcription is a solved problem and rule-based
fret mapping over true pitches outperforms guessing fret-bits from mel.

Output: list[GuitarEvent] compatible with src/inference/guitar_neural.py,
plus exports the same MIDI format.
"""
from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

# Quiet basic-pitch's missing-coreml/tflite/tf warnings on import
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("root").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.models.guitar_v1 import GuitarOnsetCRNN, OnsetCRNNConfig  # noqa: E402
from src.inference.guitar_neural import GuitarEvent, FRET_TO_MIDI  # noqa: E402
from src.model_bundle import get_active_bundle  # noqa: E402
import preprocess_guitar_windows as pgw  # noqa: E402

log = logging.getLogger("guitar_hybrid_v2")


# ─────────────────────────── basic-pitch wrapper ───────────────────────────
_BP_MODEL = None


def _get_basic_pitch_model():
    global _BP_MODEL
    if _BP_MODEL is None:
        from basic_pitch.inference import Model
        from basic_pitch import ICASSP_2022_MODEL_PATH
        _BP_MODEL = Model(ICASSP_2022_MODEL_PATH)
    return _BP_MODEL


@dataclass
class PitchNote:
    t_start: float
    t_end: float
    midi: int
    amplitude: float

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


def basic_pitch_predict(audio_path: Path, min_amplitude: float = 0.3,
                        min_pitch: int = 36, max_pitch: int = 88) -> list[PitchNote]:
    """Run basic-pitch on an audio file → filtered note events.

    Args:
        min_amplitude: drop notes with amplitude below this (suppresses noise)
        min_pitch: MIDI 36 = C2, below standard guitar low E (40).
            Override via STRUM_BP_MIN_PITCH env (e.g. 24 for bass, 21 for piano).
        max_pitch: MIDI 88 = E6, above 24th-fret high E.
            Override via STRUM_BP_MAX_PITCH env (e.g. 108 for piano).
    """
    import os as _os_bp
    if _os_bp.environ.get("STRUM_BP_MIN_PITCH"):
        min_pitch = int(_os_bp.environ["STRUM_BP_MIN_PITCH"])
    if _os_bp.environ.get("STRUM_BP_MAX_PITCH"):
        max_pitch = int(_os_bp.environ["STRUM_BP_MAX_PITCH"])
    from basic_pitch.inference import predict
    model = _get_basic_pitch_model()
    _, _, raw_events = predict(str(audio_path), model)

    notes: list[PitchNote] = []
    for ev in raw_events:
        t_start, t_end, pitch, amp, _bend = ev
        pitch = int(pitch)
        amp = float(amp)
        if amp < min_amplitude:
            continue
        if pitch < min_pitch or pitch > max_pitch:
            continue
        notes.append(PitchNote(
            t_start=float(t_start), t_end=float(t_end),
            midi=pitch, amplitude=amp,
        ))
    notes.sort(key=lambda n: n.t_start)
    return notes


# ─────────────────────────── Pitch → fret mapper ────────────────────────────
class PitchToFretMapper:
    """Rule-based pitch-set → fret-set mapping using song-wide pitch range.

    Strategy:
      * Compute song's playing range from basic-pitch p5..p95 of starting pitches
      * Bin range into 5 equal-width buckets → fret indices 0..4
      * For each onset's pitch set: bin each pitch, dedupe → fret set
      * Empty set → snap to nearest single pitch (rare fallback)

    Power chord recognition: if pitches form (root, root+7) or (root, root+12),
    map root to its bin and second pitch to root+1 fret (adjacent). This makes
    power chords feel right rather than spread across the neck.
    """

    def __init__(
        self,
        all_pitches: list[int],
        anchor_strength: float = 0.0,
    ):
        if not all_pitches:
            # Default to E2..E5 if no pitches
            self.p_lo, self.p_hi = 40.0, 76.0
        else:
            arr = np.array(all_pitches, dtype=np.float32)
            self.p_lo = float(np.percentile(arr, 5))
            self.p_hi = float(np.percentile(arr, 95))
        # STABILITY: enforce a minimum pitch range so each fret bin is at
        # least 3 semitones wide. Without this, narrow-range songs (verses
        # with consistent strumming on 1-2 notes) had bin widths < 1 semitone
        # — basic-pitch's normal ±1 semitone jitter then pushed the same
        # sung note across multiple frets every onset.
        min_range = 15.0
        if (self.p_hi - self.p_lo) < min_range:
            mid = (self.p_lo + self.p_hi) / 2.0
            self.p_lo = mid - min_range / 2.0
            self.p_hi = mid + min_range / 2.0
        self.p_range = max(self.p_hi - self.p_lo, 1.0)
        self.anchor_strength = anchor_strength
        self._last_frets: tuple[int, ...] = ()
        # SOFT cache: pitch -> (fret, last_call_idx). Used as an *advisory*:
        # if the freshly-binned fret is within ±1 of the cached fret, snap
        # to the cached value (kills basic-pitch ±1 semitone jitter on the
        # same note). Otherwise honour the fresh bin and update the cache.
        # This preserves real movement (riffs, runs, key changes) while
        # still removing the cross-fret jumping that hurt v10/v11/v12.
        self._pitch_fret_cache: dict[int, tuple[int, int]] = {}
        self._call_idx: int = 0
        self._last_call_t: float | None = None
        # Cache TTL in calls: entries older than this are forgotten so the
        # mapper adapts as the song's register shifts (verse → solo).
        self._cache_ttl_calls: int = 32
        # Section-break gap in seconds: a long silence resets the cache so
        # the next section starts fresh (a guitar solo following a quiet
        # bridge no longer inherits verse fret bindings).
        self._section_gap_s: float = 1.5

    def note_time(self, t_seconds: float) -> None:
        """Inform the mapper of the current onset time. If the time gap
        since the previous onset exceeds the section threshold, the cache
        is cleared so the next section maps from a clean slate."""
        if self._last_call_t is not None and (t_seconds - self._last_call_t) > self._section_gap_s:
            self._pitch_fret_cache.clear()
        self._last_call_t = t_seconds

    def _pitch_to_fret(self, pitch: int) -> int:
        """Linear bin into 5 frets with soft, time-aware caching.

        Algorithm:
          1. Compute the fresh linear bin for this pitch.
          2. Look up the cache for pitch and pitch±1 (still ±1 tolerant —
             an upstroke is often transcribed at P±1).
          3. If cache hit and |fresh - cached| <= 1 → return cached
             (snap nearby, kill jitter).
          4. Otherwise return fresh and update cache (allow real movement).
          5. Cache entries older than `_cache_ttl_calls` are ignored.
        """
        p = int(pitch)
        self._call_idx += 1
        norm = (pitch - self.p_lo) / self.p_range
        norm = max(0.0, min(1.0, norm))
        fresh = int(round(norm * 4))
        # Look up cache (±1 semitone tolerant), respecting TTL
        for dp in (0, -1, 1):
            entry = self._pitch_fret_cache.get(p + dp)
            if entry is None:
                continue
            cached_fret, when = entry
            if (self._call_idx - when) > self._cache_ttl_calls:
                continue
            if abs(fresh - cached_fret) <= 1:
                # Snap to cached, refresh timestamp
                self._pitch_fret_cache[p] = (cached_fret, self._call_idx)
                return cached_fret
            else:
                # Real movement — break the cache for this pitch
                break
        # Use fresh value and (re)cache
        self._pitch_fret_cache[p] = (fresh, self._call_idx)
        return fresh

    def map(self, pitches: list[int]) -> tuple[int, ...]:
        if not pitches:
            return self._last_frets if self._last_frets else (0,)

        sorted_p = sorted(set(pitches))

        # ─── Power-chord shortcut ────────────────────────────────────────
        # 2 pitches separated by perfect 5th (7 st) or octave (12 st) →
        # adjacent buttons rooted at the lower pitch's natural bin.
        if len(sorted_p) == 2:
            interval = sorted_p[1] - sorted_p[0]
            if interval in (7, 12):
                root_fret = self._pitch_to_fret(sorted_p[0])
                # adjacent button (cap at orange)
                second = min(4, root_fret + 1)
                if second == root_fret:  # at edge, use prev
                    second = max(0, root_fret - 1)
                self._last_frets = tuple(sorted({root_fret, second}))
                return self._last_frets

        # ─── 3-pitch power chord with octave doubling ────────────────────
        # (root, root+7, root+12) → 2 adjacent buttons (treat as power chord)
        if len(sorted_p) >= 3:
            ints = [sorted_p[i] - sorted_p[0] for i in range(1, len(sorted_p))]
            if 7 in ints and 12 in ints:
                root_fret = self._pitch_to_fret(sorted_p[0])
                second = min(4, root_fret + 1)
                if second == root_fret:
                    second = max(0, root_fret - 1)
                self._last_frets = tuple(sorted({root_fret, second}))
                return self._last_frets

        # ─── General mapping: bin each pitch, dedupe ─────────────────────
        frets = sorted({self._pitch_to_fret(p) for p in sorted_p})

        # If chord collapsed to 1 fret but >1 pitches present, spread across
        # neighbors so it feels like a chord
        if len(frets) == 1 and len(sorted_p) >= 2:
            f0 = frets[0]
            extras = []
            if f0 < 4:
                extras.append(f0 + 1)
            if f0 > 0 and len(sorted_p) >= 3:
                extras.append(f0 - 1)
            frets = sorted(set(frets + extras[: min(len(sorted_p) - 1, 2)]))

        self._last_frets = tuple(frets)
        return self._last_frets


# ─────────────────────────── Snap & merge ───────────────────────────────────
def snap_pitches_to_onsets(
    onset_times: np.ndarray,
    notes: list[PitchNote],
    snap_window_s: float = 0.075,
) -> list[list[PitchNote]]:
    """For each onset, return the list of basic-pitch notes that *start* near it.

    A note matches an onset if |note.t_start - onset_time| <= snap_window_s.
    Notes are not reused — each goes to the nearest onset within window.
    """
    if len(onset_times) == 0:
        return []
    onset_arr = np.asarray(onset_times, dtype=np.float64)
    buckets: list[list[PitchNote]] = [[] for _ in onset_times]

    for n in notes:
        # Find nearest onset
        idx = int(np.searchsorted(onset_arr, n.t_start))
        candidates = []
        if idx < len(onset_arr):
            candidates.append((idx, abs(onset_arr[idx] - n.t_start)))
        if idx > 0:
            candidates.append((idx - 1, abs(onset_arr[idx - 1] - n.t_start)))
        candidates = [(i, d) for i, d in candidates if d <= snap_window_s]
        if not candidates:
            continue
        best_i = min(candidates, key=lambda x: x[1])[0]
        buckets[best_i].append(n)

    return buckets


# ─────────────────────────── Learned fret mapper bridge ─────────────────────
_LEARNED_MAPPER_CACHE: dict = {}


def _learned_fret_mapping(
    onset_times: np.ndarray,
    buckets: list[list[PitchNote]],
) -> list[tuple[int, ...]]:
    """Map (CRNN onsets, BP pitch buckets) → fret subsets via learned MLP+Viterbi.

    Uses the same featurizer as scripts/guitar_basicpitch.py so the v4 MLP
    weights apply directly. Falls back to a flat (0,) sequence if anything
    in the learned pipeline fails — caller still has the rule-based mapper
    available as backup.
    """
    import os as _os
    import sys as _sys
    from pathlib import Path as _Path
    _scripts = str(_Path(__file__).resolve().parent.parent.parent / "scripts")
    if _scripts not in _sys.path:
        _sys.path.insert(0, _scripts)
    from guitar_basicpitch import _LearnedMapper, _featurize_onsets_for_learned  # noqa
    from viterbi_fret_decode import viterbi_decode  # noqa

    ckpt = _os.environ.get(
        "STRUM_GUITAR_LEARNED_CKPT", "checkpoints/fret_mapper_v4.pt"
    )
    if ckpt not in _LEARNED_MAPPER_CACHE:
        _LEARNED_MAPPER_CACHE[ckpt] = _LearnedMapper(ckpt)
    mapper = _LEARNED_MAPPER_CACHE[ckpt]

    # Build the onset list expected by the featurizer.
    onsets: list[dict] = []
    for t, bucket in zip(onset_times, buckets):
        if not bucket:
            onsets.append({"time": float(t), "pitches": [],
                           "amps": [], "durations": []})
            continue
        onsets.append({
            "time": float(t),
            "pitches": [int(n.midi) for n in bucket],
            "amps": [float(n.amplitude) for n in bucket],
            "durations": [float(getattr(n, "duration", n.t_end - n.t_start))
                          for n in bucket],
        })

    if not onsets:
        return []

    P = mapper.predict_proba(onsets)
    times = np.asarray([o["time"] for o in onsets], dtype=np.float32)
    use_vit = _os.environ.get("STRUM_GUITAR_VITERBI", "1") != "0"
    if use_vit:
        seq = viterbi_decode(
            P, times=times,
            fret_change_w=float(_os.environ.get("STRUM_GUITAR_VIT_FC", "0.20")),
            center_jump_w=float(_os.environ.get("STRUM_GUITAR_VIT_CJ", "0.03")),
            chord_switch_w=float(_os.environ.get("STRUM_GUITAR_VIT_CS", "0.10")),
            same_state_bonus=float(_os.environ.get("STRUM_GUITAR_VIT_SB", "0.10")),
            time_decay_tau=float(_os.environ.get("STRUM_GUITAR_VIT_TAU", "0.8")),
        )
    else:
        thr = float(_os.environ.get("STRUM_GUITAR_LEARNED_THR", "0.5"))
        seq = []
        for row in P:
            f = tuple(i for i, v in enumerate(row) if v >= thr)
            if not f:
                f = (int(row.argmax()),)
            seq.append(f)
    # Empty buckets: drop to fallback single open string so downstream
    # filter_empty preserves expected count (the rule mapper would have
    # produced (0,) too via _last_frets fallback).
    cleaned: list[tuple[int, ...]] = []
    for ons, frets in zip(onsets, seq):
        if not ons["pitches"]:
            cleaned.append((0,))
        else:
            cleaned.append(tuple(sorted(int(f) for f in frets)))
    return cleaned


# ─────────────────────────── Dominant-voice filter ──────────────────────────
def filter_dominant_voice(
    buckets: list[list[PitchNote]],
    all_notes: list[PitchNote],
    lead_percentile: float = 75.0,
    lead_min_fraction: float = 0.05,
    lead_separation_semitones: int = 5,
    lead_amp_ratio: float = 0.85,
    max_lead_notes: int = 2,
) -> list[list[PitchNote]]:
    """Pick the perceptually-dominant voice at each onset.

    Tuned conservatively: only collapses to lead when there's CLEAR evidence
    of a true lead overdub (high pitch, large pitch gap, comparable amplitude
    to rhythm). Otherwise the chord/rhythm bucket is kept intact so chord
    density isn't destroyed.

    Strategy:
      * lead_threshold = song-wide 80th percentile pitch.
      * If <8% of notes are in lead register → no lead voice; pass through.
      * For each onset bucket:
          - lead_notes = bucket notes ≥ lead_threshold
          - rhythm_notes = bucket notes < lead_threshold
          - Only pick lead IF: top_lead ≥ top_rhythm + 7 semitones AND
            top_lead amplitude ≥ lead_amp_ratio × max(rhythm amplitudes).
            This rules out simple chord voicings whose top note happens to
            sit in the lead register.
          - Else pass the full bucket through (preserve chords).
    """
    if not all_notes:
        return buckets
    arr = np.array([n.midi for n in all_notes], dtype=np.float32)
    lead_threshold = float(np.percentile(arr, lead_percentile))
    lead_fraction = float(np.mean(arr >= lead_threshold))
    if lead_fraction < lead_min_fraction:
        return buckets

    # Pass 1: per-onset analysis. For each bucket compute (lead_notes,
        # has_qualifying_lead) where lead qualifies if pitch-gap OR amp-ratio
        # gate passes (loose: catches leads even at sub-octave separations).
    n_onsets = len(buckets)
    lead_per_onset: list[list[PitchNote]] = []
    has_lead: list[bool] = []
    for bucket in buckets:
        if not bucket:
            lead_per_onset.append([])
            has_lead.append(False)
            continue
        lead_notes = sorted(
            [n for n in bucket if n.midi >= lead_threshold],
            key=lambda n: -n.midi,
        )
        rhythm_notes = [n for n in bucket if n.midi < lead_threshold]
        if not lead_notes:
            lead_per_onset.append([])
            has_lead.append(False)
            continue
        if not rhythm_notes:
            # Pure lead bucket
            lead_per_onset.append(lead_notes[:max_lead_notes])
            has_lead.append(True)
            continue
        top_lead = lead_notes[0]
        top_rhythm_midi = max(n.midi for n in rhythm_notes)
        max_rhythm_amp = max(n.amplitude for n in rhythm_notes)
        pitch_gap_ok = (top_lead.midi - top_rhythm_midi) >= lead_separation_semitones
        amp_ok = top_lead.amplitude >= lead_amp_ratio * max_rhythm_amp
        if pitch_gap_ok or amp_ok:
            lead_per_onset.append(lead_notes[:max_lead_notes])
            has_lead.append(True)
        else:
            lead_per_onset.append([])
            has_lead.append(False)

    # Pass 2: section detection. An onset is in a "lead-dominant section"
    # if a window of ±8 onsets around it contains ≥3 lead-bearing onsets
    # AND ≥35% of non-empty buckets in the window have lead. This anchors
    # us to lead during sections like solos / lead lines, where rhythm
    # rests for a beat would otherwise cause us to flip back to rhythm
    # and produce overcharted fret-sweep artifacts.
    window = 8
    in_lead_section: list[bool] = [False] * n_onsets
    for i in range(n_onsets):
        lo = max(0, i - window)
        hi = min(n_onsets, i + window + 1)
        leads = sum(1 for j in range(lo, hi) if has_lead[j])
        non_empty = sum(1 for j in range(lo, hi) if buckets[j])
        if leads >= 3 and non_empty > 0 and leads / non_empty >= 0.35:
            in_lead_section[i] = True

    # Pass 3: emit. Behaviour by region:
    #   * lead-dominant section + this onset has lead → chart lead only
    #   * lead-dominant section + no lead this onset → SKIP (empty bucket)
    #     Prevents "flip to rhythm on lead rests" → the source of the
    #     fret-sweeping during dual-guitar sections.
    #   * outside lead section, this onset has clear lead → chart lead only
    #   * otherwise → pass bucket through (preserve chords/rhythm)
    out: list[list[PitchNote]] = []
    for i, bucket in enumerate(buckets):
        if in_lead_section[i]:
            if has_lead[i]:
                out.append(lead_per_onset[i])
            else:
                out.append([])  # drop onset; lead is resting
        else:
            if has_lead[i] and lead_per_onset[i]:
                out.append(lead_per_onset[i])
            else:
                out.append(bucket)
    return out


# ─────────────────────────── Main charter ───────────────────────────────────
class GuitarHybridV2Charter:
    """V2 onset CRNN + basic-pitch + rule-based pitch→fret mapping."""

    def __init__(
        self,
        onset_ckpt: Path,
        config_path: Path = ROOT / "configs" / "guitar_v2.yaml",
        device: Optional[str] = None,
    ):
        self.cfg = yaml.safe_load(open(config_path))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # V2 onset CRNN
        ocfg = OnsetCRNNConfig(**self.cfg["onset"]["model"])
        self.onset = GuitarOnsetCRNN(ocfg).to(self.device).eval()
        ck = torch.load(onset_ckpt, map_location=self.device, weights_only=False)
        self.onset.load_state_dict(ck["state_dict"])
        self.onset_meta = {"epoch": ck.get("epoch"), "val_f1": ck.get("val_f1")}

        inf = self.cfg["onset"]["inference"]
        self.default_onset_thr = float(inf["peak_threshold"])
        self.default_min_dist_frames = int(inf["peak_min_distance_frames"])

        # Warm up basic-pitch model
        _get_basic_pitch_model()

    # ─── Stage 1: V2 onset CRNN ────────────────────────────────────────────
    @torch.inference_mode()
    def detect_onsets(
        self,
        audio: np.ndarray,
        threshold: float | None = None,
        latency_offset_s: float | None = None,
    ) -> np.ndarray:
        """Detect onsets and apply attack-latency correction.

        Mel-spectrogram CNNs detect at the energy peak which lags the actual
        attack transient by ~20-30ms (one mel frame at hop=512/22050). We
        subtract `latency_offset_s` so events line up with note attacks in
        playback (matches drum/vocal pipelines which use librosa onset_strength
        + per-attack refinement).

        Override via env var STRUM_GUITAR_LATENCY_MS (e.g. "0", "25", "-15").
        """
        if latency_offset_s is None:
            import os as _os
            # Diagnostic on existing charts (May 2026): chart fires ~50ms
            # EARLIER than audio onsets. Default flipped: ADD 25ms instead of
            # subtracting (env value is added to peak time).
            latency_offset_s = -float(_os.environ.get(
                "STRUM_GUITAR_LATENCY_MS", "25")) / 1000.0
        thr = threshold if threshold is not None else self.default_onset_thr
        # Optional onset-threshold env override for sweep tuning
        import os as _os2
        env_thr = _os2.environ.get("STRUM_GUITAR_PEAK_THR")
        if env_thr:
            thr = float(env_thr)
        log_mel = pgw.compute_log_mel(audio)                   # (M, T)
        x = log_mel.unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,M,T)
        logits = self.onset(x).squeeze(0)
        probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        from src.inference.guitar_neural import GuitarNeuralCharter
        peaks = GuitarNeuralCharter.peak_pick(probs, thr, self.default_min_dist_frames)
        frame_s = pgw.HOP_LENGTH / pgw.SAMPLE_RATE
        times = peaks * frame_s - latency_offset_s
        return np.clip(times, 0.0, None)

    # ─── Full pipeline ─────────────────────────────────────────────────────
    def transcribe(
        self,
        audio: np.ndarray,
        audio_path: Path,
        onset_threshold: float | None = None,
        snap_window_s: float = 0.075,
        min_pitch_amplitude: float = 0.3,
        sustain_min_duration_s: float = 0.40,
        max_chord_size: int = 3,
        harmonic_collapse: bool = False,
    ) -> list[GuitarEvent]:
        """Full hybrid pipeline.

        Args:
            audio: mono float32 @ 22050 Hz (used for V2 onset model)
            audio_path: path to original audio file (used by basic-pitch)
            harmonic_collapse: if True, collapse octave-doubled pitches at each
                onset to just the lowest. Use for bass where basic-pitch often
                detects root+octave harmonics as separate "notes".
        """
        # Stage 1: onsets
        onset_times = self.detect_onsets(audio, threshold=onset_threshold)

        # Stage 2: pitches
        notes = basic_pitch_predict(audio_path, min_amplitude=min_pitch_amplitude)

        # Stage 3: snap
        buckets = snap_pitches_to_onsets(onset_times, notes, snap_window_s)

        # Stage 3a: harmonic collapse (bass): if a bucket contains pitches
        # P and P+12 (or P+24), keep only P. basic-pitch routinely detects
        # bass overtones as separate "notes", which produced false octave
        # double-stops on bass charts.
        if harmonic_collapse:
            collapsed: list[list[PitchNote]] = []
            for bucket in buckets:
                if len(bucket) <= 1:
                    collapsed.append(bucket)
                    continue
                by_pitch: dict[int, PitchNote] = {}
                for n in bucket:
                    if n.midi not in by_pitch or n.amplitude > by_pitch[n.midi].amplitude:
                        by_pitch[n.midi] = n
                pitches_sorted = sorted(by_pitch.keys())
                roots: list[int] = []
                for p in pitches_sorted:
                    if any((p - r) % 12 == 0 and p != r for r in roots):
                        continue  # octave of an existing root
                    roots.append(p)
                collapsed.append([by_pitch[p] for p in roots])
            buckets = collapsed

        # Stage 3b: dominant-voice filter — at onsets where a lead overdub
        # sits clearly above the rhythm chord, chart only the lead. Songs
        # with no lead-register activity are passed through unchanged.
        import os as _os
        # Snapshot the pre-filter buckets so the learned mapper sees the
        # full pitch context (PC-histogram + chord-shape signal). The
        # voice filter strips chord pitches down to 1-2 notes which causes
        # the MLP to under-predict chords.
        buckets_full = [list(b) for b in buckets]
        if _os.environ.get("STRUM_GUITAR_VOICE_FILTER", "1") == "1":
            buckets = filter_dominant_voice(buckets, notes)

        # Stage 4: build mapper from all transcribed pitches.
        # STRUM_GUITAR_FRET_MAPPER=learned uses the trained MLP fret-mapper
        # (checkpoints/fret_mapper_v4.pt) + Viterbi smoothing in place of
        # the rule-based PitchToFretMapper. The mapper consumes the same
        # CRNN onsets the rule path uses, so onset count is preserved.
        learned_frets: list[tuple[int, ...]] | None = None
        if _os.environ.get("STRUM_GUITAR_FRET_MAPPER", "learned").lower() == "learned" and not harmonic_collapse:
            try:
                learned_frets = _learned_fret_mapping(onset_times, buckets_full)
            except Exception as _e:
                # Honour the docstring contract: any failure in the learned
                # pipeline (missing module, bad ckpt, shape mismatch, ...)
                # silently falls back to the rule-based PitchToFretMapper
                # below so PART GUITAR is still produced.
                log.warning(f"guitar learned fret mapper unavailable, falling back to rule mapper ({_e})")
                learned_frets = None
        mapper = PitchToFretMapper([n.midi for n in notes])

        # Stage 5: assemble events
        events: list[GuitarEvent] = []
        for ev_idx, (t, bucket) in enumerate(zip(onset_times, buckets)):
            mapper.note_time(float(t))
            pitches = [n.midi for n in bucket]
            if learned_frets is not None:
                frets = learned_frets[ev_idx]
            else:
                frets = mapper.map(pitches)
            if not frets:
                continue
            # Cap chord size: sections with multiple guitars cause basic-pitch
            # to detect 4-5 simultaneous pitches, leading to over-charted full
            # chords. Real CH/YARG charts rarely exceed 3-button chords.
            # Keep the frets corresponding to the LOUDEST pitches.
            if len(frets) > max_chord_size and bucket:
                # Sort bucket by amplitude desc and take fret bins for top-N
                sorted_bucket = sorted(bucket, key=lambda n: -n.amplitude)
                kept_pitches = [n.midi for n in sorted_bucket[:max_chord_size + 1]]
                trimmed = mapper.map(kept_pitches)
                if trimmed:
                    frets = trimmed[:max_chord_size]
                else:
                    frets = frets[:max_chord_size]
            # Confidence proxy: mean amplitude or 0.5 if empty
            amp = float(np.mean([n.amplitude for n in bucket])) if bucket else 0.5
            # Sustain check (longest constituent note)
            max_dur = max((n.duration for n in bucket), default=0.0)
            events.append(GuitarEvent(
                time_sec=float(t),
                frets=frets,
                onset_prob=amp,
                fret_probs=tuple([0.0] * 5),  # not used downstream
            ))
            # stash sustain duration on the event for export
            events[-1].__dict__["sustain_duration_s"] = (
                max_dur if max_dur >= sustain_min_duration_s else 0.0
            )
        return events


# ─────────────────────────── MIDI export with sustains ──────────────────────
def export_events_to_midi_with_sustain(
    events: list[GuitarEvent],
    output_path: Path,
    tempo_bpm: float = 120.0,
    short_note_sec: float = 0.05,
    track_name: str = "PART GUITAR",
) -> None:
    """Like export_events_to_midi but uses per-event sustain_duration_s if present."""
    import mido

    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name=track_name))
    tempo_us = int(60_000_000 / tempo_bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo_us))

    tpb = mid.ticks_per_beat
    sec_to_tick = lambda s: int(round(s * tempo_bpm / 60.0 * tpb))

    msgs: list[tuple[int, bool, int]] = []
    for ev in events:
        sustain = float(getattr(ev, "sustain_duration_s", 0.0) or 0.0)
        dur = sustain if sustain > 0 else short_note_sec
        on_t = sec_to_tick(ev.time_sec)
        off_t = sec_to_tick(ev.time_sec + dur)
        for f in ev.frets:
            if f not in FRET_TO_MIDI:
                continue
            note = FRET_TO_MIDI[f]
            msgs.append((on_t, True, note))
            msgs.append((off_t, False, note))
    msgs.sort(key=lambda x: (x[0], x[1]))   # off before on at same tick

    last_t = 0
    for tick, is_on, note in msgs:
        delta = max(0, tick - last_t)
        track.append(mido.Message(
            "note_on" if is_on else "note_off",
            note=note, velocity=100 if is_on else 0, time=delta,
        ))
        last_t = tick

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(output_path))


# ─────────────────────────── batch_pipeline adapter ─────────────────────────
_HYBRID_CHARTER_CACHE: dict[str, "GuitarHybridV2Charter"] = {}
_DEFAULT_HYBRID_ONSET_CKPT = get_active_bundle().checkpoint(
    "guitar.onset", ROOT / "checkpoints" / "guitar_v2" / "guitar_v2_onset" / "best.pt"
)
assert _DEFAULT_HYBRID_ONSET_CKPT is not None


def _get_hybrid_charter(device: str | None = None) -> "GuitarHybridV2Charter":
    key = device or "auto"
    ch = _HYBRID_CHARTER_CACHE.get(key)
    if ch is None:
        ch = GuitarHybridV2Charter(onset_ckpt=_DEFAULT_HYBRID_ONSET_CKPT, device=device)
        _HYBRID_CHARTER_CACHE[key] = ch
    return ch


def transcribe_guitar_hybrid(
    audio_path: "Path | str",
    tempo_bpm: float = 0.0,
    is_bass: bool = False,
    onset_threshold: float | None = None,
    device: str | None = None,
    max_chord_size: int | None = None,
    harmonic_collapse: bool | None = None,
):
    """Hybrid V2 (V2 onset CRNN + basic-pitch + rule pitch→fret) → GuitarChart.

    Drop-in replacement for transcribe_guitar_neural; same return type.

    For bass, defaults flip: max_chord_size=1 (singles only — real bass chords
    are rare and basic-pitch's polyphonic detection tends to false-positive
    on harmonics) and harmonic_collapse=True (collapse octave overtones).
    """
    import librosa as _lr
    from src.inference.guitar_bass import GuitarChart, GuitarNote, GuitarChord

    if max_chord_size is None:
        max_chord_size = 1 if is_bass else 3
    if harmonic_collapse is None:
        harmonic_collapse = bool(is_bass)

    audio_path = Path(audio_path)
    audio, sr = _lr.load(str(audio_path), sr=22050, mono=True)

    # RMS normalization — Demucs stems vary ~200× in level (some bass stems
    # come out at -80dB, far below the level the onset CRNN was trained on,
    # which yields zero peaks above any reasonable threshold). Normalize to
    # a target RMS so onset detection is stem-loudness-invariant. We do this
    # on a copy fed to the model only; basic-pitch is run on the file path
    # directly (it's amplitude-relative internally).
    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
    target_rms = 0.05
    if rms > 1e-8 and rms < target_rms:
        gain = min(target_rms / rms, 500.0)  # cap at +54dB
        audio = (audio * gain).astype(np.float32)
        # Soft-clip to prevent integer overflow / NaN downstream
        peak = float(np.abs(audio).max())
        if peak > 1.0:
            audio = (audio / peak).astype(np.float32)

    if tempo_bpm <= 0:
        try:
            tempo_fn = getattr(_lr.beat, "tempo", None) or _lr.feature.rhythm.tempo
            tempo_bpm = float(np.atleast_1d(tempo_fn(y=audio, sr=sr))[0])
        except Exception:
            tempo_bpm = 120.0

    ch = _get_hybrid_charter(device=device)
    events = ch.transcribe(
        audio, audio_path,
        onset_threshold=onset_threshold,
        max_chord_size=max_chord_size,
        harmonic_collapse=harmonic_collapse,
    )

    chart = GuitarChart(
        tempo_bpm=float(tempo_bpm),
        instrument="bass" if is_bass else "guitar",
    )
    for ev in events:
        t_ms = ev.time_sec * 1000.0
        sustain_s = float(getattr(ev, "__dict__", {}).get("sustain_duration_s", 0.0) or 0.0)
        dur_ms = sustain_s * 1000.0 if sustain_s > 0 else 100.0
        if len(ev.frets) >= 2:
            chart.chords.append(GuitarChord(
                time_ms=t_ms, frets=list(ev.frets), duration_ms=dur_ms,
            ))
        elif len(ev.frets) == 1:
            chart.notes.append(GuitarNote(
                time_ms=t_ms, fret=int(ev.frets[0]), duration_ms=dur_ms,
            ))
    return chart


__all__ = [
    "GuitarHybridV2Charter",
    "PitchToFretMapper",
    "PitchNote",
    "basic_pitch_predict",
    "snap_pitches_to_onsets",
    "export_events_to_midi_with_sustain",
    "transcribe_guitar_hybrid",
]
