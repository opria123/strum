"""End-to-end neural guitar inference (V2 two-stage).

Loads:
  * GuitarOnsetCRNN  → per-frame onset probabilities
  * GuitarFretClassifier → per-onset 5-bit multi-label fret prediction

Produces a list of (time_sec, frets:set[int]) events from raw audio.

Audio path → mono 22050 Hz → log-mel (matches preprocess_guitar_windows.py
exactly) → onset CRNN → peak-pick → window slice → fret classifier.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

# Project imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.models.guitar_v1 import (  # noqa: E402
    GuitarOnsetCRNN, GuitarFretClassifier,
    OnsetCRNNConfig, FretClassifierConfig,
)
from src.model_bundle import get_active_bundle  # noqa: E402
import preprocess_guitar_windows as pgw  # noqa: E402

log = logging.getLogger("guitar_neural")


# ─────────────────────────── Result types ───────────────────────────────────
@dataclass
class GuitarEvent:
    time_sec: float
    frets: tuple[int, ...]   # sorted, may be empty (open note)
    onset_prob: float
    fret_probs: tuple[float, ...]  # per-bit sigmoid, length 5

    @property
    def is_chord(self) -> bool:
        return len(self.frets) >= 2


# ─────────────────────────── Inference engine ───────────────────────────────
class GuitarNeuralCharter:
    """Two-stage neural guitar transcriber."""

    def __init__(
        self,
        onset_ckpt: Path,
        fret_ckpt: Path,
        config_path: Path = ROOT / "configs" / "guitar_v2.yaml",
        device: str | None = None,
    ):
        self.cfg = yaml.safe_load(open(config_path))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Stage 1
        ocfg = OnsetCRNNConfig(**self.cfg["onset"]["model"])
        self.onset = GuitarOnsetCRNN(ocfg).to(self.device).eval()
        ck = torch.load(onset_ckpt, map_location=self.device, weights_only=False)
        self.onset.load_state_dict(ck["state_dict"])
        self.onset_meta = {"epoch": ck.get("epoch"), "val_f1": ck.get("val_f1")}

        # Stage 2
        fcfg = FretClassifierConfig(**self.cfg["fret"]["model"])
        self.fret = GuitarFretClassifier(fcfg).to(self.device).eval()
        ck = torch.load(fret_ckpt, map_location=self.device, weights_only=False)
        self.fret.load_state_dict(ck["state_dict"])
        self.fret_meta = {"epoch": ck.get("epoch"), "val_f1": ck.get("val_f1")}

        # Inference hyperparams (config defaults; override at call time)
        inf = self.cfg["onset"]["inference"]
        self.default_onset_thr = float(inf["peak_threshold"])
        self.default_min_dist_frames = int(inf["peak_min_distance_frames"])
        self.default_fret_thr = 0.5

    # ─── Mel ────────────────────────────────────────────────────────────────
    def compute_logmel(self, audio: np.ndarray) -> torch.Tensor:
        """Match preprocess_guitar_windows.compute_log_mel exactly: log(mel+1e-8).

        Returns (n_mels, T) float tensor on CPU.
        """
        return pgw.compute_log_mel(audio)

    # ─── Stage 1 ────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def predict_onset_probs(self, log_mel: torch.Tensor) -> np.ndarray:
        """Run the onset CRNN over the full song. Returns (T,) sigmoid probs."""
        x = log_mel.unsqueeze(0).unsqueeze(0).to(self.device)   # (1,1,M,T)
        logits = self.onset(x).squeeze(0)                       # (T,)
        probs = torch.sigmoid(logits).cpu().numpy()
        return probs.astype(np.float32)

    @staticmethod
    def peak_pick(probs: np.ndarray, threshold: float, min_distance: int) -> np.ndarray:
        """Greedy local-max peak picking, returns frame indices."""
        above = probs >= threshold
        out: list[int] = []
        last = -10**9
        T = len(probs)
        for i in range(1, T - 1):
            if not above[i]:
                continue
            if probs[i] < probs[i - 1] or probs[i] < probs[i + 1]:
                continue
            if i - last < min_distance:
                if out and probs[i] > probs[out[-1]]:
                    out[-1] = i
                    last = i
                continue
            out.append(i)
            last = i
        return np.array(out, dtype=np.int64)

    # ─── Stage 2 ────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def predict_frets(
        self,
        log_mel: torch.Tensor,
        onset_frames: np.ndarray,
        batch_size: int = 256,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Slice per-onset windows from full log-mel and classify.

        Returns:
            fret_probs: (N, 5) sigmoid probabilities
            valid_mask: (N,) bool — False where window fell off the edge
        """
        n_mels, T = log_mel.shape
        win_before_samp = pgw.WIN_BEFORE_SAMP
        win_frames = pgw.WIN_FRAMES
        sample_rate = pgw.SAMPLE_RATE
        hop = pgw.HOP_LENGTH

        N = len(onset_frames)
        if N == 0:
            return np.zeros((0, 5), dtype=np.float32), np.zeros(0, dtype=bool)

        # Build mel patches
        patches = np.zeros((N, n_mels, win_frames), dtype=np.float32)
        valid = np.zeros(N, dtype=bool)
        log_mel_np = log_mel.numpy()
        for i, fr in enumerate(onset_frames):
            # Convert frame → audio sample → window-start frame, mirroring
            # preprocess. fr corresponds to onset @ fr*hop samples.
            center_sample = int(fr) * hop
            fstart = (center_sample - win_before_samp) // hop
            fend = fstart + win_frames
            if fstart < 0 or fend > T:
                continue
            patches[i] = log_mel_np[:, fstart:fend]
            valid[i] = True

        # Batch through fret head
        fret_probs = np.zeros((N, 5), dtype=np.float32)
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            chunk = torch.from_numpy(patches[s:e]).unsqueeze(1).to(self.device)
            out = self.fret(chunk)
            fret_probs[s:e] = torch.sigmoid(out["fret_logits"]).cpu().numpy()

        # Mark invalid rows with -1
        fret_probs[~valid] = -1.0
        return fret_probs, valid

    # ─── Full pipeline ──────────────────────────────────────────────────────
    def transcribe(
        self,
        audio: np.ndarray,
        onset_threshold: Optional[float] = None,
        min_distance_frames: Optional[int] = None,
        fret_threshold: Optional[float] = None,
        fret_thresholds_per_bit: Optional["np.ndarray | list[float]"] = None,
    ) -> list[GuitarEvent]:
        """Full audio → list[GuitarEvent].

        Args:
            audio: mono float32 @ 22050 Hz
            fret_thresholds_per_bit: optional length-5 array; overrides
                fret_threshold when given. Tuned values from
                outputs/guitar_v2/optimal_fret_thresholds.json fix the
                Green-fret under-prediction in the flat-0.5 default.
        """
        thr_on = onset_threshold if onset_threshold is not None else self.default_onset_thr
        min_d = min_distance_frames if min_distance_frames is not None else self.default_min_dist_frames
        if fret_thresholds_per_bit is not None:
            thr_fr_arr = np.asarray(fret_thresholds_per_bit, dtype=np.float32)
            assert thr_fr_arr.shape == (5,), f"expected length-5, got {thr_fr_arr.shape}"
        else:
            scalar = fret_threshold if fret_threshold is not None else self.default_fret_thr
            thr_fr_arr = np.full(5, scalar, dtype=np.float32)

        log_mel = self.compute_logmel(audio)                    # (M, T)
        probs = self.predict_onset_probs(log_mel)               # (T,)
        peaks = self.peak_pick(probs, thr_on, min_d)            # (N,) frame idx

        fret_probs, valid = self.predict_frets(log_mel, peaks)  # (N,5),(N,)

        frame_ms = 1000.0 * pgw.HOP_LENGTH / pgw.SAMPLE_RATE
        events: list[GuitarEvent] = []
        for i, fr in enumerate(peaks):
            if not valid[i]:
                continue
            t_sec = float(fr) * (frame_ms / 1000.0)
            bits = fret_probs[i] >= thr_fr_arr
            frets = tuple(int(b) for b in np.where(bits)[0])
            # Fallback: if no bit crosses threshold, take argmax
            if not frets:
                frets = (int(np.argmax(fret_probs[i])),)
            events.append(GuitarEvent(
                time_sec=t_sec,
                frets=frets,
                onset_prob=float(probs[fr]),
                fret_probs=tuple(float(x) for x in fret_probs[i]),
            ))
        return events


# ─────────────────────────── MIDI export ────────────────────────────────────
FRET_TO_MIDI = {0: 96, 1: 97, 2: 98, 3: 99, 4: 100}


def export_events_to_midi(
    events: list[GuitarEvent],
    output_path: Path,
    tempo_bpm: float = 120.0,
    note_duration_sec: float = 0.05,
    track_name: str = "PART GUITAR",
) -> None:
    """Write events as a Clone Hero / YARG-compatible PART GUITAR MIDI.

    Notes are short (no sustain logic — that's a separate post-process step).
    """
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
        on_t = sec_to_tick(ev.time_sec)
        off_t = sec_to_tick(ev.time_sec + note_duration_sec)
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


# ─────────────────────────── Pipeline adapter ───────────────────────────────
# Adapter so batch_pipeline.py can swap the rule-based guitar_bass.transcribe_*
# call for the neural pipeline without changing its surrounding code.
# Returns the same GuitarChart dataclass used elsewhere in the project.

_DEFAULT_ONSET_CKPT = get_active_bundle().checkpoint(
    "guitar.onset", ROOT / "checkpoints" / "guitar_v2" / "guitar_v2_onset" / "best.pt"
)
assert _DEFAULT_ONSET_CKPT is not None
_DEFAULT_FRET_CKPT  = ROOT / "checkpoints" / "guitar_v2" / "guitar_v2_fret"  / "best.pt"

# Per-bit fret thresholds tuned on the val split via
# scripts/sweep_fret_thresholds.py. Lifts event F1 from 0.170 (flat 0.5)
# to 0.175 and — importantly for playability — unbreaks Green by
# dropping bit-0 threshold to 0.30. Saved at
# outputs/guitar_v2/optimal_fret_thresholds.json.
_DEFAULT_FRET_THRESHOLDS = (0.30, 0.75, 0.475, 0.625, 0.65)
_DEFAULT_ONSET_THRESHOLD = 0.35  # tuned with the per-bit sweep above

# Singleton — Demucs-style; loading the two CRNNs is non-trivial.
_CHARTER_CACHE: dict[str, "GuitarNeuralCharter"] = {}


def _get_charter(device: str | None = None) -> "GuitarNeuralCharter":
    key = device or "auto"
    ch = _CHARTER_CACHE.get(key)
    if ch is None:
        ch = GuitarNeuralCharter(
            onset_ckpt=_DEFAULT_ONSET_CKPT,
            fret_ckpt=_DEFAULT_FRET_CKPT,
            device=device,
        )
        _CHARTER_CACHE[key] = ch
    return ch


def transcribe_guitar_neural(
    audio_path: "Path | str",
    tempo_bpm: float = 0.0,
    confidence_threshold: float | None = None,
    is_bass: bool = False,
    device: str | None = None,
):
    """Neural V2 pipeline returning a GuitarChart (drop-in for guitar_bass).

    Args:
        audio_path: stem .wav (other.wav for guitar; bass.wav for bass).
        tempo_bpm:  song tempo, used only for the returned chart's tempo field.
                    Auto-detected via librosa if 0.
        confidence_threshold: overrides onset peak threshold if not None.
        is_bass:    sets the GuitarChart.instrument field; the model itself
                    was trained on guitar stems so quality on bass is unproven.
                    For now we still call it for parity but expect lower F1.
        device:     "cuda" / "cpu" / None (auto).
    """
    import librosa as _lr
    from src.inference.guitar_bass import GuitarChart, GuitarNote, GuitarChord

    audio_path = Path(audio_path)
    audio, sr = _lr.load(str(audio_path), sr=22050, mono=True)
    if tempo_bpm <= 0:
        try:
            tempo_fn = getattr(_lr.beat, "tempo", None) or _lr.feature.rhythm.tempo
            tempo_bpm = float(np.atleast_1d(tempo_fn(y=audio, sr=sr))[0])
        except Exception:
            tempo_bpm = 120.0

    ch = _get_charter(device=device)
    events = ch.transcribe(
        audio,
        onset_threshold=confidence_threshold if confidence_threshold is not None else _DEFAULT_ONSET_THRESHOLD,
        fret_thresholds_per_bit=_DEFAULT_FRET_THRESHOLDS,
    )

    chart = GuitarChart(
        tempo_bpm=float(tempo_bpm),
        instrument="bass" if is_bass else "guitar",
    )
    for ev in events:
        t_ms = ev.time_sec * 1000.0
        if len(ev.frets) >= 2:
            chart.chords.append(GuitarChord(
                time_ms=t_ms,
                frets=list(ev.frets),
            ))
        elif len(ev.frets) == 1:
            chart.notes.append(GuitarNote(
                time_ms=t_ms,
                fret=int(ev.frets[0]),
            ))
    return chart


__all__ = [
    "GuitarEvent",
    "GuitarNeuralCharter",
    "export_events_to_midi",
    "transcribe_guitar_neural",
    "FRET_TO_MIDI",
]
