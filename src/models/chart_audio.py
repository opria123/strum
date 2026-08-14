"""Small, deterministic audio features for chart-transform conditioning.

Audio is decoded locally with FFmpeg and is deliberately not copied into a
dataset or model bundle.  The first feature set is intentionally modest: it
gives a chart-transform model aligned local energy and transient information
at every source-chart event.  It is a bridge toward sequence models, not a
replacement for stem-aware transcription.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

AUDIO_FEATURE_MODE = "rms_onset_v1"
AUDIO_FEATURE_DIM = 2
MAX_AUDIO_SAMPLE_RATE = 48_000
MAX_AUDIO_DURATION_SECONDS = 1_200.0
MAX_AUDIO_PCM_BYTES = 256 * 1024 * 1024


class AudioFeatureError(ValueError):
    """Raised when a local song cannot be safely decoded for conditioning."""


def event_audio_features(
    audio_path: Path,
    event_times_ms: list[float],
    *,
    sample_rate: int,
    window_ms: float,
    max_duration_seconds: float,
) -> list[list[float]]:
    """Return normalized local energy/transient features for chart events.

    The decoder uses an argv list rather than a shell and emits bounded mono
    PCM, so a bad path or unexpectedly long source cannot write files or
    consume unbounded memory.
    """
    if sample_rate < 1 or window_ms <= 0 or max_duration_seconds <= 0:
        raise AudioFeatureError("audio sample rate, window, and duration limit must be positive")
    if sample_rate > MAX_AUDIO_SAMPLE_RATE or max_duration_seconds > MAX_AUDIO_DURATION_SECONDS:
        raise AudioFeatureError("audio sample rate or duration limit exceeds the supported maximum")
    if sample_rate * max_duration_seconds * 4 > MAX_AUDIO_PCM_BYTES:
        raise AudioFeatureError("configured audio decode exceeds the PCM memory limit")
    if not audio_path.is_file():
        raise AudioFeatureError("audio asset is not a file")
    if event_times_ms and max(event_times_ms) > max_duration_seconds * 1000:
        raise AudioFeatureError("chart event exceeds configured audio duration limit")
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-t",
        str(max_duration_seconds),
        "-i",
        str(audio_path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    try:
        decoded = subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as error:
        raise AudioFeatureError(
            "FFmpeg is required for audio-conditioned chart training"
        ) from error
    except subprocess.CalledProcessError as error:
        raise AudioFeatureError("audio asset could not be decoded") from error

    samples = np.frombuffer(decoded.stdout, dtype="<f4").copy()
    if not samples.size:
        raise AudioFeatureError("audio asset has no decodable samples")
    samples = np.nan_to_num(samples, copy=False)
    if event_times_ms and max(event_times_ms) * sample_rate / 1000.0 >= samples.size:
        raise AudioFeatureError("audio asset ends before the source chart")
    global_rms = max(float(np.sqrt(np.mean(np.square(samples)))), 1e-6)
    half_window = max(1, round(sample_rate * window_ms / 1000.0))
    features: list[list[float]] = []
    for time_ms in event_times_ms:
        center = round(time_ms * sample_rate / 1000.0)
        before = samples[max(0, center - half_window) : max(0, center)]
        after = samples[max(0, center) : min(samples.size, center + half_window)]
        before_rms = _rms(before)
        after_rms = _rms(after)
        features.append(
            [
                float(np.log1p(after_rms / global_rms)),
                float(np.tanh((after_rms - before_rms) / global_rms)),
            ]
        )
    return features


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
