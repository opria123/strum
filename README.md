<p align="center">
  <img src="assets/logo.png" alt="STRUM Logo" width="200"/>
</p>

<h1 align="center">STRUM</h1>
<h3 align="center"><b>S</b>pectral <b>T</b>ranscription & <b>R</b>hythm <b>U</b>nderstanding <b>M</b>odel</h3>

<p align="center">
  AI-powered audio-to-chart pipeline for Clone Hero & YARG
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg" alt="PyTorch 2.x"/>
  <img src="https://img.shields.io/badge/CUDA-12.8-76b900.svg" alt="CUDA 12.8"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"/>
</p>

---

STRUM converts any song into a fully playable Clone Hero / YARG chart package — complete with **pro drums**, **guitar**, **bass**, **vocals with lyrics**, and **keys** — all generated from audio alone.

The system uses a two-stage neural drum transcription pipeline, neural onset detection with rule-based fret mapping for guitar/bass, Whisper-powered vocal transcription with pitch tracking, and spectral analysis for keyboard detection. Charts are exported as standard MIDI with four difficulty levels (Expert, Hard, Medium, Easy) and packaged with metadata, album art, and song.ini files ready for play.

## Architecture

```
                              ┌─────────────┐
                              │  Audio File  │
                              │  (WAV/MP3)   │
                              └──────┬───────┘
                                     │
                              ┌──────▼───────┐
                              │   Demucs v4  │
                              │  Separation  │
                              └──────┬───────┘
                                     │
              ┌──────────┬───────────┼───────────┬──────────┐
              ▼          ▼           ▼           ▼          ▼
         ┌────────┐ ┌────────┐ ┌─────────┐ ┌────────┐ ┌────────┐
         │ Drums  │ │ Guitar │ │  Bass   │ │ Vocals │ │  Keys  │
         │  Stem  │ │  Stem  │ │  Stem   │ │  Stem  │ │ Other  │
         └───┬────┘ └───┬────┘ └────┬────┘ └───┬────┘ └───┬────┘
             │          │           │           │          │
             ▼          ▼           ▼           ▼          ▼
        ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
        │Two-Stage│ │ Neural  │ │ Neural  │ │ Whisper │ │Spectral │
        │  CRNN   │ │ Onset + │ │ Onset + │ │ + pYIN  │ │Keyboard │
        │Ensemble │ │Rule Fret│ │Rule Fret│ │ + Align │ │Detector │
        └───┬─────┘ └───┬─────┘ └───┬─────┘ └───┬─────┘ └───┬─────┘
             │          │           │           │          │
             └──────────┴───────────┼───────────┴──────────┘
                                    ▼
                          ┌──────────────────┐
                          │   Chart Export    │
                          │  .mid + song.ini │
                          │  + album art     │
                          │  (4 difficulties)│
                          └──────────────────┘
```

## Instrument Pipelines

### Drums — Two-Stage Neural Ensemble

The drums pipeline is the flagship component, using a two-stage detection-then-classification approach:

1. **Onset Detection** — V14 `TwoStageDrumsCRNN` processes mel spectrograms (128 bins, 22050 Hz) to detect drum hit positions with **93.9% F1 score**
2. **Ensemble Classification** — 6 independently trained `OnsetClassifier` models (V2, V4, V6, V12c, V15, V16) vote on each detected onset to classify across 8 lanes (Kick, Snare, Hi-Hat, Crash, Ride, High Tom, Mid Tom, Floor Tom) achieving **85.2% F1 score**
3. **Spectral Disambiguation** — Spectral centroid analysis resolves tom/cymbal confusion in ambiguous frequency ranges
4. **Post-Processing** — Bidirectional iterative streak smoothing, kick-suppresses-floor-tom logic, rhythmic quantization, and lane conflict resolution

Pro drums are fully supported with separate tom and cymbal markers per the Clone Hero MIDI specification.

### Guitar & Bass — Neural Onset + Polyphonic Pitch + Fret Mapping

Guitar and bass share the same hybrid architecture (`src/inference/guitar_hybrid_v2.py`):

1. **Onset Detection** — `OnsetCRNN` (V2) detects note attacks on the Demucs-separated stem so vocals/drums don't trigger false positives.
2. **Polyphonic Pitch** — [Spotify Basic Pitch](https://github.com/spotify/basic-pitch) transcribes simultaneous notes (chords + single notes), with bass-specific MIDI range overrides (24–67) when running on the bass stem.
3. **Pitch → Fret Mapping** — Rule-based register allocation by default, with an optional learned `PitchToFretMapper` (V4) gated behind `STRUM_FRET_MAPPER=1`.
4. **Section-Aware Density** — An optional `SectionRouter` modulates onset peak thresholds per section (verse/chorus/solo) to prevent over- or under-charting.

### Vocals — Whisper + pYIN Pitch Tracking

1. **Lyric Transcription** — OpenAI Whisper extracts word-level timestamps from the vocal stem
2. **Pitch Detection** — `librosa.pyin` tracks vocal pitch contours at high time resolution
3. **Dynamic Alignment** — Whisper word boundaries are aligned with pitch onsets for accurate note placement
4. **Lyrics Fetching** — Optional synced lyrics from LRCLIB and Lyrics.ovh APIs
5. **Harmony Detection** — Configurable threshold (default 30%) for harmony/backing vocal phrases

### Keys — Spectral Keyboard Detection

1. **Keyboard Detection** — Spectral flatness and harmonic ratio analysis identifies keyboard-active regions in the "other" stem
2. **Note Extraction** — `librosa.onset_detect` + `librosa.piptrack` extract individual key hits
3. **Dual Output** — Both 5-lane simplified and Pro Keys (full piano range) tracks

## Tempo & Grid Alignment

STRUM uses a grid-alignment BPM refinement algorithm that searches ±5 BPM around an initial `librosa` estimate at 0.1 BPM resolution, then snaps the beat-zero phase to the first detected onset. Phase coherence is measured with circular statistics on beat positions vs. onset times. After all transcribers run, every event from every instrument is shifted by the same `phase_offset_ms` and snapped to the 32nd-note grid — with per-lane roll detection to preserve fast double-strokes and tom rolls.

This reduces post-snap grid error to <5 ms on the verified test set across drums, guitar, bass, vocals, and keys.

## Performance

### Component-level (held-out test set)

| Component | Metric | Score |
|-----------|--------|-------|
| Drums — Onset Detection (V14) | Frame F1 | 93.9% |
| Drums — Lane Classification (6-model ensemble) | Per-onset F1 | 85.2% |
| Drums — Best Single Classifier (V12c) | Per-onset F1 | 83.8% |

Evaluated on a held-out test set from 3,299 human-authored Clone Hero/YARG pro drum charts.

### End-to-end vs human-authored game charts (in-envelope benchmark, n=29)

Aggregate per-instrument onset F1 against ground-truth Clone Hero/YARG charts. Songs were sampled from a held-out pool of 3,299 candidates and pre-screened by a single audio-feature operating envelope: median Demucs `htdemucs_6s` drum-stem RMS (1-second windows, 22050 Hz mono) ≥ 0.018. Eval is Expert difficulty, ±100 ms tolerance with a per-song global offset search (±200 ms / 10 ms steps) to neutralize chart-sync conventions.

| Instrument | F1 | Precision | Recall |
|------------|------|-----------|--------|
| Drums      | 83.8% | 82.4% | 85.4% |
| Guitar     | 65.1% | 74.5% | 57.8% |
| Bass       | 69.4% | 65.8% | 73.4% |
| Vocals     | 53.9% | 63.2% | 47.0% |

Reproduce with:

```bash
python scripts/eval_benchmark.py \
  --gt-dir /path/to/charts-gt \
  --pred-dir /path/to/strum-predictions \
  --tolerance-ms 100 \
  --global-offset-search \
  --out benchmark_results.json
```

## Quick Start

### Prerequisites

- Python 3.11+
- PyTorch 2.x with CUDA
- ffmpeg
- ~6 GB disk for model checkpoints

### Installation

```bash
git clone https://github.com/oprialopez/strum.git
cd strum
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Model checkpoints (~6 GB) are not committed; download from the releases page or train locally (see below).

### Generate Charts for a Song

Drop one or more `.wav` / `.mp3` / `.flac` files in a directory and run:

```bash
# Full chart package (all instruments)
python scripts/batch_pipeline.py \
  --songs-dir /path/to/songs/ \
  --output-dir /path/to/output/

# Drums only (faster, no Demucs vocals/keys/bass passes)
python scripts/batch_infer_hybrid.py \
  --songs-dir /path/to/songs/ \
  --output-dir /path/to/output/
```

Each output folder contains `notes.mid`, `song.ini`, the source audio, and album art ready to drop into Clone Hero / YARG.

#### Backend selection (env vars)

| Variable | Values | Default | Effect |
|----------|--------|---------|--------|
| `STRUM_GUITAR_BACKEND` | `hybrid`, `neural`, `rule`, `basicpitch` | `hybrid` | Guitar transcription pipeline |
| `STRUM_BASS_BACKEND`   | `hybrid`, `neural`, `rule`, `basicpitch` | `hybrid` | Bass transcription pipeline |
| `STRUM_FRET_MAPPER`    | `0`, `1` | `0` | Use learned pitch→fret mapper instead of rules |
| `STRUM_V12C_VARIANT`   | `default`, `community` | `default` | Swap drum classifier v12c checkpoint |

### Training Your Own Models

All trainers are plain `python scripts/train_*.py` invocations driven by Hydra-style YAMLs in `configs/`. They expect a manifest of preprocessed windows produced by the matching `preprocess_*` / `build_*` script.

| Model | Preprocess | Train | Config |
|-------|------------|-------|--------|
| Drum onset detector (V14 CRNN) | `preprocess_onset_windows.py` | `python scripts/train_onset_classifier.py` (also trains the onset head) | `configs/drums_v14.yaml` |
| Drum classifier ensemble (V2/V6/V12c/V15/V16) | `preprocess_onset_windows.py` | `python scripts/train_onset_classifier.py --config configs/onset_classifier_v15.yaml` | `configs/onset_classifier*.yaml` |
| Tom-vs-cymbal refinement | (uses Demucs drum stem at train time) | `python scripts/train_tom_refinement.py` | inline |
| Guitar onset CRNN (V1/V2) | `build_guitar_manifest.py` → `preprocess_guitar_windows.py` | `python scripts/train_guitar_v1.py --config configs/guitar_v2.yaml` | `configs/guitar_v1.yaml`, `configs/guitar_v2.yaml` |
| Pitch→fret mapper | `build_mapper_dataset.py` | `python scripts/train_fret_mapper.py` | inline |
| Section classifier (verse/chorus/etc.) | `build_section_labels.py` → `preprocess_section_windows.py` | `python scripts/train_section_classifier.py` | inline |

A typical training session for the drum onset detector looks like:

```bash
python scripts/preprocess_onset_windows.py \
  --manifest /mnt/ml-data/manifest.json \
  --output-dir /mnt/ml-data/onset_windows/

python scripts/train_onset_classifier.py \
  --config configs/onset_classifier_v15.yaml
```

W&B logging is enabled by default; set `WANDB_MODE=offline` to disable.

## Project Structure

```
strum/
├── configs/                          # YAML configs (one per trainable model)
│   ├── drums_v14.yaml                # Two-stage drum onset CRNN
│   ├── onset_classifier_v{6,12_clean,15,16}.yaml  # Drum classifier ensemble
│   ├── guitar_v1.yaml, guitar_v2.yaml             # Guitar onset CRNN
│   ├── inference.yaml, preprocessing.yaml
├── checkpoints/                      # Trained weights (gitignored)
├── scripts/
│   ├── batch_pipeline.py             # ★ Full multi-instrument pipeline (entry point)
│   ├── batch_infer_hybrid.py         # Drums-only production pipeline
│   ├── chart_postprocess.py          # Snap-to-grid + rescue passes + quantization
│   ├── chart_enhancer.py             # Difficulty reduction + lane balancing
│   ├── vocals_charter.py             # Whisper + pYIN vocal transcription
│   ├── keys_charter.py               # Keyboard detection + Pro Keys export
│   ├── guitar_basicpitch.py          # Basic-Pitch guitar backend
│   ├── bass_basicpitch.py            # Basic-Pitch bass backend
│   ├── train_onset_classifier.py     # Drum classifier training
│   ├── train_tom_refinement.py       # Tom-vs-cymbal refinement training
│   ├── train_guitar_v1.py            # Guitar onset CRNN training
│   ├── train_fret_mapper.py          # Learned pitch→fret mapper training
│   ├── train_section_classifier.py   # Section (verse/chorus/etc.) training
│   ├── preprocess_onset_windows.py   # Drum window preprocessing
│   ├── preprocess_guitar_windows.py  # Guitar window preprocessing
│   ├── preprocess_section_windows.py # Section window preprocessing
│   ├── build_guitar_manifest.py      # Guitar dataset manifest builder
│   ├── build_mapper_dataset.py       # Pitch→fret dataset builder
│   └── build_section_labels.py       # Section label builder
├── src/
│   ├── models/
│   │   ├── drums_v13.py              # TwoStageDrumsCRNN architecture (V14 ckpt)
│   │   ├── onset_classifier.py       # 8-lane drum classifier
│   │   ├── onset_classifier_dataset.py, onset_classifier_cached_dataset.py
│   │   ├── drums_v14_dataset.py      # Drum onset dataset w/ bg-mel subtraction
│   │   ├── tom_refinement.py         # Tom-vs-cymbal head
│   │   ├── guitar_v1.py              # Guitar onset CRNN
│   │   ├── section_classifier.py     # Section labeler
│   │   ├── bg_mel.py                 # Background-mel subtraction
│   │   └── common.py
│   ├── inference/
│   │   ├── guitar_hybrid_v2.py       # ★ Production guitar/bass backend
│   │   ├── guitar_neural.py          # Neural-only guitar/bass backend
│   │   ├── guitar_bass.py            # GuitarChart/Note/Chord dataclasses + rule backend
│   │   ├── section_router.py         # Section-aware onset gating
│   │   └── c3_rules.py               # C3 chart rules (5-fret reduction etc.)
│   ├── preprocessing/
│   │   ├── parsers/                  # .mid and .chart parsers
│   │   ├── alignment.py              # Audio-chart alignment
│   │   └── separation.py             # Demucs wrapper
│   ├── export/
│   │   ├── midi.py                   # Pro drums + guitar/bass/keys MIDI export
│   │   └── chart.py                  # .chart format export
│   └── lyrics/                       # LRCLIB + Lyrics.ovh fetcher
├── docs/
│   ├── ARCHITECTURE.md               # Technical specification
│   └── ROADMAP.md                    # Development milestones
└── pyproject.toml
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| ML Framework | PyTorch 2.x |
| Audio Separation | Demucs v4 (HTDemucs) |
| Pitch Detection | librosa pYIN |
| Speech-to-Text | OpenAI Whisper |
| MIDI I/O | mido |
| Experiment Tracking | Weights & Biases |
| Config Management | Hydra |
| Audio Processing | librosa, soundfile |
| CLI | Click + Rich |

## Chart Output Format

STRUM generates standard Clone Hero / YARG compatible chart packages:

```
Song Name/
├── notes.mid          # MIDI chart (480 ticks/beat, 4 difficulty levels)
├── song.ini           # Metadata (artist, title, charter, BPM)
├── song.ogg           # Audio file
└── album.png          # Album art (fetched automatically)
```

Each MIDI contains up to 5 instrument tracks:
- **PART DRUMS** — 5-lane pro drums with cymbal markers (MIDI notes 96-100, tom markers 110-112)
- **PART GUITAR** — 5-fret guitar (MIDI notes 96-100)
- **PART BASS** — 5-fret bass (MIDI notes 96-100)
- **PART VOCALS** — Pitched vocal phrases with lyric events
- **PART KEYS** — 5-lane keys + optional Pro Keys

Four difficulty levels per instrument: Expert, Hard, Medium, Easy (progressive note reduction).

## Custom model bundles

The repository `checkpoints/` layout remains the default. To evaluate or ship
a fine-tuned model without replacing it, set `STRUM_MODEL_BUNDLE` to a
directory containing `strum-model-bundle.json` (or directly to that file).
Each path is relative to the manifest, so a bundle can be copied as one
directory. Components not declared by a bundle keep using the repository
defaults, which makes small targeted experiments safe.

```json
{
  "schema_version": 1,
  "model_id": "drums-v14-finetune-2026-08",
  "compatibility": {
    "manifest_schema": 1,
    "strum_version": ">=0.1.0",
    "strum_revision": "<pinned-strum-git-revision>"
  },
  "components": {
    "drums.v14_onset": {
      "checkpoint": "weights/best.pt",
      "sha256": "<64-character lowercase sha256>"
    },
    "drums.ensemble.v17": {
      "checkpoint": "weights/v17-best.pt",
      "config": "configs/onset_classifier_v17.yaml"
    },
    "guitar.onset": {"checkpoint": "weights/guitar-onset.pt"}
  }
}
```

Validate a candidate without loading model weights, or list valid bundles in
a user model directory:

```bash
python -m src.model_bundle validate /path/to/bundle --check-files --verify-hashes
python -m src.model_bundle list /path/to/models
```

Current production integration recognizes `drums.v14_onset`,
`drums.ensemble.v2` through `drums.ensemble.v17`, and `guitar.onset`.
`compatibility.strum_revision` is optional but recommended for portable
bundles: it records the STRUM Git revision the model was trained against. Set
`STRUM_SOURCE_REVISION` to the source revision pinned by a caller (such as an
editor integration) to enforce it. Otherwise validation reports the declared
revision as unverified, rather than treating source trees without Git metadata
as incompatible.

## Development

Developed on NVIDIA DGX Spark (GB10 GPU, CUDA 12.8). Trained on ~5,000 human-authored pro drum charts from the Clone Hero community.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — Technical specification
- [Roadmap](docs/ROADMAP.md) — Development milestones

## Acknowledgments

- [Demucs](https://github.com/adefossez/demucs) — Audio source separation
- [OpenAI Whisper](https://github.com/openai/whisper) — Speech recognition
- [librosa](https://librosa.org/) — Audio analysis
- Clone Hero / YARG communities — Chart format documentation

## License

MIT
