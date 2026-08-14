#!/usr/bin/env python3
"""
Hybrid inference pipeline: V14 onset detection + onset classifier ensemble.

Stage 1: V14's TwoStageDrumsCRNN onset detector (93.9% F1 for WHERE)
Stage 2: Onset classifier ensemble (85.2% F1 for WHAT)

Pipeline:
  1. Load audio → Demucs separation → drums stem
  2. Run V14 onset detector → frame-level onset probabilities
  3. Peak detection → onset times
  4. Extract dual-resolution mel windows at each onset
  5. Run onset classifier ensemble → per-class probabilities
  6. Apply per-class thresholds → MIDI export

Usage:
    python scripts/batch_infer_hybrid.py
    python scripts/batch_infer_hybrid.py --skip-separation
    python scripts/batch_infer_hybrid.py --onset-threshold 0.35
"""

import argparse
import bisect
from collections import Counter, defaultdict
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from omegaconf import OmegaConf
from scipy.signal import find_peaks
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.drums_v13 import TwoStageDrumsCRNN
from src.models.onset_classifier import OnsetClassifier
from src.models.tom_refinement import TomRefinementCNN
from src.model_bundle import get_active_bundle
from src.preprocessing.parsers.midi_parser import DrumHit, DrumChart, TempoEvent, TimeSignature
from scripts.chart_postprocess import postprocess_chart
from src.export.midi import export_all_difficulties

# Tom/cymbal CNN reclassification (currently disabled in pipeline; experimental
# training script has been removed from the repo). Importing optionally so the
# pipeline still works when the module isn't present.
try:
    from scripts.train_tomcym_cnn import TomCymCNN  # type: ignore
except ModuleNotFoundError:
    TomCymCNN = None  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Per-stage trace snapshots (debug-only, env-gated) ──
# When STRUM_TRACE=1, after each major arbiter writes the chart hits to
# `<song_folder>/_trace/<NN>_<stage>.json`. Used by scripts/trace_hits.py
# to inspect how a hit at a given timestamp evolves across stages.
TRACE_ENABLED = os.environ.get("STRUM_TRACE", "0") == "1"
_trace_counter = {"n": 0}


def _trace_snapshot(chart, stage_name: str, song_folder: "Path"):
    if not TRACE_ENABLED or chart is None:
        return
    try:
        import json as _json
        trace_dir = song_folder / "_trace"
        trace_dir.mkdir(parents=True, exist_ok=True)
        idx = _trace_counter["n"]
        _trace_counter["n"] += 1
        path = trace_dir / f"{idx:02d}_{stage_name}.json"
        hits = [
            {
                "time_ms": float(h.time_ms),
                "lane": int(h.lane),
                "is_cymbal": bool(h.is_cymbal),
                "velocity": int(h.velocity),
            }
            for h in chart.hits
        ]
        path.write_text(_json.dumps({"stage": stage_name, "hits": hits}))
    except Exception as _e:
        logger.warning(f"  [trace] snapshot {stage_name} failed: {_e}")


def _reset_trace():
    _trace_counter["n"] = 0

# ── Paths ──
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output/hybrid")
MODEL_BUNDLE = get_active_bundle()


def _bundle_component_paths(
    component: str,
    checkpoint: str,
    config: str | None = None,
) -> tuple[str, str | None]:
    """Resolve a bundle override while retaining the historic local layout."""
    default_checkpoint = PROJECT_ROOT / checkpoint
    default_config = PROJECT_ROOT / config if config else None
    resolved_checkpoint = MODEL_BUNDLE.checkpoint(component, default_checkpoint)
    resolved_config = MODEL_BUNDLE.config(component, default_config)
    return str(resolved_checkpoint), str(resolved_config) if resolved_config else None

# ── V14 onset detector ──
V14_CHECKPOINT, _ = _bundle_component_paths(
    "drums.v14_onset", "checkpoints/drums_v14/best.pt"
)
CYMBAL_ONSET_CHECKPOINT = "checkpoints/drums_cymbal_onset/best_union_f1.pt"
USE_CYMBAL_ONSET = os.environ.get("STRUM_CYMBAL_ONSET", "0") == "1"  # Disabled: +onsets but ensemble misclassifies them
CYMBAL_ONSET_THRESHOLD = 0.35  # Lower than main onset (0.50) — cymbal head is targeted

# ── Multi-class onset head ──
MC_ONSET_CHECKPOINT = "checkpoints/drums_mc_onset/best.pt"
USE_MC_ONSET = os.environ.get("STRUM_MC_ONSET", "1") == "1"  # MC head for hybrid overlay

# ── Phase 3: purpose-built multi-class onset model (separate backbone) ──
PHASE3_CHECKPOINT = "checkpoints/drums_phase3/best.pt"
USE_PHASE3 = os.environ.get("STRUM_PHASE3", "1") == "1"  # Phase 3 replaces old MC head
# MC rescue (pre-ensemble injection) — DISABLED: ensemble misclassifies rescued onsets
MC_RESCUE_CLASSES = []  # was [3, 4, 5, 6, 7] — disabled in favor of hybrid overlay
MC_RESCUE_THRESHOLD = 0.70       # MC class prob must be > this to rescue
MC_ONSET_FLOOR = 0.15            # V14 onset prob must be > this (some signal)
MC_MIN_DISTANCE_MS = 30.0        # Don't rescue within this ms of an existing onset

# ── MC hybrid overlay (post-chart): use MC head directly for classes where it beats V14 ──
# Per-class optimal thresholds for MC hybrid overlay:
MC_HYBRID_ENABLED = os.environ.get("STRUM_MC_HYBRID", "1") == "1"
MC_HYBRID_CLASSES = {  # cls_idx → threshold (high: overlay only adds confident predictions)
    2: 0.70,   # HiHat
    3: 0.60,   # HighTom — refined by TomCNN
    4: 0.70,   # Ride
    5: 0.60,   # LowTom — refined by TomCNN
    6: 0.65,   # Crash
    7: 0.50,   # FloorTom — refined by TomCNN
}
MC_HYBRID_TOLERANCE_MS = 50.0  # Match tolerance for comparing chart hits vs MC peaks

# ── Phase 3 reclassification: fix misclassified existing chart hits ──
PHASE3_RECLASS_ENABLED = os.environ.get("STRUM_RECLASS", "1") == "1"
# Rules: (from_cls, to_cls) → (min_target_prob, min_margin=target-source)
# All thresholds env-tunable (STRUM_P3_<FROM>_<TO>_TH and _MARGIN) for sweeping.
def _p3_rule(name: str, default_th: float, default_margin: float) -> tuple[float, float]:
    return (
        float(os.environ.get(f"STRUM_P3_{name}_TH", str(default_th))),
        float(os.environ.get(f"STRUM_P3_{name}_MARGIN", str(default_margin))),
    )

PHASE3_RECLASS_RULES = {
    # Tightened defaults (2026-04-25 ear-test feedback): old thresholds caused
    # 50-80 HiHat→Ride flips per song which were almost always wrong, plus
    # cascading Crash→Ride drift. Raised target probs and margins so only
    # very confident reclassifications survive.
    (6, 4): _p3_rule("CRASH_RIDE", 0.55, 0.20),    # Crash→Ride (was 0.40,0.10)
    (2, 4): _p3_rule("HIHAT_RIDE", 0.45, 0.25),    # HiHat→Ride  (was 0.15,0.10)
    (2, 3): _p3_rule("HIHAT_HITOM", 0.45, 0.20),   # HiHat→HighTom (was 0.15,0.05)
    (1, 2): _p3_rule("SNARE_HIHAT", 0.40, 0.30),   # Snare→HiHat (was 0.15,0.20)
    (4, 5): _p3_rule("RIDE_LOTOM", 0.50, 0.10),    # Ride→LowTom (was 0.35,0.00)
    (1, 4): _p3_rule("SNARE_RIDE", 0.45, 0.25),    # Snare→Ride (was 0.15,0.15)
    (6, 2): _p3_rule("CRASH_HIHAT", 0.45, 0.20),   # Crash→HiHat (was 0.15,0.10)
}

# ── Phase 3 onset rescue: add missing onsets detected by Phase 3 ──
# Only adds peaks where NO chart hit exists on ANY lane within tolerance
PHASE3_RESCUE_ENABLED = os.environ.get("STRUM_RESCUE", "1") == "1"
PHASE3_RESCUE_THRESHOLD = float(os.environ.get("STRUM_RESCUE_TH", "0.50"))
PHASE3_RESCUE_TOLERANCE_MS = 50.0
PHASE3_RESCUE_MIN_DISTANCE_MS = 30.0

# ── Spectral reclassification: fix cross-lane errors using HNR/stereo/RMS ──
SPECTRAL_RECLASS_ENABLED = os.environ.get("STRUM_SPECTRAL_RECLASS", "0") == "1"  # v25: disabled by default — net-harmful after Phase 3 added
# HiHat→Snare: stereo_corr < threshold AND hnr > threshold
SPECTRAL_RECLASS_SC_TH = float(os.environ.get("STRUM_SR_SC", "0.96"))
SPECTRAL_RECLASS_HNR_SNARE = float(os.environ.get("STRUM_SR_HNR_S", "2.0"))
SPECTRAL_RECLASS_RMS_SNARE = float(os.environ.get("STRUM_SR_RMS_S", "0.40"))
# Crash→Ride: hnr > threshold AND rms > threshold
SPECTRAL_RECLASS_HNR_CRASH = float(os.environ.get("STRUM_SR_HNR_C", "2.0"))
SPECTRAL_RECLASS_RMS_CRASH = float(os.environ.get("STRUM_SR_RMS_C", "0.20"))
SPECTRAL_RECLASS_SR = 22050
SPECTRAL_RECLASS_WINDOW_MS = 60.0

V14_SR = 44100
V14_N_FFT = 2048
V14_HOP = 512
V14_N_MELS = 128

# ── Tom refinement CNN ──
TOM_REFINEMENT_CHECKPOINT = os.environ.get(
    "STRUM_TOM_REFINEMENT_CKPT",
    "checkpoints/tom_refinement_demucs/best.pt",
)
USE_TOM_REFINEMENT = os.environ.get("STRUM_TOM_REFINEMENT", "1") == "1"
TOM_REFINEMENT_CONTEXT_FRAMES = 43  # ~500ms context at 86 fps
TOM_REFINEMENT_N_CQT = 84  # LFCC bins used as CQT proxy
TOM_CLASSES_MC = {3: 1, 5: 2, 7: 3}  # MC class idx → tom refinement label (1=High, 2=Low, 3=Floor)
TOM_LABEL_TO_MC = {1: 3, 2: 5, 3: 7}  # tom refinement label → MC class idx

# ── Onset classifier ensemble ──
def _ensemble_entry(name: str, version: str, config: str, checkpoint: str) -> dict[str, str]:
    resolved_checkpoint, resolved_config = _bundle_component_paths(
        f"drums.ensemble.{version}", checkpoint, config
    )
    assert resolved_config is not None
    return {"name": name, "config": resolved_config, "checkpoint": resolved_checkpoint}


ENSEMBLE_MODELS = [
    _ensemble_entry("V2", "v2", "configs/onset_classifier.yaml", "checkpoints/onset_classifier/best_f1.pt"),
    _ensemble_entry("V4", "v4", "configs/onset_classifier_v4.yaml", "checkpoints/onset_classifier_v4/best_f1.pt"),
    _ensemble_entry("V6", "v6", "configs/onset_classifier_v6.yaml", "checkpoints/onset_classifier_v6/best_f1.pt"),
    _ensemble_entry("V12c", "v12c", "configs/onset_classifier_v12_clean.yaml", "checkpoints/onset_classifier_v12_clean/best_f1.pt"),
    _ensemble_entry("V15", "v15", "configs/onset_classifier_v15.yaml", "checkpoints/onset_classifier_v15/best_f1.pt"),
    _ensemble_entry("V16", "v16", "configs/onset_classifier_v16.yaml", "checkpoints/onset_classifier_v16/best_f1.pt"),
    _ensemble_entry("V17", "v17", "configs/onset_classifier_v17.yaml", "checkpoints/onset_classifier_v17/best_f1.pt"),
]

# Env-controlled V12c variant swap (A/B test community-trained vs clean-trained V12c).
# STRUM_V12C_VARIANT=community uses configs/onset_classifier_v12c_community.yaml
# + checkpoints/onset_classifier_v12c_community/best_f1.pt. Architecture is identical
# (lowfreq branch, HPSS, enhanced spectral) so window extraction is unchanged.
_V12C_VARIANT = os.environ.get("STRUM_V12C_VARIANT", "clean")
if _V12C_VARIANT == "community":
    for _i, _m in enumerate(ENSEMBLE_MODELS):
        if _m["name"] == "V12c":
            ENSEMBLE_MODELS[_i] = {
                "name": "V12c",
                "config": "configs/onset_classifier_v12c_community.yaml",
                "checkpoint": "checkpoints/onset_classifier_v12c_community/best_f1.pt",
            }
            break

# ── Onset window extraction params (must match training preprocessing) ──
OC_SR = 44100
FINE_N_FFT = 1024
FINE_HOP = 256
COARSE_N_FFT = 4096
COARSE_HOP = 512
N_MELS = 128
WINDOW_BEFORE_MS = 100.0
WINDOW_AFTER_MS = 400.0

WINDOW_BEFORE_SAMP = int(WINDOW_BEFORE_MS / 1000 * OC_SR)
WINDOW_AFTER_SAMP = int(WINDOW_AFTER_MS / 1000 * OC_SR)
WINDOW_SAMPLES = WINDOW_BEFORE_SAMP + WINDOW_AFTER_SAMP
FINE_FRAMES = WINDOW_SAMPLES // FINE_HOP + 1    # 87
COARSE_FRAMES = WINDOW_SAMPLES // COARSE_HOP + 1  # 44

# CQT params for V12c
CQT_FMIN = 30.0
CQT_BINS_PER_OCT = 24
CQT_N_BINS = 144
CQT_TRIM_BINS = 128  # Trimmed to match n_mels
CQT_HOP = 512
CQT_FRAMES = WINDOW_SAMPLES // CQT_HOP + 1  # 44

# ── Class mapping ──
CLASS_NAMES = ['Kick', 'Snare', 'HiHat', 'HighTom', 'Ride', 'LowTom', 'Crash', 'FloorTom']
CLASS_TO_LANE = {
    0: (0, False),  # Kick → lane 0
    1: (1, False),  # Snare → lane 1
    2: (2, True),   # HiHat → lane 2, cymbal
    3: (2, False),  # HighTom → lane 2, tom
    4: (3, True),   # Ride → lane 3, cymbal
    5: (3, False),  # LowTom → lane 3, tom
    6: (4, True),   # Crash → lane 4, cymbal
    7: (4, False),  # FloorTom → lane 4, tom
}

AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac"}


# ══════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════

def load_v14_onset_detector(device: torch.device) -> TwoStageDrumsCRNN:
    """Load V14 TwoStageDrumsCRNN, optionally with cymbal/MC onset heads."""
    use_cymbal = USE_CYMBAL_ONSET and Path(CYMBAL_ONSET_CHECKPOINT).exists()
    use_mc = USE_MC_ONSET and Path(MC_ONSET_CHECKPOINT).exists()

    # Choose checkpoint: MC > cymbal > V14
    if use_mc:
        logger.info(f"Loading V14 + MC onset head from {MC_ONSET_CHECKPOINT}")
        ckpt = torch.load(MC_ONSET_CHECKPOINT, map_location=device, weights_only=False)
    elif use_cymbal:
        logger.info(f"Loading V14 + cymbal onset head from {CYMBAL_ONSET_CHECKPOINT}")
        ckpt = torch.load(CYMBAL_ONSET_CHECKPOINT, map_location=device, weights_only=False)
    else:
        logger.info(f"Loading V14 onset detector from {V14_CHECKPOINT}")
        ckpt = torch.load(V14_CHECKPOINT, map_location=device, weights_only=False)

    cfg = ckpt.get("config", {}).get("model", {})

    model = TwoStageDrumsCRNN(
        n_mels=cfg.get("n_mels", 128),
        conv_channels=cfg.get("conv_channels", [64, 128, 256, 512]),
        freq_subbands=cfg.get("freq_subbands", [32, 64, 96, 128]),
        subband_proj_dim=cfg.get("subband_proj_dim", 256),
        lstm_hidden=cfg.get("lstm_hidden", 640),
        lstm_layers=cfg.get("lstm_layers", 3),
        attention_heads=cfg.get("attention_heads", 10),
        attention_type=cfg.get("attention_type", "flash"),
        attention_window=cfg.get("attention_window", 512),
        dropout=0.0,
        onset_detector_hidden=cfg.get("onset_detector_hidden", 320),
        classifier_hidden=cfg.get("classifier_hidden", 640),
        num_classes=cfg.get("num_classes", 8),
        predict_velocity=cfg.get("predict_velocity", True),
        cymbal_onset_head=use_cymbal and not use_mc,
    )

    if use_mc:
        model.add_multiclass_onset_head()

    model = model.to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    epoch = ckpt.get("epoch", "?")
    best_f1 = ckpt.get("best_f1", "?")
    if use_mc:
        logger.info(f"  V14 + MC onset head loaded (epoch {epoch})")
    elif use_cymbal:
        logger.info(f"  V14 + cymbal head loaded (epoch {epoch})")
    else:
        logger.info(f"  V14 loaded (epoch {epoch}, best_f1={best_f1})")
    return model


def load_phase3_model(device: torch.device) -> TwoStageDrumsCRNN | None:
    """Load the Phase 3 purpose-built multi-class onset model (separate backbone).

    Phase 3 has its own backbone trained from scratch for 8-class onset detection.
    It runs independently from V14 and replaces the old MC head's frame probs.
    """
    if not USE_PHASE3 or not Path(PHASE3_CHECKPOINT).exists():
        return None

    logger.info(f"Loading Phase 3 MC model from {PHASE3_CHECKPOINT}")
    ckpt = torch.load(PHASE3_CHECKPOINT, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}).get("model", {})

    model = TwoStageDrumsCRNN(
        n_mels=cfg.get("n_mels", 128),
        conv_channels=cfg.get("conv_channels", [64, 128, 256, 512]),
        freq_subbands=cfg.get("freq_subbands", [32, 64, 96, 128]),
        subband_proj_dim=cfg.get("subband_proj_dim", 256),
        lstm_hidden=cfg.get("lstm_hidden", 640),
        lstm_layers=cfg.get("lstm_layers", 3),
        attention_heads=cfg.get("attention_heads", 10),
        attention_type=cfg.get("attention_type", "flash"),
        attention_window=cfg.get("attention_window", 512),
        dropout=0.0,
        onset_detector_hidden=cfg.get("onset_detector_hidden", 320),
        classifier_hidden=cfg.get("classifier_hidden", 640),
        num_classes=cfg.get("num_classes", 8),
        predict_velocity=cfg.get("predict_velocity", False),
    )
    model.add_multiclass_onset_head()
    model = model.to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    metrics = ckpt.get("metrics", {})
    best_f1 = metrics.get("Overall_F1", metrics.get("BestSweep_F1", "?"))
    logger.info(f"  Phase 3 loaded (epoch {epoch}, F1={best_f1}%)")
    return model


def _compute_log_mel_with_optional_bg_subtract(
    audio: torch.Tensor,
    mel_transform: "torchaudio.transforms.MelSpectrogram",
    model_tag: str = "all",
) -> torch.Tensor:
    """Compute log-mel for V14/Phase3 input, optionally background-subtracted.

    Delegates to src.models.bg_mel.bg_subtract_log_mel so training
    (DrumsV14FullSongDataset) and inference use byte-identical math.

    Env knobs:
      STRUM_MEL_BG_SUBTRACT: "0" / "1" / "all" / "v14" / "p3"
      STRUM_MEL_BG_ALPHA: subtraction strength 0..1 (default 0.3)
      STRUM_MEL_BG_PCTL: percentile for background estimate (default 20)
      STRUM_MEL_BG_WINDOW_SEC: rolling window seconds (default 1.0)
    """
    from src.models.bg_mel import bg_subtract_log_mel
    power_mel = mel_transform(audio)
    return bg_subtract_log_mel(
        power_mel,
        sample_rate=V14_SR,
        hop_length=V14_HOP,
        model_tag=model_tag,
    )


def run_phase3_inference(
    model: TwoStageDrumsCRNN,
    audio_path: Path,
    device: torch.device,
    segment_duration: float = 10.0,
    overlap: float = 0.5,
) -> np.ndarray:
    """Run Phase 3 model on audio and return per-frame MC onset probs.

    Uses the same overlap-average strategy as V14 for consistency.

    Returns:
        mc_probs: (T, 8) per-frame per-class onset probabilities
    """
    y, sr = librosa.load(str(audio_path), sr=V14_SR, mono=True)
    audio = torch.from_numpy(y).unsqueeze(0)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=V14_SR, n_fft=V14_N_FFT, hop_length=V14_HOP, n_mels=V14_N_MELS,
    )
    mel_spec = _compute_log_mel_with_optional_bg_subtract(audio, mel_transform, model_tag="phase3")

    total_frames = mel_spec.shape[-1]
    segment_frames = int(segment_duration * V14_SR / V14_HOP)
    hop_frames = int(segment_frames * (1 - overlap))

    mc_probs = np.zeros((total_frames, 8), dtype=np.float32)
    count = np.zeros(total_frames)

    num_segments = max(1, (total_frames - segment_frames) // hop_frames + 1)

    with torch.no_grad():
        for i in tqdm(range(num_segments), desc="  Phase3 MC inference", leave=False):
            start = i * hop_frames
            end = min(start + segment_frames, total_frames)
            segment = mel_spec[:, :, start:end]
            if segment.shape[-1] < segment_frames:
                segment = torch.nn.functional.pad(segment, (0, segment_frames - segment.shape[-1]))
            segment = segment.unsqueeze(0).to(device)
            outputs = model(segment)
            probs = outputs["mc_onset_probs"].squeeze(0).cpu().numpy()  # (T, 8)
            actual_len = min(end - start, probs.shape[0])
            mc_probs[start:start + actual_len] += probs[:actual_len]
            count[start:start + actual_len] += 1

    count = np.maximum(count, 1)
    mc_probs /= count[:, None]

    return mc_probs


def build_onset_classifier(config) -> OnsetClassifier:
    """Build onset classifier from config."""
    m = config.model
    return OnsetClassifier(
        num_classes=m.num_classes,
        branch_channels=list(m.branch_channels),
        context_size=m.context_size,
        context_hidden=m.context_hidden,
        classifier_hidden=m.classifier_hidden,
        spectral_dim=m.get("spectral_dim", 32),
        dropout=0.0,  # no dropout at inference
        use_freq_attn=m.get("use_freq_attn", False),
        use_hpss=m.get("use_hpss", False),
        enhanced_spectral=m.get("enhanced_spectral", False),
        use_contrastive=False,
        use_aux_head=False,
        use_dual_head=m.get("use_dual_head", False),
        tom_head_hidden=m.get("tom_head_hidden", 256),
        use_lowfreq_branch=m.get("use_lowfreq_branch", False),
        use_lowfreq_spectral=m.get("use_lowfreq_spectral", False),
    )


def load_ensemble(device: torch.device) -> list[dict]:
    """Load all onset classifier ensemble models."""
    models = []
    for eidx, entry in enumerate(ENSEMBLE_MODELS):
        name = entry["name"]
        cfg_path = entry["config"]
        ckpt_path = entry["checkpoint"]

        if not Path(ckpt_path).exists() or not Path(cfg_path).exists():
            logger.warning(f"  {name}: missing checkpoint or config, skipping")
            continue

        config = OmegaConf.load(cfg_path)
        model = build_onset_classifier(config).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        needs_cqt = config.model.get("use_lowfreq_branch", False) or config.model.get("use_lowfreq_spectral", False)
        epoch = ckpt.get("epoch", -1) + 1
        logger.info(f"  {name}: epoch {epoch}, needs_cqt={needs_cqt}")

        models.append({
            "name": name,
            "model": model,
            "config": config,
            "needs_cqt": needs_cqt,
            "ensemble_idx": eidx,  # original index in ENSEMBLE_MODELS for weight lookup
        })

    logger.info(f"  Loaded {len(models)} ensemble models")
    return models


# Default path for the binary tom/cymbal CNN classifier checkpoint
TOMCYM_CHECKPOINT = Path(__file__).parent.parent / "checkpoints" / "tomcym_v2" / "best.pt"


def load_tomcym_classifier(
    device: torch.device,
    checkpoint_path: Path | None = None,
) -> dict | None:
    """Load the binary tom/cymbal CNN classifier if available.

    Returns dict with 'model', 'norm_mean', 'norm_std' or None if not trained yet.
    """
    ckpt_path = checkpoint_path or TOMCYM_CHECKPOINT
    if not ckpt_path.exists() or TomCymCNN is None:
        logger.info("  Tom/cymbal classifier not found — skipping binary reclassification")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = TomCymCNN(dropout=0.0)  # no dropout at inference
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    metrics = ckpt.get("metrics", {})
    logger.info(f"  Tom/cymbal CNN classifier loaded: macro_f1={metrics.get('macro_f1', '?'):.3f}")

    return {
        "model": model,
        "norm_mean": float(ckpt["norm_mean"]),
        "norm_std": float(ckpt["norm_std"]),
    }


def load_tom_refinement(device: torch.device) -> dict | None:
    """Load the tom refinement CNN if checkpoint exists."""
    ckpt_path = Path(TOM_REFINEMENT_CHECKPOINT)
    if not USE_TOM_REFINEMENT or not ckpt_path.exists():
        if USE_TOM_REFINEMENT:
            logger.info("  Tom refinement checkpoint not found — skipping")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = TomRefinementCNN(
        n_mel=V14_N_MELS,
        n_cqt=TOM_REFINEMENT_N_CQT,
        context_frames=TOM_REFINEMENT_CONTEXT_FRAMES,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    best_f1 = ckpt.get("best_tom_f1", ckpt.get("best_f1", "?"))
    logger.info(f"  Tom refinement CNN loaded: best_tom_f1={best_f1}")

    # Precompute mel/CQT transforms (same params as precompute_tom_windows.py)
    # Keep on CPU — LFCC doesn't support GPU
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=V14_SR, n_fft=V14_N_FFT, hop_length=V14_HOP, n_mels=V14_N_MELS,
    )
    cqt_transform = torchaudio.transforms.LFCC(
        sample_rate=V14_SR, n_lfcc=TOM_REFINEMENT_N_CQT,
        speckwargs={'n_fft': V14_N_FFT, 'hop_length': V14_HOP},
    )

    return {
        "model": model,
        "mel_transform": mel_transform,
        "cqt_transform": cqt_transform,
    }


def apply_tomcym_reclassification(
    chart: DrumChart,
    audio_path: Path,
    tomcym: dict,
    device: torch.device,
    tom_threshold: float = 0.30,
    cym_threshold: float = 0.72,
) -> DrumChart:
    """Run CNN tom/cymbal classifier on shared-lane hits and reclassify.

    Only reclassifies hits where the CNN disagrees with the ensemble AND
    is confident. Uses mel spectrogram patches for maximum accuracy.

    Threshold semantics: CNN outputs prob(cymbal).
    - Override ensemble's cymbal → tom when prob < tom_threshold
    - Override ensemble's tom → cymbal when prob > cym_threshold

    Conservative thresholds to avoid creating new errors:
      tom_threshold=0.30: tom_precision=63.8% on test set
      cym_threshold=0.72: only overrides clearly wrong toms

    Args:
        chart: The chart to reclassify
        audio_path: Path to drums stem audio
        tomcym: Dict with 'model', 'norm_mean', 'norm_std'
        device: Torch device
        tom_threshold: Override to tom when CNN prob < this
        cym_threshold: Override to cymbal when CNN prob > this

    Returns:
        Modified chart with reclassified hits
    """
    from scripts.preprocess_tomcym_mel import extract_mel_patch, SAMPLE_RATE, N_MELS

    SHARED_LANES = {2, 3, 4}
    shared_indices = [i for i, h in enumerate(chart.hits) if h.lane in SHARED_LANES]

    if not shared_indices:
        return chart

    # Load audio once
    y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)

    # Extract mel patches for all shared-lane hits
    patches_list = []
    valid_indices = []
    patch_shape = None

    for idx in shared_indices:
        hit = chart.hits[idx]
        patch = extract_mel_patch(y, sr, hit.time_ms)
        if patch is not None:
            if patch_shape is None:
                patch_shape = patch.shape
            if patch.shape == patch_shape:
                patches_list.append(patch)
                valid_indices.append(idx)

    if not patches_list:
        return chart

    # Stack, normalize, add channel dim
    patches = np.stack(patches_list)  # (N, n_mels, n_frames)
    patches = (patches - tomcym["norm_mean"]) / tomcym["norm_std"]
    patches = patches[:, np.newaxis, :, :]  # (N, 1, n_mels, n_frames)
    patches_t = torch.from_numpy(patches).float().to(device)

    model = tomcym["model"]
    with torch.no_grad():
        logits = model(patches_t).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()  # prob of cymbal (class 1)

    # Reclassify: only override when CNN is confident
    n_cym_to_tom = 0
    n_tom_to_cym = 0

    for feat_idx, chart_idx in enumerate(valid_indices):
        hit = chart.hits[chart_idx]
        cym_prob = probs[feat_idx]
        current_is_cymbal = hit.is_cymbal

        if current_is_cymbal and cym_prob < tom_threshold:
            # Ensemble says cymbal, CNN says tom with high confidence
            chart.hits[chart_idx] = DrumHit(
                time_ms=hit.time_ms,
                tick=hit.tick,
                lane=hit.lane,
                is_cymbal=False,
                velocity=hit.velocity,
            )
            n_cym_to_tom += 1
        elif not current_is_cymbal and cym_prob > cym_threshold:
            # Ensemble says tom, CNN says cymbal with high confidence
            chart.hits[chart_idx] = DrumHit(
                time_ms=hit.time_ms,
                tick=hit.tick,
                lane=hit.lane,
                is_cymbal=True,
                velocity=hit.velocity,
            )
            n_tom_to_cym += 1

    if n_tom_to_cym > 0 or n_cym_to_tom > 0:
        logger.info(f"  Tom/cymbal CNN reclassification: {n_cym_to_tom} cym→tom, {n_tom_to_cym} tom→cym")

    return chart


# ══════════════════════════════════════════════════════════════
# Audio processing
# ══════════════════════════════════════════════════════════════

def separate_drums(audio_path: Path, output_dir: Path) -> Path:
    """Separate drums stem using Demucs Python API."""
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    demucs_out = output_dir / "demucs_temp"
    demucs_out.mkdir(parents=True, exist_ok=True)

    drums_path = demucs_out / "drums.wav"
    if drums_path.exists() and drums_path.stat().st_size > 0:
        logger.info(f"  Demucs cache hit: {drums_path}")
        return drums_path

    logger.info("  Separating drums with Demucs (htdemucs_ft)...")

    y, sr = librosa.load(str(audio_path), sr=44100, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])  # mono → stereo

    model_demucs = get_model(os.environ.get("STRUM_DEMUCS_MODEL", "htdemucs_ft"))
    model_demucs.eval()

    wav_tensor = torch.from_numpy(y).float().unsqueeze(0)
    with torch.no_grad():
        sources = apply_model(model_demucs, wav_tensor, progress=True)

    source_names = model_demucs.sources
    drums_idx = source_names.index("drums")
    drums_audio = sources[0, drums_idx].cpu().numpy()

    sf.write(str(drums_path), drums_audio.T, 44100)
    logger.info(f"  Drums stem: {drums_path}")
    return drums_path


# ── 6-stem drum micro-separation (Jarredou MDX23C) ────────────
# Used by the per-onset RMS lane arbiter to disambiguate Tom-vs-Cymbal
# confusions caused by hi-hat / cymbal bleed in the Demucs drums stem.
# Validated 2026-04-27 across 3 songs: Tom→Cymbal fix-rate ≥98% with
# +40dB mean stem-RMS margin (see scripts/analyze_drumsep_lane_arbiter.py).
DRUMSEP_STEM_NAMES = ["kick", "snare", "toms", "hh", "ride", "crash"]
DRUMSEP_DEFAULT_MSST = "/home/opria123/drumsep/msst"
DRUMSEP_DEFAULT_CKPT = "/home/opria123/drumsep/models/jarredou_drumsep_6stem.ckpt"
DRUMSEP_DEFAULT_CFG = "/home/opria123/drumsep/models/jarredou_drumsep_6stem.yaml"


def separate_drums_6stem(drums_path: Path, output_dir: Path) -> Path | None:
    """Run Jarredou MDX23C 6-stem drum separation on the Demucs drums stem.

    Caches output stems in `output_dir / "demucs_temp" / "stems_6"`.
    Returns the path to the stems dir, or None if separation is unavailable
    or fails. The caller should treat None as "skip arbiter, use base probs".
    """
    msst_dir = Path(os.environ.get("STRUM_DRUMSEP_MSST", DRUMSEP_DEFAULT_MSST))
    ckpt = Path(os.environ.get("STRUM_DRUMSEP_CKPT", DRUMSEP_DEFAULT_CKPT))
    cfg = Path(os.environ.get("STRUM_DRUMSEP_CFG", DRUMSEP_DEFAULT_CFG))

    stems_dir = output_dir / "demucs_temp" / "stems_6"
    # Cache hit: already separated for this song
    expected = [stems_dir / f"{n}.wav" for n in DRUMSEP_STEM_NAMES]
    if all(p.exists() for p in expected):
        logger.info(f"  6-stem drumsep cache hit: {stems_dir}")
        return stems_dir

    if not (msst_dir.exists() and ckpt.exists() and cfg.exists()):
        logger.warning(
            f"  6-stem drumsep skipped: missing assets "
            f"(msst={msst_dir.exists()}, ckpt={ckpt.exists()}, cfg={cfg.exists()})"
        )
        return None

    # MSST inference.py wants a folder of inputs and writes outputs into
    # `<store_dir>/<input_basename>/<stem>.wav`.  Stage drums.wav in a
    # temp folder named after the stem dir's parent so the layout is
    # predictable, then move stems into place.
    stems_dir.mkdir(parents=True, exist_ok=True)
    stage_in = output_dir / "demucs_temp" / "stems_6_stage_in"
    stage_out = output_dir / "demucs_temp" / "stems_6_stage_out"
    if stage_in.exists():
        shutil.rmtree(stage_in)
    if stage_out.exists():
        shutil.rmtree(stage_out)
    stage_in.mkdir(parents=True, exist_ok=True)
    staged_drums = stage_in / "drums.wav"
    try:
        os.symlink(drums_path.resolve(), staged_drums)
    except OSError:
        shutil.copy2(drums_path, staged_drums)

    logger.info("  Running 6-stem drumsep (Jarredou MDX23C)...")
    t0 = time.time()
    cmd = [
        sys.executable, "inference.py",
        "--model_type", "mdx23c",
        "--config_path", str(cfg),
        "--start_check_point", str(ckpt),
        "--input_folder", str(stage_in),
        "--store_dir", str(stage_out),
        "--device_ids", "0",
    ]
    try:
        subprocess.run(
            cmd, cwd=str(msst_dir),
            check=True, capture_output=True, text=True, timeout=900,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning(f"  6-stem drumsep failed: {e}")
        if hasattr(e, "stderr") and e.stderr:
            logger.warning(f"  stderr tail: {e.stderr[-500:]}")
        return None

    # MSST writes outputs to <stage_out>/drums/<stem>.wav. Move into stems_dir.
    src_dir = stage_out / "drums"
    if not src_dir.exists():
        # Some MSST versions flatten when there's a single input
        src_dir = stage_out
    n_moved = 0
    for stem in DRUMSEP_STEM_NAMES:
        src = src_dir / f"{stem}.wav"
        if not src.exists():
            logger.warning(f"  6-stem drumsep missing output: {src}")
            return None
        shutil.move(str(src), str(stems_dir / f"{stem}.wav"))
        n_moved += 1
    shutil.rmtree(stage_in, ignore_errors=True)
    shutil.rmtree(stage_out, ignore_errors=True)

    logger.info(f"  6-stem drumsep done ({time.time()-t0:.1f}s, {n_moved} stems)")
    return stems_dir


def _stem_rms_db_at(stem_audio: np.ndarray, sr: int, t_sec: float, win_ms: float = 50.0) -> float:
    """RMS in dB over a centered window in a (mono or stereo) stem array."""
    if stem_audio.ndim > 1:
        x = stem_audio.mean(axis=0) if stem_audio.shape[0] < stem_audio.shape[1] else stem_audio.mean(axis=1)
    else:
        x = stem_audio
    half = int(win_ms * 1e-3 * sr / 2)
    c = int(t_sec * sr)
    a = max(0, c - half)
    b = min(len(x), c + half)
    if b <= a:
        return -120.0
    seg = x[a:b]
    rms = float(np.sqrt(np.mean(seg ** 2) + 1e-12))
    return 20.0 * np.log10(rms + 1e-12)


def _stem_rms_db_window(stem_audio: np.ndarray, sr: int, t_sec: float,
                        offset_ms: float, win_ms: float) -> float:
    """RMS in dB over a window placed at `t_sec + offset_ms`, length `win_ms`.

    Negative offset = window ends before t (pre-onset). Positive offset =
    window starts after t (post-onset). offset_ms refers to the window's
    near edge relative to t.
    """
    if stem_audio.ndim > 1:
        x = stem_audio.mean(axis=0) if stem_audio.shape[0] < stem_audio.shape[1] else stem_audio.mean(axis=1)
    else:
        x = stem_audio
    n_win = int(win_ms * 1e-3 * sr)
    c = int(t_sec * sr)
    if offset_ms < 0:
        # window of length n_win ending at c + offset_samples
        end = c + int(offset_ms * 1e-3 * sr)
        start = end - n_win
    else:
        start = c + int(offset_ms * 1e-3 * sr)
        end = start + n_win
    a = max(0, start)
    b = min(len(x), end)
    if b <= a:
        return -120.0
    seg = x[a:b]
    rms = float(np.sqrt(np.mean(seg ** 2) + 1e-12))
    return 20.0 * np.log10(rms + 1e-12)


def _toms_has_attack(toms_audio: np.ndarray, sr: int, t_sec: float,
                     attack_db: float = 6.0,
                     pre_offset_ms: float = -60.0,
                     pre_win_ms: float = 40.0,
                     post_offset_ms: float = 0.0,
                     post_win_ms: float = 40.0) -> tuple[bool, float]:
    """Check whether the toms stem has a discrete attack (transient) at t.

    Returns (has_attack, attack_db_value). True when post-onset RMS
    exceeds pre-onset RMS by at least `attack_db`. Cymbal-wash bleed in
    the toms stem manifests as sustained energy (post≈pre, fails check);
    a real tom strike has post >> pre.
    """
    pre_db = _stem_rms_db_window(toms_audio, sr, t_sec, pre_offset_ms, pre_win_ms)
    post_db = _stem_rms_db_window(toms_audio, sr, t_sec, post_offset_ms, post_win_ms)
    delta = post_db - pre_db
    return (delta >= attack_db), delta


def apply_drumsep_lane_arbiter(
    probs: np.ndarray,
    onset_times_ms: list[float],
    stems_dir: Path,
    win_ms: float = 50.0,
    margin_db: float = 20.0,
    toms_floor_db: float = -40.0,
) -> np.ndarray:
    """Per-onset stem-RMS arbiter for Tom-vs-Cymbal lane confusions.

    Implements the per-onset loudness-curve approach from arXiv 2509.24853
    (Sept 2025) restricted to the Tom-vs-Cymbal axis where stem separation
    gives an unambiguous physical signal (validated +40 dB mean margin
    across 3 songs).

    For every onset, compares `toms.wav` RMS vs each cymbal-stem RMS in a
    centred 50ms window. When toms wins by `margin_db` AND toms is above
    `toms_floor_db`, transfers cymbal probability mass to the paired tom
    column. Conservative: never inflates a tom that wasn't already a real
    onset, only redirects existing cymbal mass.

    Args:
        probs: (N, 8) class probabilities (K, S, HH, HiTom, Ride, LoTom, Crash, FloorTom).
        onset_times_ms: onset timestamps in ms.
        stems_dir: directory containing kick/snare/toms/hh/ride/crash .wav stems.
        win_ms: RMS window width in ms.
        margin_db: required toms_db - cym_db to flip a cymbal→tom.
        toms_floor_db: required toms_db absolute energy gate (silence guard).

    Returns:
        Modified (N, 8) probs.
    """
    pairs = [(2, 3, "hh"), (4, 5, "ride"), (6, 7, "crash")]  # (cym_idx, tom_idx, stem_name)

    needed = ["toms", "hh", "ride", "crash"]
    stems = {}
    sr = None
    for name in needed:
        p = stems_dir / f"{name}.wav"
        if not p.exists():
            logger.warning(f"  Drumsep arbiter: missing stem {p}, skipping")
            return probs
        x, x_sr = sf.read(str(p), always_2d=False)
        if sr is None:
            sr = x_sr
        # Mono-mix
        if x.ndim > 1:
            x = x.mean(axis=1)
        stems[name] = x.astype(np.float32, copy=False)

    out = probs.copy()
    n_flipped = {name: 0 for _, _, name in pairs}
    margins = {name: [] for _, _, name in pairs}
    n_skipped_cym_attack = 0
    transfer_pct = float(os.environ.get("STRUM_DRUMSEP_TRANSFER", "0.85"))
    cym_attack_block = os.environ.get("STRUM_DRUMSEP_CYM_ATTACK_BLOCK", "0") == "1"
    cym_attack_db = float(os.environ.get("STRUM_DRUMSEP_CYM_ATTACK_DB", "6.0"))

    for i, t_ms in enumerate(onset_times_ms):
        t = t_ms / 1000.0
        toms_db = _stem_rms_db_at(stems["toms"], sr, t, win_ms)
        if toms_db < toms_floor_db:
            continue
        for cym_idx, tom_idx, stem_name in pairs:
            cym_p = out[i, cym_idx]
            if cym_p < 0.10:
                continue
            cym_db = _stem_rms_db_at(stems[stem_name], sr, t, win_ms)
            margin = toms_db - cym_db
            if margin >= margin_db:
                # Veto: cymbal stem itself shows a clear attack at this onset
                # → it IS a real cymbal hit, toms stem energy is wash bleed.
                if cym_attack_block:
                    has_cym_attack, _ = _toms_has_attack(
                        stems[stem_name], sr, t, attack_db=cym_attack_db,
                    )
                    if has_cym_attack:
                        n_skipped_cym_attack += 1
                        continue
                # Transfer mass from cymbal → tom
                boost = cym_p * transfer_pct
                out[i, tom_idx] = min(1.0, out[i, tom_idx] + boost)
                out[i, cym_idx] = cym_p * (1.0 - transfer_pct)
                n_flipped[stem_name] += 1
                margins[stem_name].append(margin)

    total = sum(n_flipped.values())
    if total > 0 or n_skipped_cym_attack > 0:
        bits = []
        for _, _, name in pairs:
            if n_flipped[name]:
                mm = float(np.mean(margins[name]))
                bits.append(f"{name}→tom={n_flipped[name]} (μ={mm:+.1f}dB)")
        veto_note = f", {n_skipped_cym_attack} vetoed (cym stem has attack)" if n_skipped_cym_attack else ""
        if bits:
            logger.info(f"  Drumsep arbiter: {total} flips ({', '.join(bits)}){veto_note}")
        else:
            logger.info(f"  Drumsep arbiter: 0 flips{veto_note}")
    else:
        logger.info("  Drumsep arbiter: 0 flips")
    return out


def apply_drumsep_hits_arbiter(
    chart: DrumChart,
    stems_dir: Path,
    win_ms: float = 50.0,
    margin_db: float = 20.0,
    toms_floor_db: float = -40.0,
) -> DrumChart:
    """Final pass on emitted hits: flip cymbal→tom on the same lane when
    the 6-stem toms.wav dominates the paired cymbal stem at the hit time.

    Mirrors `apply_drumsep_lane_arbiter` but operates on `chart.hits` after
    all postprocess and Phase 3 stages, so the physical-separation signal
    has the final word.
    """
    # Lane → cymbal stem name (lane index per CLASS_TO_LANE)
    lane_to_cym = {2: "hh", 3: "ride", 4: "crash"}
    needed = ["toms", "hh", "ride", "crash"]
    stems = {}
    sr = None
    for name in needed:
        p = stems_dir / f"{name}.wav"
        if not p.exists():
            logger.warning(f"  Drumsep hits arbiter: missing stem {p}, skipping")
            return chart
        x, x_sr = sf.read(str(p), always_2d=False)
        if sr is None:
            sr = x_sr
        if x.ndim > 1:
            x = x.mean(axis=1)
        stems[name] = x.astype(np.float32, copy=False)

    n_flipped = {n: 0 for n in lane_to_cym.values()}
    margins = {n: [] for n in lane_to_cym.values()}
    n_skipped_cym_attack = 0
    n_skipped_no_toms_attack = 0
    n_skipped_fill_end = 0
    cym_attack_block = os.environ.get("STRUM_DRUMSEP_CYM_ATTACK_BLOCK", "0") == "1"
    cym_attack_db = float(os.environ.get("STRUM_DRUMSEP_CYM_ATTACK_DB", "6.0"))
    # NEW: require the toms stem to show a discrete attack (post-RMS exceeds
    # pre-RMS by attack_db). Prevents tom-roll decay / kick+snare bleed from
    # masquerading as a tom hit when the model classified the onset as a cymbal.
    # OFF by default: 12-song eval (Apr 2026) showed it cuts 38% of Cym→Tom
    # errors but doubles Tom→Cym errors and degrades Alkaline class-acc by 3pp.
    # The veto is too coarse; drumsep arbiter's correct flips overlap with the
    # perceptually-bad ones. Available behind STRUM_DRUMSEP_TOMS_ATTACK=1.
    toms_attack_required = os.environ.get("STRUM_DRUMSEP_TOMS_ATTACK", "0") == "1"
    toms_attack_db = float(os.environ.get("STRUM_DRUMSEP_TOMS_ATTACK_DB", "4.0"))
    # NEW: music-context veto for "crash at end of tom fill". When a Crash
    # (lane 4) hit is preceded by ≥N tom hits within `fill_window_ms`, the
    # toms.wav stem RMS is dominated by lingering tom decay rather than a
    # fresh tom attack. Block the flip — trust the model's cymbal call.
    # ON by default (v25.1, Apr 2026). Targets bug #3: "tom roll ending in
    # crash → marked as floor tom". Settings tuned via 12-song eval:
    # 2 toms / 800ms eliminated 19 Cym→Tom errors, +0.7pp Crash recall.
    fill_end_veto = os.environ.get("STRUM_DRUMSEP_FILL_END_VETO", "1") == "1"
    fill_window_ms = float(os.environ.get("STRUM_DRUMSEP_FILL_WIN_MS", "800"))
    fill_min_toms = int(os.environ.get("STRUM_DRUMSEP_FILL_MIN_TOMS", "2"))
    # Pre-compute sorted list of tom-hit times for O(log N) neighbourhood query
    tom_lanes = {2, 3, 4}
    tom_times = sorted(
        h.time_ms for h in chart.hits
        if (h.lane in tom_lanes and not h.is_cymbal)
    )
    import bisect
    new_hits = []
    for hit in chart.hits:
        if not hit.is_cymbal or hit.lane not in lane_to_cym:
            new_hits.append(hit)
            continue
        cym_name = lane_to_cym[hit.lane]
        t = hit.time_ms / 1000.0
        toms_db = _stem_rms_db_at(stems["toms"], sr, t, win_ms)
        if toms_db < toms_floor_db:
            new_hits.append(hit)
            continue
        cym_db = _stem_rms_db_at(stems[cym_name], sr, t, win_ms)
        margin = toms_db - cym_db
        if margin >= margin_db:
            # Veto 0 (default): "crash at end of tom fill" — if N+ toms exist
            # within fill_window_ms BEFORE this hit, lingering tom decay
            # explains the toms.wav dominance and the model's cymbal call
            # is trustworthy. Crash (lane 4) only — hi-hat and ride don't
            # exhibit this pattern.
            if fill_end_veto and hit.lane == 4:
                lo = bisect.bisect_left(tom_times, hit.time_ms - fill_window_ms)
                hi = bisect.bisect_left(tom_times, hit.time_ms)
                if (hi - lo) >= fill_min_toms:
                    n_skipped_fill_end += 1
                    new_hits.append(hit)
                    continue
            # Veto 1 (opt-in): toms stem must have a fresh attack (transient).
            # Sustained tom-roll decay or kick/snare bleed will fail this check.
            if toms_attack_required:
                has_toms_attack, _ = _toms_has_attack(
                    stems["toms"], sr, t, attack_db=toms_attack_db,
                )
                if not has_toms_attack:
                    n_skipped_no_toms_attack += 1
                    new_hits.append(hit)
                    continue
            # Veto 2 (opt-in): cymbal stem itself has an attack at this onset.
            if cym_attack_block:
                has_cym_attack, _ = _toms_has_attack(
                    stems[cym_name], sr, t, attack_db=cym_attack_db,
                )
                if has_cym_attack:
                    n_skipped_cym_attack += 1
                    new_hits.append(hit)
                    continue
            new_hits.append(DrumHit(
                time_ms=hit.time_ms, tick=hit.tick, lane=hit.lane,
                is_cymbal=False, velocity=hit.velocity,
            ))
            n_flipped[cym_name] += 1
            margins[cym_name].append(margin)
        else:
            new_hits.append(hit)

    total = sum(n_flipped.values())
    if total > 0 or n_skipped_cym_attack > 0 or n_skipped_no_toms_attack > 0 or n_skipped_fill_end > 0:
        bits = []
        for name in ("hh", "ride", "crash"):
            if n_flipped[name]:
                mm = float(np.mean(margins[name]))
                bits.append(f"{name}→tom={n_flipped[name]} (μ={mm:+.1f}dB)")
        veto_bits = []
        if n_skipped_fill_end:
            veto_bits.append(f"{n_skipped_fill_end} vetoed (crash at end of fill)")
        if n_skipped_no_toms_attack:
            veto_bits.append(f"{n_skipped_no_toms_attack} vetoed (no toms attack)")
        if n_skipped_cym_attack:
            veto_bits.append(f"{n_skipped_cym_attack} vetoed (cym stem has attack)")
        veto_note = (", " + ", ".join(veto_bits)) if veto_bits else ""
        if bits:
            logger.info(f"  Drumsep hits arbiter: {total} flips ({', '.join(bits)}){veto_note}")
        else:
            logger.info(f"  Drumsep hits arbiter: 0 flips{veto_note}")
    else:
        logger.info("  Drumsep hits arbiter: 0 flips")

    return DrumChart(
        hits=new_hits,
        tempo_events=chart.tempo_events,
        time_signatures=chart.time_signatures,
        ticks_per_beat=chart.ticks_per_beat,
    )


def apply_crash_ride_costack_veto(
    chart: DrumChart,
    stack_tol_ms: float = 30.0,
) -> DrumChart:
    """Drop the Ride from a Crash+Ride co-stack (within ±30 ms).

    Bug (May 2026 ear-test): pure crash hits frequently get a phantom
    Ride charted alongside them, producing an audibly wrong "double
    cymbal" when a clean crash should sit alone. Investigated across all
    12 sample songs vs ground-truth charts:

      Total Crash+Ride co-stacks (within ±30 ms): 230
        Both legit in GT:   1   ( 0.4%)  → would be wrongly modified
        Crash-only in GT: 135  (58.7%)
        Ride-only in GT:   9   ( 3.9%)
        Phantom in GT:    85  (37.0%)

    => 95.7 % of co-stacks should drop the Ride. Real Crash+Ride
    co-stacks (drummer plays crash AND ride at the same instant) are
    extraordinarily rare — only 1 case across 230 in our 12-song sample.

    Implementation: for each Crash hit (lane 4 cymbal), drop any Ride
    hits (lane 3 cymbal) within ±`stack_tol_ms`. Crash is preserved.

    This veto MUST run after `phase3_cymbal_cooccurrence_rescue`,
    `spectral_reclassify`, and `apply_crash_ride_arbiter` so it operates
    on the final emitted hit set.
    """
    import bisect
    crash_times = sorted(h.time_ms for h in chart.hits if h.lane == 4 and h.is_cymbal)
    if not crash_times:
        logger.info("  Crash+Ride co-stack veto: no Crash hits, skipping")
        return chart

    n_dropped = 0
    new_hits = []
    for hit in chart.hits:
        if hit.lane == 3 and hit.is_cymbal:
            pos = bisect.bisect_left(crash_times, hit.time_ms - stack_tol_ms)
            # any crash within ±stack_tol_ms?
            stacked = False
            while pos < len(crash_times) and crash_times[pos] <= hit.time_ms + stack_tol_ms:
                stacked = True
                break
            if stacked:
                n_dropped += 1
                continue
        new_hits.append(hit)

    if n_dropped > 0:
        logger.info(
            f"  Crash+Ride co-stack veto: dropped {n_dropped} Ride hits stacked on Crash "
            f"(±{stack_tol_ms:.0f}ms)"
        )
    else:
        logger.info("  Crash+Ride co-stack veto: 0 dropped")

    return DrumChart(
        hits=new_hits,
        tempo_events=chart.tempo_events,
        time_signatures=chart.time_signatures,
        ticks_per_beat=chart.ticks_per_beat,
    )


def apply_snare_in_hh_roll_veto(
    chart: DrumChart,
    stems_dir: Path,
    hh_window_ms: float = 300.0,
    min_hh_neighbours: int = 4,
    max_other_snares: int = 1,
    attack_db: float = -8.0,
) -> DrumChart:
    """Suppress phantom Snare hits stacked inside dense HiHat rolls.

    Bug #1 (Apr 2026 ear-test): during dense HH-roll passages (16th-note
    hi-hat patterns) the classifier sometimes fires Snare on every other
    onset alongside HiHat, where the truth is HiHat-only. Builder emits
    them stacked, surviving Phase 3.

    Music-context veto, mirroring the fill-end veto pattern that fixed
    bug #3:
      1. Onset must be a Snare (lane 1).
      2. ≥`min_hh_neighbours` HiHat hits within ±`hh_window_ms` (dense roll).
      3. ≤`max_other_snares` other Snares in the same window (this snare
         is isolated, not part of a real backbeat pair).
      4. snare.wav must show a clear DROP after the onset
         (post-RMS - pre-RMS < `attack_db`, default -8 dB). Real snare
         hits show post >= pre (a transient). Phantom snares stacked on
         top of an HH roll show post << pre because the snare stem only
         carries decaying bleed from a previous real snare. The strict
         -8 dB requirement was tuned on Every Avenue (Apr 2026): clean
         separation between real backbeats (Δ ∈ [-7, +5] dB) and
         phantoms (Δ ∈ [-20, -10] dB), giving 100% veto precision.

    Genuine backbeats survive (3) — at typical 120–160 BPM the snare
    backbeats fall on beats 2 & 4 (~750ms apart), so a real backbeat in
    a ±300ms window is alone but still has a snare-stem attack (4).
    Two-snare flam fills survive (3).
    """
    snare_path = stems_dir / "snare.wav"
    if not snare_path.exists():
        logger.warning(f"  Snare-in-HH-roll veto: missing snare.wav, skipping")
        return chart

    snare_audio, sr = sf.read(str(snare_path), always_2d=False)
    if snare_audio.ndim > 1:
        snare_audio = snare_audio.mean(axis=1)
    snare_audio = snare_audio.astype(np.float32, copy=False)

    hh_times = sorted(h.time_ms for h in chart.hits if h.lane == 2 and h.is_cymbal)
    snare_times = sorted(h.time_ms for h in chart.hits if h.lane == 1)

    import bisect
    n_vetoed = 0
    deltas = []
    new_hits = []
    for hit in chart.hits:
        if hit.lane != 1 or hit.is_cymbal:
            new_hits.append(hit)
            continue
        # (2) hi-hat density
        lo = bisect.bisect_left(hh_times, hit.time_ms - hh_window_ms)
        hi = bisect.bisect_right(hh_times, hit.time_ms + hh_window_ms)
        if (hi - lo) < min_hh_neighbours:
            new_hits.append(hit)
            continue
        # (3) snare isolation
        slo = bisect.bisect_left(snare_times, hit.time_ms - hh_window_ms)
        shi = bisect.bisect_right(snare_times, hit.time_ms + hh_window_ms)
        if (shi - slo) - 1 > max_other_snares:
            new_hits.append(hit)
            continue
        # (4) snare-stem attack check
        t_sec = hit.time_ms / 1000.0
        has_attack, delta = _toms_has_attack(
            snare_audio, sr, t_sec, attack_db=attack_db,
        )
        if has_attack:
            new_hits.append(hit)
            continue
        # All conditions met — phantom snare in HH roll
        n_vetoed += 1
        deltas.append(delta)

    if n_vetoed > 0:
        mm = float(np.mean(deltas))
        logger.info(
            f"  Snare-in-HH-roll veto: {n_vetoed} suppressed "
            f"(μ snare attack Δ={mm:+.1f}dB, threshold={attack_db}dB)"
        )
    else:
        logger.info("  Snare-in-HH-roll veto: 0 suppressed")

    return DrumChart(
        hits=new_hits,
        tempo_events=chart.tempo_events,
        time_signatures=chart.time_signatures,
        ticks_per_beat=chart.ticks_per_beat,
    )


def apply_kick_double_rescue(
    chart: DrumChart,
    stems_dir: Path,
    gap_min_ms: float = 220.0,
    gap_max_ms: float = 400.0,
    midpoint_tol_ms: float = 50.0,
    overlap_tol_ms: float = 60.0,
    onset_delta: float = 0.03,
    onset_wait: int = 1,
    attack_db: float = 4.0,
) -> DrumChart:
    """Rescue missed kicks in fast double/triple-kick patterns.

    Bug #2 (Apr 2026 ear-test): in fast kick passages (e.g. Yellowcard
    Miles Apart), V14 onset detector's min-distance constraint suppresses
    the middle of 3-kick triples, leaving 2 detected kicks separated by a
    gap of roughly twice the local kick spacing.

    Music-context rescue:
      1. Find pairs of consecutive chart kicks separated by `gap_min_ms`
         to `gap_max_ms` (suggests one missed middle kick).
      2. Run librosa onset detection on `kick.wav` (loose params).
      3. If a kick-stem onset exists at the midpoint ±`midpoint_tol_ms`
         AND no chart hit exists there within ±`overlap_tol_ms`
         AND kick.wav shows a discrete attack (≥`attack_db`),
         insert a Kick (lane 0) at that time.

    Validated 2026-04-28 on Miles Apart: gap [220,400]ms produced 4
    candidates with 100% precision (4/4 matched GT kicks).
    """
    kick_path = stems_dir / "kick.wav"
    if not kick_path.exists():
        logger.warning(f"  Kick double-rescue: missing kick.wav, skipping")
        return chart

    kick_audio, sr = sf.read(str(kick_path), always_2d=False)
    if kick_audio.ndim > 1:
        kick_audio = kick_audio.mean(axis=1)
    kick_audio = kick_audio.astype(np.float32, copy=False)

    # Loose librosa onset detection: catches softer kicks that V14 missed.
    stem_onsets = librosa.onset.onset_detect(
        y=kick_audio, sr=sr, units='time', hop_length=512,
        backtrack=False, delta=onset_delta, wait=onset_wait,
    )
    stem_onsets = np.asarray(stem_onsets, dtype=np.float32)

    # Pre-sort chart kick times for O(log N) overlap check.
    # We only check overlap against existing kicks (lane 0) — midpoints
    # routinely coincide with backbeat snare/HH, so checking all hits
    # would veto every candidate.
    chart_kicks = sorted(h.time_ms for h in chart.hits if h.lane == 0)

    if len(chart_kicks) < 4:
        logger.info("  Kick double-rescue: 0 added (too few kicks)")
        return chart

    # Estimate per-song chart→audio time offset by taking median of
    # (chart_kick − nearest_stem_onset) for matched-up kicks. Some songs
    # have 50–150ms drift between MIDI chart times and raw audio.
    stem_onsets_ms = stem_onsets * 1000.0
    diffs = []
    for k in chart_kicks:
        d = stem_onsets_ms - k
        j = int(np.argmin(np.abs(d)))
        if abs(d[j]) <= 150.0:
            diffs.append(float(-d[j]))  # offset = chart - stem
    if not diffs:
        logger.info("  Kick double-rescue: 0 added (no stem alignment)")
        return chart
    chart_audio_offset_ms = float(np.median(diffs))

    import bisect
    new_kicks_ms: list[float] = []
    for i in range(len(chart_kicks) - 1):
        gap = chart_kicks[i + 1] - chart_kicks[i]
        if not (gap_min_ms <= gap <= gap_max_ms):
            continue
        # Audio-time midpoint = chart-time midpoint − offset.
        midpoint_audio_ms = 0.5 * (chart_kicks[i] + chart_kicks[i + 1]) - chart_audio_offset_ms
        # Find closest stem onset within ±midpoint_tol_ms (in audio time)
        dist = np.abs(stem_onsets_ms - midpoint_audio_ms)
        j = int(np.argmin(dist))
        if dist[j] > midpoint_tol_ms:
            continue
        cand_audio_ms = float(stem_onsets_ms[j])
        # Convert back to chart time for insertion + overlap check.
        cand_chart_ms = cand_audio_ms + chart_audio_offset_ms
        # Skip if any chart KICK already exists in overlap window.
        lo = bisect.bisect_left(chart_kicks, cand_chart_ms - overlap_tol_ms)
        hi = bisect.bisect_right(chart_kicks, cand_chart_ms + overlap_tol_ms)
        if hi > lo:
            continue
        # Require kick-stem to show a fresh attack at the candidate (audio time).
        has_atk, _ = _toms_has_attack(
            kick_audio, sr, cand_audio_ms / 1000.0, attack_db=attack_db,
        )
        if not has_atk:
            continue
        new_kicks_ms.append(cand_chart_ms)

    if not new_kicks_ms:
        logger.info("  Kick double-rescue: 0 added")
        return chart

    # Build new hits list with the rescued kicks inserted.
    # Velocity 90 (slightly below typical 100) marks rescued hits.
    rescued_hits = [
        DrumHit(time_ms=t, tick=0, lane=0, is_cymbal=False, velocity=90)
        for t in new_kicks_ms
    ]
    merged = list(chart.hits) + rescued_hits
    merged.sort(key=lambda h: (h.time_ms, h.lane))
    logger.info(f"  Kick double-rescue: {len(new_kicks_ms)} added")
    return DrumChart(
        hits=merged,
        tempo_events=chart.tempo_events,
        time_signatures=chart.time_signatures,
        ticks_per_beat=chart.ticks_per_beat,
    )


def apply_crash_ride_arbiter(
    chart: DrumChart,
    stems_dir: Path,
    win_ms: float = 50.0,
    margin_db: float = 6.0,
    floor_db: float = -45.0,
) -> DrumChart:
    """Pairwise crash-vs-ride disambiguation on cymbal hits.

    Same proven pattern as `apply_drumsep_hits_arbiter` (tom-vs-cymbal): for
    each cymbal hit on lane 3 (Ride) or lane 4 (Crash), compare crash.wav RMS
    vs ride.wav RMS in a 50 ms window centered on the onset.  If the
    competing stem dominates by `margin_db` AND is itself above `floor_db`,
    flip the lane (3↔4).  Both stems share similar typical loudness ranges
    so a tighter margin (~6 dB) than the tom-vs-cymbal arbiter (20 dB) works.

    Targets the largest residual error class in v23 (~425 Crash↔Ride
    confusions across 5 evaluable songs).  Pairwise comparison is robust to
    bleed because both stems experience similar leakage; the relative
    ordering survives.
    """
    needed = ["ride", "crash"]
    stems = {}
    sr = None
    for name in needed:
        p = stems_dir / f"{name}.wav"
        if not p.exists():
            logger.warning(f"  Crash/ride arbiter: missing stem {p}, skipping")
            return chart
        x, x_sr = sf.read(str(p), always_2d=False)
        if sr is None:
            sr = x_sr
        if x.ndim > 1:
            x = x.mean(axis=1)
        stems[name] = x.astype(np.float32, copy=False)

    # lane 3 = Ride; lane 4 = Crash.  paired_stem_for[lane] is the OTHER stem.
    own_for = {3: "ride", 4: "crash"}
    other_for = {3: "crash", 4: "ride"}
    other_lane = {3: 4, 4: 3}
    n_flipped = {3: 0, 4: 0}  # lane index BEFORE flip
    margins = {3: [], 4: []}
    new_hits = []
    for hit in chart.hits:
        if not hit.is_cymbal or hit.lane not in own_for:
            new_hits.append(hit)
            continue
        t = hit.time_ms / 1000.0
        own_db = _stem_rms_db_at(stems[own_for[hit.lane]], sr, t, win_ms)
        oth_db = _stem_rms_db_at(stems[other_for[hit.lane]], sr, t, win_ms)
        if oth_db < floor_db:
            new_hits.append(hit)
            continue
        margin = oth_db - own_db
        if margin >= margin_db:
            new_hits.append(DrumHit(
                time_ms=hit.time_ms, tick=hit.tick, lane=other_lane[hit.lane],
                is_cymbal=True, velocity=hit.velocity,
            ))
            n_flipped[hit.lane] += 1
            margins[hit.lane].append(margin)
        else:
            new_hits.append(hit)

    total = sum(n_flipped.values())
    if total > 0:
        bits = []
        if n_flipped[3]:
            mm = float(np.mean(margins[3]))
            bits.append(f"ride→crash={n_flipped[3]} (μ={mm:+.1f}dB)")
        if n_flipped[4]:
            mm = float(np.mean(margins[4]))
            bits.append(f"crash→ride={n_flipped[4]} (μ={mm:+.1f}dB)")
        logger.info(f"  Crash/ride arbiter: {total} flips ({', '.join(bits)})")
    else:
        logger.info("  Crash/ride arbiter: 0 flips")

    return DrumChart(
        hits=new_hits,
        tempo_events=chart.tempo_events,
        time_signatures=chart.time_signatures,
        ticks_per_beat=chart.ticks_per_beat,
    )


def apply_stem_attack_veto(
    chart: DrumChart,
    stems_dir: Path,
    attack_db: float = 1.0,
    active_floor_db: float = -50.0,
    loud_rescue_db: float = -25.0,
    pre_offset_ms: float = -30.0,
    pre_win_ms: float = 25.0,
    post_offset_ms: float = 0.0,
    post_win_ms: float = 20.0,
) -> DrumChart:
    """Foundational per-hit stem-attack veto.

    For every emitted hit, looks up the matching 6-stem (kick/snare/toms/
    hh/ride/crash) and requires evidence that this drum actually fired
    at this moment. Three independent paths to keep a hit:
      1. Attack present: post_RMS >= pre_RMS + attack_db (fresh transient)
      2. Loud rescue: post_RMS >= loud_rescue_db (loud sound, assumed real
         even without clean attack — handles dense mixes / overlapping
         strikes where the pre-window already contains sustained energy)
      3. Default: drop

    Plus: stem must be at least active (post >= active_floor_db) — if the
    stem is essentially silent, nothing real happened on this drum.

    Catches structural error families that arbiters can't fix:
      - Phantom hits where the model fired but no actual drum struck
      - Overcharting (extra tom in a roll, ride+crash where only crash
        played) — only the stems with real attacks survive
      - Cross-class leakage (HH-roll snares in Every Avenue) — snare
        stem is silent during HH-only sections, so phantom snares drop

    Tunable via env: STRUM_STEM_VETO_ATTACK_DB, STRUM_STEM_VETO_FLOOR_DB,
    STRUM_STEM_VETO_LOUD_DB.
    """
    # (lane, is_cymbal) → stem name
    lane_cym_to_stem = {
        (0, False): "kick",
        (1, False): "snare",
        (2, True):  "hh",
        (2, False): "toms",
        (3, True):  "ride",
        (3, False): "toms",
        (4, True):  "crash",
        (4, False): "toms",
    }
    needed = ["kick", "snare", "toms", "hh", "ride", "crash"]
    stems = {}
    sr = None
    for name in needed:
        p = stems_dir / f"{name}.wav"
        if not p.exists():
            logger.warning(f"  Stem-attack veto: missing stem {p}, skipping")
            return chart
        x, x_sr = sf.read(str(p), always_2d=False)
        if sr is None:
            sr = x_sr
        if x.ndim > 1:
            x = x.mean(axis=1)
        stems[name] = x.astype(np.float32, copy=False)

    n_dropped = {n: 0 for n in needed}
    n_kept = {n: 0 for n in needed}
    new_hits = []
    for hit in chart.hits:
        key = (hit.lane, hit.is_cymbal)
        stem_name = lane_cym_to_stem.get(key)
        if stem_name is None:
            new_hits.append(hit)
            continue
        t = hit.time_ms / 1000.0
        post_db = _stem_rms_db_window(
            stems[stem_name], sr, t, post_offset_ms, post_win_ms,
        )
        pre_db = _stem_rms_db_window(
            stems[stem_name], sr, t, pre_offset_ms, pre_win_ms,
        )
        attack_delta = post_db - pre_db
        # Three keep paths:
        #   (1) clean attack present
        #   (2) loud audio there (real hit buried in mix → no clean delta)
        # Plus active gate: stem essentially silent → drop
        if post_db < active_floor_db:
            n_dropped[stem_name] += 1
        elif attack_delta >= attack_db or post_db >= loud_rescue_db:
            new_hits.append(hit)
            n_kept[stem_name] += 1
        else:
            n_dropped[stem_name] += 1

    total_dropped = sum(n_dropped.values())
    total_kept = sum(n_kept.values())
    if total_dropped > 0:
        bits = [f"{n}={n_dropped[n]}" for n in needed if n_dropped[n] > 0]
        logger.info(
            f"  Stem-attack veto: dropped {total_dropped}/{total_dropped + total_kept} hits "
            f"({', '.join(bits)})"
        )
    else:
        logger.info(f"  Stem-attack veto: 0 dropped (all {total_kept} hits corroborated)")

    return DrumChart(
        hits=new_hits,
        tempo_events=chart.tempo_events,
        time_signatures=chart.time_signatures,
        ticks_per_beat=chart.ticks_per_beat,
    )


def analyze_audio(audio_path: Path) -> dict:
    """Analyze audio for tempo, tempo changes, and duration.

    Returns a dict with:
      - tempo_bpm: global BPM (refined via grid-alignment search)
      - tempo_events: list of TempoEvent for tempo changes
      - duration_ms / duration_sec: audio length
    """
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    duration_sec = len(y) / sr

    # --- Beat tracking ---
    # Note: do NOT pass onset_envelope to beat_track – its internal onset
    # strength computation uses different defaults and produces more stable
    # beat positions than a separately-computed envelope.
    tempo_est, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Initial BPM from median IBI (robust to outlier beats).
    # Median IBI is more reliable than librosa's global estimate because
    # it is not fooled by octave ambiguity or short-term noise.
    if len(beat_times) >= 4:
        ibis = np.diff(beat_times)
        base_bpm = 60.0 / np.median(ibis)
    else:
        base_bpm = float(tempo_est[0]) if hasattr(tempo_est, "__len__") else float(tempo_est)

    # Normalize to 60-200 range
    while base_bpm < 60:
        base_bpm *= 2
    while base_bpm > 200:
        base_bpm /= 2

    # --- Grid-alignment BPM refinement ---
    # Two-pass search: first ±5 BPM at 0.1 resolution, then ±0.5 BPM
    # at 0.01 resolution around the best candidate.  This gives much
    # lower drift than 0.1 alone (0.01 BPM error ≈ 14ms drift in 4 min
    # vs 140ms for 0.1 BPM error).
    bpm = round(base_bpm * 2) / 2  # fallback: nearest 0.5
    if len(beat_times) >= 8:
        def _grid_search(bt, lo_100x, hi_100x, scale):
            """Search BPMs in [lo/scale, hi/scale] and return best."""
            best_err = float("inf")
            best_bpm = lo_100x / scale
            for bpm_int in range(lo_100x, hi_100x):
                cand = bpm_int / scale
                per = 60.0 / cand
                phases = (bt % per) / per * 2 * np.pi
                mean_phase = np.arctan2(np.sin(phases).mean(), np.cos(phases).mean())
                if mean_phase < 0:
                    mean_phase += 2 * np.pi
                offset = mean_phase / (2 * np.pi) * per
                residuals = (bt - offset) % per
                errors = np.minimum(residuals, per - residuals)
                me = float(np.mean(errors))
                if me < best_err:
                    best_err = me
                    best_bpm = cand
            return best_bpm

        # Pass 1: coarse (0.1 BPM), ±5 BPM
        lo1 = max(400, int(base_bpm * 10) - 50)
        hi1 = int(base_bpm * 10) + 51
        coarse_bpm = _grid_search(beat_times, lo1, hi1, 10.0)

        # Pass 2: fine (0.01 BPM), ±0.5 BPM around coarse result
        lo2 = max(4000, int(coarse_bpm * 100) - 50)
        hi2 = int(coarse_bpm * 100) + 51
        bpm = _grid_search(beat_times, lo2, hi2, 100.0)

    # --- Tempo change detection ---
    # Use windowed median of IBIs.  Only flag a change if the local BPM
    # deviates from the current segment BPM by >3 BPM for >=8 consecutive
    # beats.  Beat tracker noise has std ≈ 1.7 BPM, so 3 BPM avoids
    # false positives while catching real tempo shifts.
    tempo_events: list[TempoEvent] = []
    # v25.6: tempo change detection is OFF by default. Empirical study of 12
    # GT charts shows 12/12 are single-tempo. The previous detector produced
    # false positives via half-tempo beat-tracker glitches (~250 BPM windows
    # halved to 124.69 BPM "change" against true 125), causing Clone Hero to
    # render bar lines with wrong BPM in the spurious segment, breaking
    # bar-vs-note alignment. Set STRUM_TEMPO_CHANGE_DETECT=1 to re-enable.
    enable_tc = os.environ.get("STRUM_TEMPO_CHANGE_DETECT", "0") == "1"
    if enable_tc and len(beat_times) >= 12:
        ibis = np.diff(beat_times)
        window = 4
        local_bpms = []
        local_times = []
        for i in range(len(ibis) - window + 1):
            local_bpms.append(60.0 / np.median(ibis[i : i + window]))
            local_times.append(beat_times[i + window // 2])

        # Normalize local BPMs to be octave-close to global BPM so that
        # half/double-tempo beat-tracker glitches do not register as changes.
        def _octave_normalize(b: float, target: float) -> float:
            if b <= 0 or target <= 0:
                return b
            while b < target * 0.7:
                b *= 2.0
            while b > target * 1.4:
                b /= 2.0
            return b
        local_bpms = [_octave_normalize(x, bpm) for x in local_bpms]

        # Walk through local BPMs and detect change points
        current_bpm = bpm  # start with global refined BPM
        tempo_events.append(TempoEvent(tick=0, tempo_bpm=round(current_bpm, 2), time_ms=0.0))
        CHANGE_THRESH = 5.0  # BPM (raised from 3.0; beat-tracker IBI noise std ≈1.7)
        MIN_PERSIST = 16  # consecutive windows (~16 beats; raised from 8)

        i = 0
        while i < len(local_bpms):
            if abs(local_bpms[i] - current_bpm) > CHANGE_THRESH:
                # Check persistence
                new_bpm_values = []
                j = i
                while j < len(local_bpms) and abs(local_bpms[j] - current_bpm) > CHANGE_THRESH:
                    new_bpm_values.append(local_bpms[j])
                    j += 1
                if len(new_bpm_values) >= MIN_PERSIST:
                    new_bpm = round(float(np.median(new_bpm_values)), 2)
                    # Normalize to 60-200
                    while new_bpm < 60:
                        new_bpm *= 2
                    while new_bpm > 200:
                        new_bpm /= 2
                    change_time_ms = round(local_times[i] * 1000, 1)
                    tempo_events.append(TempoEvent(tick=0, tempo_bpm=new_bpm, time_ms=change_time_ms))
                    current_bpm = new_bpm
                    i = j
                    continue
            i += 1
    else:
        tempo_events.append(TempoEvent(tick=0, tempo_bpm=round(bpm, 2), time_ms=0.0))

    # v26 phase offset: use the FIRST detected beat, mod beat_period.
    #
    # Earlier versions averaged the phase of all 100+ librosa beats (circular
    # mean).  That looked robust on paper but failed in practice: the
    # grid-aligned BPM (e.g. 82.50) and the BPM librosa actually used to place
    # beats (e.g. 83.35) do not match exactly, so beat phases drift through
    # the modulo and the mean lands far from the true downbeat.  On Here
    # Tonight that produced phase=571 ms when the real first downbeat is at
    # 162 ms — every note was shifted >½ bar past the audio.
    #
    # The first beat is one precisely-located sample (within ±1 librosa hop ≈
    # ±23 ms) and corresponds to an actual onset.  It is the best reference
    # for "where does bar 1 start in the audio" and removes all averaging
    # error.  All hits + the audio start are then shifted by the same value
    # so MIDI tick 0 lines up with the audio's first downbeat in Clone Hero.
    beat_period_ms = 60_000.0 / bpm
    first_beat_ms = float(beat_times[0]) * 1000.0 if len(beat_times) > 0 else 0.0
    if len(beat_times) >= 1:
        phase_offset_ms = first_beat_ms % beat_period_ms
        # Snap closer side: if past half period, trim forward to next beat.
        if phase_offset_ms > beat_period_ms / 2:
            phase_offset_ms -= beat_period_ms
        if phase_offset_ms < 0:
            phase_offset_ms += beat_period_ms
        phase_source = "first_beat"
    else:
        phase_offset_ms = 0.0
        phase_source = "none"

    # Optional small calibration for Clone Hero's audio output latency.
    # Default 0 — first_beat-derived phase is already accurate to a frame.
    # Tune via STRUM_PHASE_CALIBRATE_MS if needed.
    calibrate_ms = float(os.environ.get("STRUM_PHASE_CALIBRATE_MS", "0.0"))
    phase_offset_ms = phase_offset_ms + calibrate_ms
    if phase_offset_ms >= beat_period_ms:
        phase_offset_ms -= beat_period_ms
    if phase_offset_ms < 0:
        phase_offset_ms += beat_period_ms

    logger.info(
        f"  Tempo: {bpm:.2f} BPM, {len(tempo_events)} tempo event(s); "
        f"phase_offset={phase_offset_ms:.1f} ms ({phase_source}, "
        f"first beat at {first_beat_ms:.1f} ms)"
    )

    return {
        "tempo_bpm": round(bpm, 2),
        "tempo_events": tempo_events,
        "duration_ms": duration_sec * 1000,
        "duration_sec": duration_sec,
        "phase_offset_ms": float(phase_offset_ms),
        "first_beat_ms": float(first_beat_ms),
    }


# ══════════════════════════════════════════════════════════════
# Stage 1: V14 onset detection
# ══════════════════════════════════════════════════════════════

def detect_onsets_v14(
    model: TwoStageDrumsCRNN,
    audio_path: Path,
    device: torch.device,
    onset_threshold: float = 0.4,
    segment_duration: float = 10.0,
    overlap: float = 0.5,
    min_distance_ms: float = 20.0,
) -> tuple[list[float], np.ndarray]:
    """Run V14 onset detector and return onset times + class probs.

    Returns:
        Tuple of (onset_times_ms, v14_class_probs).
        v14_class_probs: (N, 8) array of V14 frame-level class predictions
        at each detected onset.  The BiLSTM has full-song context so its
        tom/cymbal discrimination is far superior to the isolated-window
        ensemble (99%+ accuracy on conflict pairs).
    """
    # Load audio
    y, sr = librosa.load(str(audio_path), sr=V14_SR, mono=True)
    audio = torch.from_numpy(y).unsqueeze(0)

    # Compute mel spectrogram (same as V14 training, optionally BG-subtracted)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=V14_SR, n_fft=V14_N_FFT, hop_length=V14_HOP, n_mels=V14_N_MELS,
    )
    mel_spec = _compute_log_mel_with_optional_bg_subtract(audio, mel_transform, model_tag="v14")

    total_frames = mel_spec.shape[-1]
    segment_frames = int(segment_duration * V14_SR / V14_HOP)
    hop_frames = int(segment_frames * (1 - overlap))

    # Overlap-averaged onset probabilities
    onset_probs = np.zeros(total_frames)
    cymbal_onset_probs = np.zeros(total_frames) if model.has_cymbal_onset else None
    mc_onset_probs = np.zeros((total_frames, 8), dtype=np.float32) if model.has_multiclass_onset else None
    class_probs_avg = np.zeros((total_frames, 8), dtype=np.float32)
    count = np.zeros(total_frames)

    num_segments = max(1, (total_frames - segment_frames) // hop_frames + 1)

    with torch.no_grad():
        for i in tqdm(range(num_segments), desc="  V14 onset detection", leave=False):
            start = i * hop_frames
            end = min(start + segment_frames, total_frames)
            segment = mel_spec[:, :, start:end]
            if segment.shape[-1] < segment_frames:
                segment = torch.nn.functional.pad(segment, (0, segment_frames - segment.shape[-1]))
            segment = segment.unsqueeze(0).to(device)
            outputs = model(segment)
            probs = outputs["onset_probs"].squeeze(0).squeeze(-1).cpu().numpy()
            c_probs = outputs["class_probs"].squeeze(0).cpu().numpy()  # (T, 8)
            actual_len = min(end - start, len(probs))
            onset_probs[start:start + actual_len] += probs[:actual_len]
            class_probs_avg[start:start + actual_len] += c_probs[:actual_len]
            if cymbal_onset_probs is not None:
                cym_probs = outputs["cymbal_onset_probs"].squeeze(0).squeeze(-1).cpu().numpy()
                cymbal_onset_probs[start:start + actual_len] += cym_probs[:actual_len]
            if mc_onset_probs is not None:
                mc_probs = outputs["mc_onset_probs"].squeeze(0).cpu().numpy()  # (T, 8)
                mc_onset_probs[start:start + actual_len] += mc_probs[:actual_len]
            count[start:start + actual_len] += 1

    count = np.maximum(count, 1)
    onset_probs /= count
    class_probs_avg /= count[:, None]
    if cymbal_onset_probs is not None:
        cymbal_onset_probs /= count
    if mc_onset_probs is not None:
        mc_onset_probs /= count[:, None]

    # Peak detection — primary onset detector
    min_distance_frames = max(1, int(min_distance_ms / 1000 * V14_SR / V14_HOP))
    peaks, properties = find_peaks(
        onset_probs,
        height=onset_threshold,
        distance=min_distance_frames,
    )
    primary_peaks = set(peaks.tolist())

    # Cymbal onset head: find additional peaks not in primary set
    cymbal_rescued = 0
    if cymbal_onset_probs is not None:
        cymbal_peaks, _ = find_peaks(
            cymbal_onset_probs,
            height=CYMBAL_ONSET_THRESHOLD,
            distance=min_distance_frames,
        )
        for cp in cymbal_peaks:
            # Skip if already detected by primary onset detector (within min_distance)
            too_close = any(abs(cp - pp) <= min_distance_frames for pp in primary_peaks)
            if not too_close:
                primary_peaks.add(cp)
                cymbal_rescued += 1

    all_peaks = sorted(primary_peaks)

    # MC onset head: per-class rescue for rare classes
    mc_rescued = 0
    mc_rescued_classes = {}
    mc_rescue_class_probs = {}  # frame → MC class probs (used as v14_class_probs for rescued onsets)
    if mc_onset_probs is not None:
        mc_min_dist_frames = max(1, int(MC_MIN_DISTANCE_MS / 1000 * V14_SR / V14_HOP))
        sorted_peaks = np.array(all_peaks)  # for fast distance check

        for cls_idx in MC_RESCUE_CLASSES:
            # Find peaks in this class's onset probability channel
            cls_probs = mc_onset_probs[:, cls_idx]
            cls_peaks, _ = find_peaks(
                cls_probs,
                height=MC_RESCUE_THRESHOLD,
                distance=min_distance_frames,
            )
            for cp in cls_peaks:
                # Must have SOME V14 onset signal (not pure noise)
                if onset_probs[cp] < MC_ONSET_FLOOR:
                    continue
                # Skip if too close to an existing onset
                if len(sorted_peaks) > 0:
                    insert_pos = np.searchsorted(sorted_peaks, cp)
                    too_close = False
                    for check_pos in [insert_pos - 1, insert_pos]:
                        if 0 <= check_pos < len(sorted_peaks):
                            if abs(sorted_peaks[check_pos] - cp) <= mc_min_dist_frames:
                                too_close = True
                                break
                    if too_close:
                        continue
                # Rescue this onset
                primary_peaks.add(cp)
                sorted_peaks = np.sort(np.append(sorted_peaks, cp))
                mc_rescue_class_probs[cp] = mc_onset_probs[cp]  # (8,) MC class probs
                mc_rescued += 1
                mc_rescued_classes[cls_idx] = mc_rescued_classes.get(cls_idx, 0) + 1

        all_peaks = sorted(primary_peaks)

    # Convert frame indices to milliseconds
    onset_times_ms = [(frame * V14_HOP / V14_SR) * 1000 for frame in all_peaks]

    # Extract V14 class probs at each detected onset (±2 frame max)
    # For MC-rescued onsets, use MC head's class probs instead (it knows what class it detected)
    v14_class_probs = np.zeros((len(all_peaks), 8), dtype=np.float32)
    for idx, frame in enumerate(all_peaks):
        if frame in mc_rescue_class_probs:
            # MC-rescued: use MC head's per-class probs as the "V14" signal
            # This ensures the arbiter trusts the MC head's classification
            v14_class_probs[idx] = mc_rescue_class_probs[frame]
        else:
            f_start = max(0, frame - 2)
            f_end = min(total_frames, frame + 3)
            v14_class_probs[idx] = class_probs_avg[f_start:f_end].max(axis=0)

    mc_detail = ""
    if mc_rescued > 0:
        cls_detail = ", ".join(f"{CLASS_NAMES[c]}={n}" for c, n in sorted(mc_rescued_classes.items()))
        mc_detail = f" [+{mc_rescued} MC-rescued: {cls_detail}]"

    logger.info(f"  V14 detected {len(all_peaks)} onsets "
                f"(threshold={onset_threshold}, "
                f"prob range={onset_probs.min():.3f}–{onset_probs.max():.3f})"
                + (f" [+{cymbal_rescued} cymbal-rescued]" if cymbal_rescued else "")
                + mc_detail)

    return onset_times_ms, v14_class_probs, mc_onset_probs


# ══════════════════════════════════════════════════════════════
# Stage 2: Onset classification with ensemble
# ══════════════════════════════════════════════════════════════

def extract_onset_windows(
    audio_path: Path,
    onset_times_ms: list[float],
    needs_cqt: bool = True,
) -> dict:
    """Extract dual-resolution mel windows (+ optional CQT) at onset positions.

    Uses torchaudio mel transforms (matching training preprocessing exactly)
    and per-window extraction (not full-song slicing) for fidelity.

    Returns dict with:
        mel_fine: (N, 128, 87) float32
        mel_coarse: (N, 128, 44) float32
        cqt: (N, 128, 44) float32 or None
        valid_mask: (N,) bool — True if window fits within audio
    """
    y, sr = librosa.load(str(audio_path), sr=OC_SR, mono=True)
    audio_len = len(y)

    # torchaudio mel transforms — must match training preprocessing exactly
    fine_mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=OC_SR, n_fft=FINE_N_FFT, hop_length=FINE_HOP,
        n_mels=N_MELS, power=2.0,
    )
    coarse_mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=OC_SR, n_fft=COARSE_N_FFT, hop_length=COARSE_HOP,
        n_mels=N_MELS, power=2.0,
    )

    # Pre-compute full CQT if needed (librosa — same as training)
    cqt_full = None
    if needs_cqt:
        C = np.abs(librosa.cqt(
            y=y, sr=OC_SR, hop_length=CQT_HOP,
            fmin=CQT_FMIN, n_bins=CQT_N_BINS, bins_per_octave=CQT_BINS_PER_OCT,
        ))
        cqt_full = np.log(C + 1e-8).astype(np.float32)
        cqt_full = cqt_full[:CQT_TRIM_BINS, :]  # Trim 144 → 128

    N = len(onset_times_ms)
    mel_fine_windows = np.zeros((N, N_MELS, FINE_FRAMES), dtype=np.float32)
    mel_coarse_windows = np.zeros((N, N_MELS, COARSE_FRAMES), dtype=np.float32)
    cqt_windows = np.zeros((N, CQT_TRIM_BINS, CQT_FRAMES), dtype=np.float32) if needs_cqt else None
    valid_mask = np.ones(N, dtype=bool)

    for i, t_ms in enumerate(onset_times_ms):
        center_sample = int(t_ms / 1000 * OC_SR)
        start_sample = center_sample - WINDOW_BEFORE_SAMP
        end_sample = center_sample + WINDOW_AFTER_SAMP

        if start_sample < 0 or end_sample > audio_len:
            valid_mask[i] = False
            continue

        # Extract audio window and compute mel per-window (matches training)
        window = y[start_sample:end_sample]
        if len(window) < WINDOW_SAMPLES:
            window = np.pad(window, (0, WINDOW_SAMPLES - len(window)))
        audio_t = torch.from_numpy(window).float().unsqueeze(0)

        with torch.no_grad():
            mf = torch.log(fine_mel_transform(audio_t) + 1e-8).squeeze(0).numpy()
            mc = torch.log(coarse_mel_transform(audio_t) + 1e-8).squeeze(0).numpy()

        # Pad/trim to exact frame count (matches training preprocess)
        if mf.shape[-1] >= FINE_FRAMES:
            mel_fine_windows[i] = mf[:, :FINE_FRAMES]
        else:
            mel_fine_windows[i, :, :mf.shape[-1]] = mf

        if mc.shape[-1] >= COARSE_FRAMES:
            mel_coarse_windows[i] = mc[:, :COARSE_FRAMES]
        else:
            mel_coarse_windows[i, :, :mc.shape[-1]] = mc

        # CQT window (sliced from full-song CQT — same as training)
        if needs_cqt and cqt_full is not None:
            cqt_start = start_sample // CQT_HOP
            cqt_end = cqt_start + CQT_FRAMES
            if cqt_end <= cqt_full.shape[1]:
                cqt_windows[i] = cqt_full[:, cqt_start:cqt_end]
            else:
                avail = max(0, cqt_full.shape[1] - cqt_start)
                if avail > 0:
                    cqt_windows[i, :, :avail] = cqt_full[:, cqt_start:cqt_start + avail]

    return {
        "mel_fine": mel_fine_windows,
        "mel_coarse": mel_coarse_windows,
        "cqt": cqt_windows,
        "valid_mask": valid_mask,
    }


def build_context_vectors(onset_times_ms: list[float], class_probs: np.ndarray | None = None) -> np.ndarray:
    """Build onset context vectors: 4 prev + 4 next onsets' class probabilities.

    For the first pass (before classification), use zeros.
    For a second pass, use the class probabilities from the first.

    Returns: (N, 64) float32 — 8 classes × 8 neighbors
    """
    N = len(onset_times_ms)
    context = np.zeros((N, 64), dtype=np.float32)

    if class_probs is None:
        return context

    for i in range(N):
        vec = []
        # 4 previous onsets
        for j in range(i - 4, i):
            if 0 <= j < N:
                vec.extend(class_probs[j].tolist())
            else:
                vec.extend([0.0] * 8)
        # 4 next onsets
        for j in range(i + 1, i + 5):
            if 0 <= j < N:
                vec.extend(class_probs[j].tolist())
            else:
                vec.extend([0.0] * 8)
        context[i] = np.array(vec[:64], dtype=np.float32)

    return context


def classify_onsets_ensemble(
    ensemble: list[dict],
    windows: dict,
    context: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Run onset classifier ensemble with per-class model weighting.

    Uses PER_CLASS_WEIGHTS to weight each model differently for each class,
    then returns (N, 8) sigmoid probabilities.
    """
    N = windows["mel_fine"].shape[0]
    all_logits = []

    for entry in ensemble:
        model = entry["model"]
        name = entry["name"]
        needs_cqt = entry["needs_cqt"]

        model_logits = np.zeros((N, 8), dtype=np.float32)

        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)

                t_fine = torch.from_numpy(
                    windows["mel_fine"][start:end]
                ).unsqueeze(1).to(device)
                t_coarse = torch.from_numpy(
                    windows["mel_coarse"][start:end]
                ).unsqueeze(1).to(device)
                t_ctx = torch.from_numpy(context[start:end]).to(device)

                kwargs = {}
                if needs_cqt and windows["cqt"] is not None:
                    t_cqt = torch.from_numpy(
                        windows["cqt"][start:end]
                    ).unsqueeze(1).to(device)
                    kwargs["mel_lowfreq"] = t_cqt

                logits = model(t_fine, t_coarse, t_ctx, **kwargs)
                model_logits[start:end] = logits.cpu().numpy()

        all_logits.append(model_logits)
        logger.info(f"    {name}: inference done")

    # Per-class weighted average of logits
    # Each model gets a different weight for each class
    # Use ensemble_idx to map loaded models to their weight positions
    weighted_logits = np.zeros((N, 8), dtype=np.float32)
    for c in range(8):
        weights = PER_CLASS_WEIGHTS[c]
        total_w = 0.0
        for model_idx, entry in enumerate(ensemble):
            eidx = entry.get("ensemble_idx", model_idx)
            if eidx < len(weights):
                w = weights[eidx]
            else:
                w = 0.0
            if w > 0:
                weighted_logits[:, c] += w * all_logits[model_idx][:, c]
                total_w += w
        # Renormalize if some models were missing
        if total_w > 0 and total_w < 0.99:
            weighted_logits[:, c] /= total_w

    return weighted_logits  # return logits; sigmoid applied after spectral correction


# ══════════════════════════════════════════════════════════════
# Per-class thresholds & weights (optimized via eval_hybrid_ensemble.py
# on the test set using V2/V4/V6/V12c ensemble)
# ══════════════════════════════════════════════════════════════

# Per-class model weights (order matches ENSEMBLE_MODELS: V2, V4, V6, V12c, V15, V16, V17)
# V12c dominates most classes; V6 good on toms; V15 bleed robustness
# V16: FloorTom specialist (76.4% F1, only 1.3% FloorTom→Crash confusion)
# V17: Crash-aware (84.2% Crash F1, 85.7% recall) — rarest-class sampling fix
# NOTE: Crash uses V17 SOLO — weighted logit averaging lets 6 crash-blind models
#       outvote 1 crash-aware model, so V17 must be sole crash decision-maker
PER_CLASS_WEIGHTS = {
    0: [0.25, 0.0, 0.25, 0.30, 0.15, 0.05, 0.00],   # Kick      - V17 OFF (existing models fine)
    1: [0.25, 0.15, 0.10, 0.25, 0.20, 0.05, 0.00],   # Snare     - V17 OFF
    2: [0.25, 0.05, 0.15, 0.30, 0.20, 0.05, 0.00],   # HiHat     - V17 OFF
    3: [0.20, 0.0, 0.30, 0.20, 0.30, 0.00, 0.00],    # HighTom   - V17 OFF
    4: [0.15, 0.05, 0.25, 0.30, 0.25, 0.00, 0.00],   # Ride      - V17 OFF
    5: [0.15, 0.10, 0.10, 0.30, 0.30, 0.05, 0.00],   # LowTom    - V17 OFF
    6: [0.00, 0.0, 0.00, 0.00, 0.00, 0.00, 1.00],    # Crash     - V17 SOLO (84.2% F1, fixes 52% miss rate)
    7: [0.08, 0.02, 0.25, 0.15, 0.25, 0.25, 0.00],   # FloorTom  - V17 OFF
}

# Optional per-class V12c weight scaling for ensemble re-tuning experiments.
# STRUM_V12C_SCALE="kick:1.5,hihat:0.5,floortom:0.5" multiplies V12c (idx 3) weight
# in those classes by the given factor, then renormalizes the per-class row.
def _apply_v12c_scale_overrides() -> None:
    spec = os.environ.get("STRUM_V12C_SCALE")
    if not spec:
        return
    name_to_cls = {
        "kick": 0, "snare": 1, "hihat": 2, "hitom": 3, "hightom": 3,
        "ride": 4, "lotom": 5, "lowtom": 5, "crash": 6, "flotom": 7, "floortom": 7,
    }
    overrides: dict[int, float] = {}
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        k, v = tok.split(":")
        cls = name_to_cls[k.strip().lower()]
        overrides[cls] = float(v)
    for cls, scale in overrides.items():
        row = list(PER_CLASS_WEIGHTS[cls])
        if 3 < len(row):
            row[3] *= scale
            # Renormalize so the row still sums to ~1.0 (preserve total signal mass)
            s = sum(row)
            if s > 0:
                row = [w / s for w in row]
        PER_CLASS_WEIGHTS[cls] = row
        print(f"  STRUM_V12C_SCALE: class {cls} V12c×{scale:.2f} → row={[round(x,3) for x in row]}", flush=True)


_apply_v12c_scale_overrides()

DEFAULT_CLASS_THRESHOLDS = [
    # Demucs-domain calibration (2026-04-24): cymbal thresholds lowered
    # because Demucs degrades cymbal energy ~15% vs clean stems, leaving
    # 3,782 V14-detected onsets dropped at ensemble thresholding.
    # `moderate` config A/B: +820 correct (+2.24pp) on 23 held-out songs.
    float(os.environ.get("STRUM_TH_KICK",   "0.30")),  # Kick
    float(os.environ.get("STRUM_TH_SNARE",  "0.25")),  # Snare   (was 0.30)
    float(os.environ.get("STRUM_TH_HIHAT",  "0.35")),  # HiHat   (v25: was 0.25; freed Crash/Snare/Kick recall)
    float(os.environ.get("STRUM_TH_HITOM",  "0.12")),  # HighTom
    float(os.environ.get("STRUM_TH_RIDE",   "0.28")),  # Ride    (v25: was 0.18; freed Crash recall)
    float(os.environ.get("STRUM_TH_LOTOM",  "0.12")),  # LowTom
    float(os.environ.get("STRUM_TH_CRASH",  "0.35")),  # Crash   (was 0.55)
    float(os.environ.get("STRUM_TH_FLOTOM", "0.12")),  # FloorTom
]

# Lane pairs for tom/cymbal conflict resolution:
# (cymbal_class_idx, tom_class_idx)
LANE_CONFLICT_PAIRS = [
    (2, 3),  # Yellow: HiHat vs HighTom
    (4, 5),  # Blue: Ride vs LowTom
    (6, 7),  # Green: Crash vs FloorTom
]


# ══════════════════════════════════════════════════════════════
# Spectral tom/cymbal disambiguation
# ══════════════════════════════════════════════════════════════

def compute_spectral_features(audio_path: Path, onset_times_ms: list[float]) -> dict:
    """Compute spectral features at each onset for tom/cymbal reclassification.

    Computes:
      - low_ratio: energy ratio of 200-1000 Hz vs 3000-10000 Hz
        High ratio (>3) = tonal/tom-like, Low ratio (<0.5) = broadband/cymbal-like

    Returns:
        dict with 'low_ratio': (N,) array
    """
    y, sr = librosa.load(str(audio_path), sr=OC_SR, mono=True)
    n_onsets = len(onset_times_ms)
    low_ratios = np.ones(n_onsets, dtype=np.float32)  # neutral default

    window_samples = int(0.040 * sr)  # 40ms analysis window
    offset_samples = int(0.005 * sr)  # 5ms after onset

    for i, t_ms in enumerate(onset_times_ms):
        center = int(t_ms / 1000 * sr)
        start = center + offset_samples
        end = start + window_samples

        if start < 0 or end > len(y):
            continue

        window = y[start:end]
        if np.max(np.abs(window)) < 1e-6:
            continue

        S = np.abs(np.fft.rfft(window * np.hanning(len(window))))
        freqs = np.fft.rfftfreq(len(window), 1.0 / sr)

        low_mask = (freqs >= 200) & (freqs <= 1000)
        high_mask = (freqs >= 3000) & (freqs <= 10000)
        low_energy = np.sum(S[low_mask] ** 2)
        high_energy = np.sum(S[high_mask] ** 2)
        low_ratios[i] = low_energy / (high_energy + 1e-10)

    return {"low_ratio": low_ratios}


def _compute_onset_rms(
    audio_path: Path,
    onset_times_ms: list[float],
    window_ms: float = 500.0,
) -> np.ndarray:
    """Compute local RMS energy around each onset from the drums stem.

    Uses a wide window (500ms) to capture the section-level energy rather
    than just the onset transient. This reliably separates real drum
    sections (RMS > 0.04) from Demucs bleed regions (RMS < 0.03).

    Returns an array of shape (N,).
    """
    y, sr = librosa.load(str(audio_path), sr=OC_SR, mono=True)
    n = len(onset_times_ms)
    rms = np.zeros(n, dtype=np.float32)
    half_win = int(window_ms / 1000 * sr / 2)

    for i, t_ms in enumerate(onset_times_ms):
        center = int(t_ms / 1000 * sr)
        start = max(0, center - half_win)
        end = min(len(y), center + half_win)
        if end > start:
            rms[i] = np.sqrt(np.mean(y[start:end] ** 2))

    return rms


def _compute_spectral_centroid_features(
    audio_path: Path,
    onset_times_ms: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute spectral centroid and high-frequency energy percentage per onset.

    Used by the spectral safety net in build_chart to identify cymbal hits
    that are spectrally tom-like (low centroid, no high-freq shimmer).

    Returns:
        (centroids, high_pcts): arrays of shape (N,)
    """
    y, sr = librosa.load(str(audio_path), sr=OC_SR, mono=True)
    n = len(onset_times_ms)
    centroids = np.full(n, 5000.0, dtype=np.float32)  # default = cymbal-like
    high_pcts = np.full(n, 50.0, dtype=np.float32)

    window_samples = int(0.040 * sr)
    offset_samples = int(0.005 * sr)

    for i, t_ms in enumerate(onset_times_ms):
        center = int(t_ms / 1000 * sr)
        start = center + offset_samples
        end = start + window_samples
        if start < 0 or end > len(y):
            continue
        chunk = y[start:end]
        if np.max(np.abs(chunk)) < 1e-6:
            continue

        S = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
        freqs = np.fft.rfftfreq(len(chunk), 1.0 / sr)

        total_s = np.sum(S)
        if total_s > 0:
            centroids[i] = np.sum(freqs * S) / total_s

        total_e = np.sum(S ** 2)
        if total_e > 0:
            high_mask = freqs > 3000
            high_pcts[i] = np.sum(S[high_mask] ** 2) / total_e * 100

    return centroids, high_pcts


def apply_spectral_reclassification(
    probs: np.ndarray,
    spectral: dict,
    onset_times_ms: list[float],
    tom_ratio_threshold: float = 5.0,
    transfer_pct: float = 0.4,
    kick_gate: float = 0.25,
    cymbal_cap: float = 0.80,
) -> np.ndarray:
    """Reclassify cymbal hits as toms using fill detection + spectral evidence.

    Two-stage approach:
    1. Detect fill candidate regions: rapid bursts of 3+ hits where hihat/ride
       drops out and spectral ratio is elevated. Aggressively reclassify within
       confirmed fill regions.
    2. For remaining onsets, apply conservative spectral reclassification only
       when spectral evidence is very strong (ratio > 2*threshold) and kick
       probability is low.
    """
    corrected = probs.copy()
    pairs = [(2, 3), (4, 5), (6, 7)]  # (cymbal_idx, tom_idx) = (HiHat/HighTom, Ride/LowTom, Crash/FloorTom)
    low_ratio = spectral["low_ratio"]
    times = np.array(onset_times_ms)
    n = len(times)

    # Diagnostic: gated by STRUM_FILL_DEBUG=<start_ms>:<end_ms> (e.g. "10000:25000")
    # Logs each candidate cluster overlapping that window so we can see
    # WHY fills aren't rescuing tom hits.
    _dbg_env = os.environ.get("STRUM_FILL_DEBUG", "")
    if _dbg_env and ":" in _dbg_env:
        try:
            _dbg_lo, _dbg_hi = (float(x) for x in _dbg_env.split(":", 1))
        except ValueError:
            _dbg_lo, _dbg_hi = None, None
    else:
        _dbg_lo, _dbg_hi = None, None
    _dbg_active = _dbg_lo is not None and _dbg_hi is not None

    # ── Stage 1: Fill detection ──
    # Find rapid-fire clusters (3+ onsets with small inter-onset gaps).
    # Default 150ms catches 32nd-note fills.  16th-note fills at moderate
    # tempo (~80 BPM = 182ms gaps) need a higher threshold but raising
    # globally creates false-positive clusters from kick-snare grooves
    # (kick→snare alternation at 80 BPM = 363ms; 16th hihat = 90ms).
    # Sweep via STRUM_FILL_MAX_GAP_MS to investigate per-song.
    fill_max_gap_ms = float(os.environ.get("STRUM_FILL_MAX_GAP_MS", "150"))
    # Reject clusters longer than this — guards against a runaway cluster
    # caused by a wide gap threshold consuming a whole verse groove.
    fill_max_dur_ms = float(os.environ.get("STRUM_FILL_MAX_DUR_MS", "3000"))
    fill_mask = np.zeros(n, dtype=bool)
    n_fill_reclassified = 0
    n_fills_found = 0

    if n >= 3:
        gaps = np.diff(times)
        # Build clusters of rapid hits
        cluster_start = 0
        clusters = []
        for i in range(1, n):
            if gaps[i - 1] > fill_max_gap_ms:  # break
                if i - cluster_start >= 3:
                    clusters.append((cluster_start, i))
                cluster_start = i
        if n - cluster_start >= 3:
            clusters.append((cluster_start, n))
        # Reject overly long clusters (false-positive groove detection)
        clusters = [
            (s, e) for s, e in clusters
            if (times[e - 1] - times[s]) <= fill_max_dur_ms
        ]

        for start, end in clusters:
            indices = list(range(start, end))

            # Check 1: hihat/ride must be LOW (absolute) — during fills,
            # drummer lifts hand from hihat. Use absolute threshold, not relative.
            cluster_hihat = np.mean(corrected[indices, 2])  # HiHat
            cluster_ride = np.mean(corrected[indices, 4])   # Ride
            hihat_absent = cluster_hihat < 0.30 and cluster_ride < 0.30

            # Check 2: spectral evidence — at least 30% of cluster is tonal
            cluster_ratios = low_ratio[indices]
            pct_tonal = np.mean(cluster_ratios > tom_ratio_threshold * 0.6)

            # Check 3: is a crash within 400ms after cluster end?
            crash_follows = False
            end_time = times[end - 1]
            crash_idx = np.where((times > end_time) & (times < end_time + 400))[0]
            if len(crash_idx) > 0:
                crash_follows = np.any(corrected[crash_idx, 6] > 0.3)  # Crash col

            # Score the fill candidate — hihat absence is mandatory
            cluster_overlaps_dbg = (
                _dbg_active
                and times[end - 1] >= _dbg_lo
                and times[start] <= _dbg_hi
            )
            if not hihat_absent:
                if cluster_overlaps_dbg:
                    logger.info(
                        f"  [fill-dbg] cluster t={times[start]:.0f}-{times[end-1]:.0f}ms "
                        f"n={end-start} REJECTED-hihat_present "
                        f"hihat={cluster_hihat:.2f} ride={cluster_ride:.2f}"
                    )
                continue  # Not a fill if hihat/ride is still playing

            fill_score = 0.4  # hihat absence confirmed
            if pct_tonal > 0.3:
                fill_score += 0.3
            if crash_follows:
                fill_score += 0.3

            # Diagnostic trace: dump every cluster overlapping the debug window,
            # whether or not it ends up being treated as a fill.
            if cluster_overlaps_dbg:
                logger.info(
                    f"  [fill-dbg] cluster t={times[start]:.0f}-{times[end-1]:.0f}ms "
                    f"n={end-start} hihat={cluster_hihat:.2f} ride={cluster_ride:.2f} "
                    f"hihat_absent={hihat_absent} pct_tonal={pct_tonal:.2f} "
                    f"crash_follows={crash_follows} fill_score={fill_score:.2f} "
                    f"thr={tom_ratio_threshold:.2f}"
                )

            if fill_score >= 0.7:
                n_fills_found += 1
                # Reclassify: for non-kick/snare onsets in the fill, transfer cymbal→tom.
                # Per-cluster floor; per-onset transfer scales with spectral confidence.
                fill_transfer = min(0.6, transfer_pct + fill_score * 0.3)
                # Confidence cap (max transfer when spectral evidence is strong).
                # Capped at 0.95 — never zero out a cymbal column entirely, in case
                # a real crash sits inside a fill.
                max_transfer = float(os.environ.get("STRUM_FILL_MAX_TRANSFER", "0.95"))
                # Per-onset transfer interpolates from `fill_transfer` (low_ratio at
                # threshold) up to `max_transfer` (low_ratio >= 4× threshold).
                # This is the key fix for high-confidence V14 cymbal mistakes:
                # when low_ratio is decisively tom-like (e.g. 20+), transfer ~all
                # of the wrong cymbal probability mass to the tom columns.
                conf_scale = float(os.environ.get("STRUM_FILL_CONF_SCALE", "4.0"))
                for i in indices:
                    fill_mask[i] = True
                    # Skip kick-dominated onsets even in fills
                    if corrected[i, 0] > 0.5:
                        if cluster_overlaps_dbg:
                            logger.info(
                                f"    [fill-dbg]   t={times[i]:.0f}ms SKIP-kick "
                                f"kick_p={corrected[i,0]:.2f} low_ratio={low_ratio[i]:.2f} "
                                f"crash_p={corrected[i,6]:.2f} ride_p={corrected[i,4]:.2f} "
                                f"hh_p={corrected[i,2]:.2f}"
                            )
                        continue
                    # Require per-onset spectral evidence
                    if low_ratio[i] < tom_ratio_threshold * 0.5:
                        if cluster_overlaps_dbg:
                            logger.info(
                                f"    [fill-dbg]   t={times[i]:.0f}ms SKIP-low_ratio "
                                f"low_ratio={low_ratio[i]:.2f}<{tom_ratio_threshold*0.5:.2f} "
                                f"crash_p={corrected[i,6]:.2f} ride_p={corrected[i,4]:.2f} "
                                f"hh_p={corrected[i,2]:.2f} kick_p={corrected[i,0]:.2f}"
                            )
                        continue
                    # Confidence-scaled transfer: stronger spectral evidence →
                    # more aggressive cymbal→tom move.  At threshold: fill_transfer.
                    # At conf_scale× threshold: max_transfer.  Linear in between.
                    conf = min(
                        1.0,
                        max(0.0,
                            (low_ratio[i] - tom_ratio_threshold)
                            / max(1e-6, tom_ratio_threshold * (conf_scale - 1.0))
                        ),
                    )
                    per_onset_transfer = fill_transfer + conf * (max_transfer - fill_transfer)
                    for cym_idx, tom_idx in pairs:
                        if corrected[i, cym_idx] < 0.10:
                            continue
                        boost = corrected[i, cym_idx] * per_onset_transfer
                        corrected[i, tom_idx] = min(1.0, corrected[i, tom_idx] + boost)
                        corrected[i, cym_idx] *= (1.0 - per_onset_transfer)
                        n_fill_reclassified += 1
                    if cluster_overlaps_dbg:
                        logger.info(
                            f"    [fill-dbg]   t={times[i]:.0f}ms RESCUED "
                            f"low_ratio={low_ratio[i]:.2f} transfer={per_onset_transfer:.2f} "
                            f"final crash_p={corrected[i,6]:.2f} ride_p={corrected[i,4]:.2f} "
                            f"hh_p={corrected[i,2]:.2f} hi_tom={corrected[i,3]:.2f} "
                            f"lo_tom={corrected[i,5]:.2f} fl_tom={corrected[i,7]:.2f}"
                        )

    # ── Stage 2: Disabled ──
    # Isolated spectral reclassification creates too many scattered false-positive
    # toms. Only fills-based reclassification (Stage 1) is used.
    n_spectral_reclassified = 0

    tom_like = int(np.sum(low_ratio > tom_ratio_threshold))
    logger.info(f"  Fill detection: {n_fills_found} fills found, {n_fill_reclassified} transfers")
    logger.info(f"  (Stage 2 disabled, {tom_like} tom-like onsets not reclassified)")

    # After-the-fact dump of every onset in the debug window with its outcome.
    if _dbg_active:
        in_win = np.where((times >= _dbg_lo) & (times <= _dbg_hi))[0]
        logger.info(f"  [fill-dbg] {len(in_win)} onsets in window {_dbg_lo:.0f}-{_dbg_hi:.0f}ms:")
        for i in in_win:
            tag = "in-fill" if fill_mask[i] else "OUTSIDE-fill"
            logger.info(
                f"    [fill-dbg]   t={times[i]:.0f}ms {tag} "
                f"low_ratio={low_ratio[i]:.2f} kick_p={corrected[i,0]:.2f} "
                f"snare_p={corrected[i,1]:.2f} hh_p={corrected[i,2]:.2f} "
                f"ride_p={corrected[i,4]:.2f} crash_p={corrected[i,6]:.2f} "
                f"hi_tom={corrected[i,3]:.2f} lo_tom={corrected[i,5]:.2f} fl_tom={corrected[i,7]:.2f}"
            )

    return corrected


# ══════════════════════════════════════════════════════════════
# Chart building
# ══════════════════════════════════════════════════════════════

def build_chart(
    onset_times_ms: list[float],
    class_probs: np.ndarray,
    valid_mask: np.ndarray,
    tempo_events: list[TempoEvent],
    thresholds: list[float] | None = None,
    spectral_centroids: np.ndarray | None = None,
    spectral_high_pcts: np.ndarray | None = None,
    onset_rms: np.ndarray | None = None,
    probs_pass1: np.ndarray | None = None,
    v14_class_probs: np.ndarray | None = None,
) -> DrumChart:
    """Convert classification results into a DrumChart.

    For each onset, emit hits for classes above threshold.
    Resolves tom/cymbal lane conflicts using V14 BiLSTM class predictions
    when available (99%+ accuracy on conflict pairs due to full-song context).
    Falls back to spectral heuristics when V14 probs are not provided.
    Caps hand notes at 2 per onset (drummer has 2 hands).
    """
    if thresholds is None:
        thresholds = DEFAULT_CLASS_THRESHOLDS

    hits = []
    class_counts = {c: 0 for c in range(8)}
    lane_conflicts_resolved = 0
    hand_caps_applied = 0
    spectral_overrides = 0
    kick_suppressed_ftom = 0
    kick_suppress_ens_protected = 0
    spectral_conflict_bias = 0
    v14_arbiter_used = 0
    crash_rescued = 0
    crash_rescue_b = 0
    crash_vetoed = 0
    crash_cofired = 0
    crash_post_suppress = 0
    ride_cofired = 0
    crash_spectral_cofired = 0

    # Hand classes = everything except Kick (idx 0). Kick is played with foot.
    HAND_CLASSES = {1, 2, 3, 4, 5, 6, 7}
    has_spectral = (spectral_centroids is not None and spectral_high_pcts is not None)
    has_v14 = v14_class_probs is not None

    # Pre-compute which onsets have ensemble-fired toms on shared lanes.
    # Used for fill-context detection: during fills, toms fire rapidly
    # across multiple shared lanes (HighTom → LowTom → FloorTom).
    # Only positions where the ensemble is above-threshold are counted,
    # which is far more selective than raw V14 probs.
    _CONFLICT_LANE_MAP = {2: (2, 3), 3: (4, 5), 4: (6, 7)}
    _ens_tom_times_by_lane: dict[int, list[float]] = {ln: [] for ln in [2, 3, 4]}
    for ix, tms in enumerate(onset_times_ms):
        if not valid_mask[ix]:
            continue
        for ln, (ci, ti) in _CONFLICT_LANE_MAP.items():
            if class_probs[ix, ti] > thresholds[ti]:
                _ens_tom_times_by_lane[ln].append(tms)
    for ln in [2, 3, 4]:
        _ens_tom_times_by_lane[ln].sort()

    # Pre-compute ensemble-fired crash positions for section detection.
    # Used by crash rescue: only rescue in sections where the ensemble
    # independently fires crash at nearby onsets (ens_crash > threshold).
    # This prevents rescue from adding crashes in breakdown sections
    # where V14's crash bias is the only signal.
    CRASH_IDX_PRECOMP = 6
    _ens_crash_times: list[float] = []
    for ix, tms in enumerate(onset_times_ms):
        if not valid_mask[ix]:
            continue
        if class_probs[ix, CRASH_IDX_PRECOMP] > thresholds[CRASH_IDX_PRECOMP]:
            _ens_crash_times.append(tms)
    _ens_crash_times.sort()

    def _in_crash_section(time_ms: float, window_ms: float = 5000.0,
                          min_nearby: int = 1) -> bool:
        """True if ≥min_nearby ensemble-fired crashes exist within ±window."""
        lo = bisect.bisect_left(_ens_crash_times, time_ms - window_ms)
        hi = bisect.bisect_right(_ens_crash_times, time_ms + window_ms)
        return (hi - lo) >= min_nearby

    # Pre-compute ride section detection: ensemble-fired ride positions.
    # Used by ride co-fire to only add rides within established ride sections
    # (≥4 ride hits in ±3s).  Ride plays on nearly every beat during verse/
    # chorus sections — if we detect a ride section, missing rides at
    # kick/snare positions are very likely genuine co-occurrences.
    RIDE_IDX_PRECOMP = 4
    _ens_ride_times: list[float] = []
    for ix, tms in enumerate(onset_times_ms):
        if not valid_mask[ix]:
            continue
        if class_probs[ix, RIDE_IDX_PRECOMP] > thresholds[RIDE_IDX_PRECOMP]:
            _ens_ride_times.append(tms)
    _ens_ride_times.sort()

    def _in_ride_section(time_ms: float, window_ms: float = 2000.0,
                         min_nearby: int = 6) -> bool:
        """True if ≥min_nearby ensemble-fired rides exist within ±window."""
        lo = bisect.bisect_left(_ens_ride_times, time_ms - window_ms)
        hi = bisect.bisect_right(_ens_ride_times, time_ms + window_ms)
        return (hi - lo) >= min_nearby

    # Pre-compute FloorTom section detection: ensemble-fired FloorTom positions.
    # Used by kick-suppresses-FloorTom to protect FloorTom when Song uses
    # toms heavily (e.g., tribal, prog-rock sections with kick+FloorTom).
    FTOM_IDX_PRECOMP = 7
    _ens_ftom_times: list[float] = []
    for ix, tms in enumerate(onset_times_ms):
        if not valid_mask[ix]:
            continue
        if class_probs[ix, FTOM_IDX_PRECOMP] > thresholds[FTOM_IDX_PRECOMP]:
            _ens_ftom_times.append(tms)
    _ens_ftom_times.sort()

    def _in_ftom_section(time_ms: float, window_ms: float = 3000.0,
                         min_nearby: int = 4) -> bool:
        """True if ≥min_nearby ensemble-fired FloorToms exist within ±window."""
        lo = bisect.bisect_left(_ens_ftom_times, time_ms - window_ms)
        hi = bisect.bisect_right(_ens_ftom_times, time_ms + window_ms)
        return (hi - lo) >= min_nearby

    def _ens_fill_context(time_ms: float, lane: int, window_ms: float = 300.0) -> bool:
        """True if ensemble fires tom on ≥1 other shared lane within ±window."""
        for other_lane in [2, 3, 4]:
            if other_lane == lane:
                continue
            times = _ens_tom_times_by_lane[other_lane]
            lo = bisect.bisect_left(times, time_ms - window_ms)
            hi = bisect.bisect_right(times, time_ms + window_ms)
            if hi > lo:
                return True
        return False

    v14_fill_protected = 0
    v14_fill_rescued = 0

    # RMS energy gate: compute adaptive threshold from the onset RMS
    # distribution.  Onsets with RMS far below the typical level are
    # Demucs bleed (guitar/vocal leaking into the drums stem) rather
    # than real drum hits.  Use 15% of the 75th percentile — this
    # cleanly separates bleed (RMS < 0.03) from real drums (RMS > 0.04).
    rms_gated = 0
    if onset_rms is not None:
        valid_rms = onset_rms[valid_mask.astype(bool)]
        positive_rms = valid_rms[valid_rms > 0]
        if len(positive_rms) > 0:
            rms_p75 = np.percentile(positive_rms, 75)
            rms_threshold = rms_p75 * 0.15
        else:
            rms_threshold = 0.0
    else:
        rms_threshold = 0.0

    for i, t_ms in enumerate(onset_times_ms):
        if not valid_mask[i]:
            continue

        # RMS energy gate: skip onsets in low-energy regions (Demucs bleed)
        if onset_rms is not None and onset_rms[i] < rms_threshold:
            rms_gated += 1
            continue

        # Collect which classes fire (above threshold)
        fired = set()
        for cls in range(8):
            if class_probs[i, cls] > thresholds[cls]:
                fired.add(cls)

        # ── V14-gated cymbal rescue ────────────────────────────────
        # Diagnostic (2026-04-24): on Demucs audio, 81.6% of pipeline
        # misses are events V14 already detected — concentrated on
        # cymbals (HiHat 1208, Ride 1190, Crash 414).  Cause: Demucs
        # degrades cymbal energy ~15%, so ensemble probs land just
        # under threshold even though V14's BiLSTM (full-song context)
        # is confident.  Rescue rule: when V14 strongly classifies a
        # cymbal AND ensemble at least weakly agrees AND no competing
        # tom on the same lane, fire the cymbal.  Tight V14 gates keep
        # FPs from exploding.
        if has_v14 and not os.environ.get("STRUM_NO_V14_CYM_RESCUE"):
            # (cym_idx, tom_idx, v14_thresh, ens_min)
            for cym_idx, tom_idx, v14_th, ens_min in (
                (2, 3, 0.50, 0.10),  # HiHat vs HighTom
                (4, 5, 0.50, 0.08),  # Ride  vs LowTom
                (6, 7, 0.55, 0.08),  # Crash vs FloorTom
            ):
                if cym_idx in fired or tom_idx in fired:
                    continue
                if v14_class_probs[i, cym_idx] <= v14_th:
                    continue
                # V14 must prefer cymbal over tom on this lane
                if v14_class_probs[i, cym_idx] <= v14_class_probs[i, tom_idx]:
                    continue
                if class_probs[i, cym_idx] < ens_min:
                    continue
                fired.add(cym_idx)

        # ── Crash co-fire threshold ────────────────────────────────
        # When Kick or Snare fires but Crash doesn't, V17's crash logit
        # often lands in 0.25-0.55 (below the 0.55 main threshold but
        # clearly above noise).  Use a lower co-fire threshold to recover
        # these.  The downstream V14 crash veto (v14<0.25 & ens<0.75)
        # acts as safety net against false positives.
        CRASH_COFIRE_THRESH = 0.30
        if 6 not in fired and (0 in fired or 1 in fired) and not os.environ.get("STRUM_NO_CRASH_BOOST"):
            if class_probs[i, 6] > CRASH_COFIRE_THRESH:
                fired.add(6)
                crash_cofired += 1
                if crash_cofired <= 5:
                    logger.debug(
                        f"  Crash co-fire @{t_ms:.0f}ms: ens_crash={class_probs[i, 6]:.3f} "
                        f"kick={class_probs[i, 0]:.3f} snare={class_probs[i, 1]:.3f}"
                    )

        # ── Ride co-fire (ride-section gated) ───────────────────────
        # When Kick or Snare fires within an established ride section
        # (≥6 rides in ±2s), add ride ONLY when both ensemble and V14
        # show ride signal.  V2 was too aggressive (ens>0.05 OR v14):
        # 3,300 FP from 716 TP.  Now require conjunctive gate.
        # Error analysis: 57% of missed Ride → Kick, 31% → Snare.
        RIDE_COFIRE_ENS_THRESH = 0.10
        RIDE_COFIRE_V14_THRESH = 0.30
        if (4 not in fired and (0 in fired or 1 in fired)
                and not os.environ.get("STRUM_NO_COOCCURRENCE_RESCUE")
                and _in_ride_section(t_ms)):
            ens_ride = class_probs[i, 4]
            v14_ride_ok = (
                has_v14
                and v14_class_probs[i, 4] > RIDE_COFIRE_V14_THRESH
                and v14_class_probs[i, 4] > v14_class_probs[i, 5]  # Ride > LowTom
            )
            if ens_ride > RIDE_COFIRE_ENS_THRESH and v14_ride_ok:
                fired.add(4)
                ride_cofired += 1
                if ride_cofired <= 5:
                    v14r = v14_class_probs[i, 4] if has_v14 else -1
                    logger.debug(
                        f"  Ride co-fire @{t_ms:.0f}ms: ens_ride={ens_ride:.3f} "
                        f"v14_ride={v14r:.3f}"
                    )

        # ── V14-gated crash co-fire boost ──────────────────────────────
        # When the standard crash co-fire (0.30) didn't trigger but V14
        # (with full-song context) confidently says Crash AND we're in
        # a crash section: lower threshold to 0.10.
        # Error analysis: 59% of missed Crash → Kick, 17% → Snare.
        # V14 + crash-section double-gate prevents false positives.
        CRASH_V14_COFIRE_THRESH = 0.10
        if (6 not in fired and (0 in fired or 1 in fired)
                and not os.environ.get("STRUM_NO_COOCCURRENCE_RESCUE")
                and not os.environ.get("STRUM_NO_CRASH_BOOST")
                and _in_crash_section(t_ms, window_ms=3000.0, min_nearby=1)):
            v14_crash_ok = (
                has_v14
                and v14_class_probs[i, 6] > 0.60
                and v14_class_probs[i, 6] > v14_class_probs[i, 7]  # Crash > FloorTom
            )
            if v14_crash_ok and class_probs[i, 6] > CRASH_V14_COFIRE_THRESH:
                fired.add(6)
                crash_spectral_cofired += 1
                if crash_spectral_cofired <= 5:
                    logger.debug(
                        f"  Crash V14 co-fire @{t_ms:.0f}ms: ens_crash={class_probs[i, 6]:.3f} "
                        f"v14_crash={v14_class_probs[i, 6]:.3f}"
                    )

        # Resolve lane conflicts on shared tom/cymbal lanes.
        # PRIMARY: V14 BiLSTM class predictions (full-song context, 99%+
        # tom/cym accuracy).  V14 sees the entire song through its BiLSTM
        # and self-attention, learning that crashes come with kicks in
        # choruses and toms descend in fills — contextual patterns that
        # isolated 500ms windows cannot capture.
        # Fill-context aware: when V14-predicted toms exist on other shared
        # lanes within ±300ms (indicating a fill/roll), override ratios are
        # adjusted — tom→cymbal harder, cymbal→tom easier.
        # FALLBACK: pass-1 ensemble probs when V14 unavailable.
        _CONFLICT_LANE = {(2, 3): 2, (4, 5): 3, (6, 7): 4}
        for cym_idx, tom_idx in LANE_CONFLICT_PAIRS:
            lane_num = _CONFLICT_LANE[(cym_idx, tom_idx)]
            fill_ctx = _ens_fill_context(t_ms, lane_num)
            if cym_idx in fired and tom_idx in fired:
                # Both fire — use V14 to pick winner (no fill bias here
                # to avoid regressing the many correct cymbal decisions).
                lane_conflicts_resolved += 1
                if has_v14:
                    v14_cym = v14_class_probs[i, cym_idx]
                    v14_tom = v14_class_probs[i, tom_idx]
                    if v14_cym >= v14_tom:
                        fired.discard(tom_idx)
                    else:
                        fired.discard(cym_idx)
                    v14_arbiter_used += 1
                else:
                    # Fallback: ensemble pass-1 probs
                    conflict_probs = probs_pass1 if probs_pass1 is not None else class_probs
                    cym_p = conflict_probs[i, cym_idx]
                    tom_p = conflict_probs[i, tom_idx]
                    if cym_p >= tom_p:
                        fired.discard(tom_idx)
                    else:
                        fired.discard(cym_idx)
            elif tom_idx in fired and cym_idx not in fired:
                # Tom fires alone — V14 can override to cymbal if it's
                # confident the hit is a cymbal (which the ensemble's low
                # tom threshold may have triggered incorrectly).
                if has_v14:
                    v14_cym = v14_class_probs[i, cym_idx]
                    v14_tom = v14_class_probs[i, tom_idx]
                    # Fill context: require much stronger V14 confidence
                    # to override tom→cymbal during fills (protects real
                    # tom fills from being flipped to cymbals).
                    override_ratio = 3.0 if fill_ctx else 1.5

                    # Ensemble-informed ratio lowering: when the ensemble
                    # assigns higher probability to the cymbal class than
                    # the tom class (even though cymbal didn't clear its
                    # higher absolute threshold), both ensemble AND V14
                    # agree — lower the bar to a direct comparison.
                    # Fixes HiHat→HighTom and Ride→LowTom misclassification
                    # caused by the asymmetric thresholds (0.44 vs 0.12).
                    # ONLY for Yellow/Blue lanes — Green lane (Crash/FloorTom)
                    # has naturally higher cymbal probs and needs the fill
                    # detection + spectral disambiguation instead.
                    ens_cym = class_probs[i, cym_idx]
                    ens_tom = class_probs[i, tom_idx]
                    if not fill_ctx and ens_cym > ens_tom and lane_num in (2, 3):
                        override_ratio = 1.0

                    # Green lane guard: V14 has strong Crash bias (median
                    # crash prob ~0.91) so FloorTom→Crash overrides fire
                    # at nearly every onset.  Require the ensemble (V17)
                    # to give at least moderate crash signal before allowing
                    # the flip.  Without this, 100+ false crashes appear
                    # from FloorTom onsets (threshold 0.12) being converted.
                    green_blocked = (
                        lane_num == 4
                        and ens_cym < thresholds[cym_idx] * 0.3
                    )

                    if not green_blocked and v14_cym > v14_tom * override_ratio:
                        fired.discard(tom_idx)
                        fired.add(cym_idx)
                        lane_conflicts_resolved += 1
                        v14_arbiter_used += 1
                    elif fill_ctx and v14_cym > v14_tom * 1.5:
                        v14_fill_protected += 1
                else:
                    # Fallback: ensemble-based swap.
                    # Use relative comparison: if ensemble prefers cymbal
                    # over tom, swap.  The old absolute-threshold gate
                    # (cym_p >= cym_thresh * 0.8) blocked swaps where
                    # cymbal was clearly more likely but below its high
                    # threshold (e.g. HiHat=0.30 vs HighTom=0.15).
                    conflict_probs = probs_pass1 if probs_pass1 is not None else class_probs
                    cym_p = conflict_probs[i, cym_idx]
                    tom_p = conflict_probs[i, tom_idx]
                    if cym_p > tom_p * 1.2:
                        fired.discard(tom_idx)
                        fired.add(cym_idx)
                        lane_conflicts_resolved += 1
            elif cym_idx in fired and tom_idx not in fired:
                # Cymbal fires alone — V14 can override to tom if it's
                # confident the hit is a tom (e.g., during fills).
                if has_v14:
                    v14_cym = v14_class_probs[i, cym_idx]
                    v14_tom = v14_class_probs[i, tom_idx]
                    # Fill context: lower threshold to let V14 rescue
                    # toms that the ensemble misclassified as cymbals.
                    override_ratio = 1.15 if fill_ctx else 1.5
                    if v14_tom > v14_cym * override_ratio:
                        fired.discard(cym_idx)
                        fired.add(tom_idx)
                        lane_conflicts_resolved += 1
                        v14_arbiter_used += 1
                        if fill_ctx:
                            v14_fill_rescued += 1

        # Kick-suppresses-FloorTom: when Kick fires confidently and
        # FloorTom barely cleared its low threshold, suppress FloorTom.
        # Kick and FloorTom share low-frequency energy; real simultaneous
        # kick+floor tom hits are rare.
        # Protection: when V14 says FloorTom > Crash, keep it.
        # NOTE: V14's crash bias (median 0.91) means this protection
        # rarely fires — effectively always suppresses FloorTom with Kick.
        KICK_IDX, FTOM_IDX = 0, 7
        CRASH_IDX_KS = 6
        ftom_was_suppressed = False
        if KICK_IDX in fired and FTOM_IDX in fired:
            if has_v14:
                v14_ftom_ks = v14_class_probs[i, FTOM_IDX]
                v14_crash_ks = v14_class_probs[i, CRASH_IDX_KS]
                if v14_ftom_ks > v14_crash_ks:
                    pass  # V14 says FloorTom — keep it
                else:
                    fired.discard(FTOM_IDX)
                    kick_suppressed_ftom += 1
                    ftom_was_suppressed = True
            else:
                fired.discard(FTOM_IDX)
                kick_suppressed_ftom += 1
                ftom_was_suppressed = True

        # ── V14-driven crash injection after FloorTom suppression ──
        # When kick-suppress removed FloorTom, the green lane is empty.
        # If V14 confidently says Crash (>0.80) AND V14 FloorTom is near
        # zero (<0.10), this was a real crash that V17 missed.  Inject it
        # directly — bypasses the restrictive _in_crash_section + ens_crash
        # gates that block normal crash rescue (V17 gives near-zero for
        # many GT crashes masked by kick energy).
        # Diagnostic: at GT crash positions, V14 FloorTom mean=0.004;
        # at real FloorTom positions, V14 FloorTom is much higher.
        if ftom_was_suppressed and has_v14 and CRASH_IDX_KS not in fired and not os.environ.get("STRUM_NO_CRASH_BOOST"):
            v14_crash_ps = v14_class_probs[i, CRASH_IDX_KS]
            v14_ftom_ps = v14_class_probs[i, FTOM_IDX]
            if v14_crash_ps > 0.80 and v14_ftom_ps < 0.10:
                fired.add(CRASH_IDX_KS)
                crash_post_suppress += 1
                if crash_post_suppress <= 5:
                    logger.debug(
                        f"  Crash post-suppress @{t_ms:.0f}ms: v14_crash={v14_crash_ps:.3f} "
                        f"v14_ftom={v14_ftom_ps:.3f} ens_crash={class_probs[i, CRASH_IDX_KS]:.3f}"
                    )

        # Cap hand notes at 2 per onset (drummer has 2 hands)
        hand_fired = fired & HAND_CLASSES
        if len(hand_fired) > 2:
            hand_caps_applied += 1
            # Keep the 2 hand classes with highest probability
            ranked = sorted(hand_fired, key=lambda c: class_probs[i, c], reverse=True)
            for cls in ranked[2:]:
                fired.discard(cls)

        # ── Tom-singleton constraint ──────────────────────────────
        # Drummer physics: a single onset can be at most ONE tom hit
        # (the stick that struck a tom can't strike a second tom in
        # the same instant). When the multi-label classifier fires
        # 2+ tom heads (HighTom=3, LowTom=5, FloorTom=7) for one
        # onset — common in fast tom rolls where adjacent toms have
        # similar spectral signatures — keep only the highest-prob
        # tom. Cymbal+tom and snare+tom remain valid (different hands).
        TOM_CLASSES = {3, 5, 7}
        tom_fired = fired & TOM_CLASSES
        if len(tom_fired) > 1:
            best_tom = max(tom_fired, key=lambda c: class_probs[i, c])
            for cls in tom_fired:
                if cls != best_tom:
                    fired.discard(cls)

        # ── V14-based crash veto ────────────────────────────────────
        # When the ensemble fires crash but V14's BiLSTM (with full-song
        # context) has low crash confidence, suppress the crash.
        # Targets the ~56 extra crashes where the ensemble fires at
        # kick/snare positions that aren't crashes in the GT.
        # Only vetos borderline ensemble decisions (ens < 0.65).
        CRASH_IDX, FTOM_IDX_CONST = 6, 7
        KICK_IDX = 0
        if CRASH_IDX in fired and has_v14:
            v14_crash_v = v14_class_probs[i, CRASH_IDX]
            ens_crash_v = class_probs[i, CRASH_IDX]
            # Veto when V14 doesn't agree.  Raised ens threshold from
            # 0.65→0.75 to catch more false crashes at kick positions
            # where V17 gives moderate activation but V14's full-song
            # context says no crash.
            if v14_crash_v < 0.25 and ens_crash_v < 0.75:
                fired.discard(CRASH_IDX)
                crash_vetoed += 1
                if crash_vetoed <= 5:
                    logger.debug(
                        f"  Crash veto @{t_ms:.0f}ms: v14_crash={v14_crash_v:.3f} "
                        f"ens_crash={ens_crash_v:.3f}"
                    )

        # ── V14-based crash rescue ─────────────────────────────────
        # When crash is below ensemble threshold but V14's BiLSTM
        # confidently classifies it as crash AND kick co-occurs,
        # rescue the crash.  This fixes the #1 error mode: kick+crash
        # co-occurrence where the ensemble under-predicts crash in
        # isolated 500ms windows.
        #
        # With V17 SOLO for crash, rescue must require moderate V17
        # signal (ens_crash >= 0.15) — if V17 gives near-zero crash
        # probability, V14's crash bias shouldn't override it.
        # Also gated by crash-section detection: ≥2 ensemble crashes
        # within ±5s prevents rescue in breakdown/quiet sections.
        V14_CRASH_RESCUE_THRESH = 0.90
        green_lane_empty = (CRASH_IDX not in fired and FTOM_IDX_CONST not in fired)
        if green_lane_empty and has_v14 and _in_crash_section(t_ms, window_ms=3000.0, min_nearby=2):
            v14_crash = v14_class_probs[i, CRASH_IDX]
            v14_ftom = v14_class_probs[i, FTOM_IDX_CONST]
            kick_prob = class_probs[i, KICK_IDX]
            ens_crash_r = class_probs[i, CRASH_IDX]
            if (v14_crash >= V14_CRASH_RESCUE_THRESH
                    and v14_crash > v14_ftom
                    and kick_prob >= 0.5
                    and ens_crash_r >= 0.05):
                # Path A: strict V14 + kick co-occurrence
                fired.add(CRASH_IDX)
                crash_rescued += 1
                if crash_rescued <= 5:
                    logger.debug(
                        f"  Crash rescue @{t_ms:.0f}ms: v14_crash={v14_crash:.3f} "
                        f"ens_crash={class_probs[i, CRASH_IDX]:.3f} kick={kick_prob:.3f}"
                    )
            elif (v14_crash >= 0.80
                    and v14_crash > v14_ftom * 1.5
                    and class_probs[i, CRASH_IDX] >= 0.35):
                # Path B (tightened): V14 must be highly confident, no kick
                # requirement.  Targets crash-alone positions — 45% of
                # GT crashes appear without kicks but the ensemble's
                # isolated 500ms windows under-predict crash there.
                fired.add(CRASH_IDX)
                crash_rescue_b += 1
                if crash_rescue_b <= 5:
                    logger.debug(
                        f"  Crash rescue-B @{t_ms:.0f}ms: v14_crash={v14_crash:.3f} "
                        f"ens_crash={class_probs[i, CRASH_IDX]:.3f}"
                    )

        for cls in fired:
            lane, is_cymbal = CLASS_TO_LANE[cls]
            hits.append(DrumHit(
                time_ms=t_ms,
                tick=0,
                lane=lane,
                is_cymbal=is_cymbal,
                velocity=100,
            ))
            class_counts[cls] += 1

    # Sort by time
    hits.sort(key=lambda h: h.time_ms)

    # ── Descending tom fill rescue ──────────────────────────────
    # Canonical drum fill pattern: HighTom(Y) → LowTom(B) → FloorTom(G)
    # at rapid spacing (< 150ms).  When the pipeline correctly detects
    # the first tom but misclassifies the next onset as a cymbal, the
    # fill sounds wrong.  Rescue pattern:
    #   HighTom on lane 2 →  Ride on lane 3 within 150ms  → flip to LowTom
    #   LowTom on lane 3 → Crash on lane 4 within 150ms  → flip to FloorTom
    # V14 confirmation: only rescue if V14 doesn't strongly prefer cymbal
    # at the target position (prevents false rescues on legitimate cymbals).
    FILL_RESCUE_PAIRS = [
        (2, 3),   # Yellow tom → Blue cymbal → flip to Blue tom
        (3, 4),   # Blue tom → Green cymbal → flip to Green tom
    ]
    FILL_RESCUE_MAX_DT = 100  # ms — tight window for fill patterns

    # Build time → V14 probs lookup for fill rescue confirmation
    _v14_by_time: dict[float, np.ndarray] = {}
    if has_v14:
        for ix, tms in enumerate(onset_times_ms):
            if valid_mask[ix]:
                _v14_by_time[tms] = v14_class_probs[ix]
    fill_rescue_count = 0
    for h_idx in range(len(hits) - 1):
        h1 = hits[h_idx]
        for src_lane, dst_lane in FILL_RESCUE_PAIRS:
            if h1.lane != src_lane or h1.is_cymbal:
                continue  # h1 must be a tom on src_lane
            # Look ahead for cymbal on dst_lane within FILL_RESCUE_MAX_DT
            for h2_idx in range(h_idx + 1, len(hits)):
                h2 = hits[h2_idx]
                dt = h2.time_ms - h1.time_ms
                if dt > FILL_RESCUE_MAX_DT:
                    break
                if dt < 10:
                    continue
                if h2.lane == dst_lane and h2.is_cymbal:
                    # V14 confirmation: skip rescue if V14 strongly
                    # prefers cymbal at this position.
                    cym_cls, tom_cls = {2: (2, 3), 3: (4, 5), 4: (6, 7)}[dst_lane]
                    if has_v14:
                        v14_probs = _v14_by_time.get(h2.time_ms)
                        if v14_probs is not None:
                            v14_c = v14_probs[cym_cls]
                            v14_t = v14_probs[tom_cls]
                            if v14_c > v14_t * 2.0:
                                break  # V14 clearly says cymbal, skip
                    # Flip cymbal → tom
                    h2.is_cymbal = False
                    class_counts[cym_cls] -= 1
                    class_counts[tom_cls] += 1
                    fill_rescue_count += 1
                    logger.debug(
                        f"  Fill rescue: t={h2.time_ms:.0f}ms lane={dst_lane} "
                        f"(triggered by tom at {h1.time_ms:.0f}ms lane={src_lane}, dt={dt:.0f}ms)"
                    )
                    break

    # Streak smoothing on shared tom/cymbal lanes: prevent flip-flop.
    # Find short streaks (≤2 consecutive same-type hits) flanked by
    # longer runs of the opposite type, and convert the short streak to
    # match its surroundings.  Bidirectional: fixes both stray toms in
    # cymbal sections AND stray cymbals in tom sections.
    # Runs iteratively until no more changes (cascading short streaks).
    #
    # Spectral protection: tom→cymbal flips are blocked when the spectral
    # centroid is < 3000 Hz (audio is tom-like). This preserves real tom
    # hits in fills/rolls that briefly visit a lane dominated by cymbals.
    # Class count adjustments: lane → (cymbal_class, tom_class)
    LANE_CLASS_MAP = {2: (2, 3), 3: (4, 5), 4: (6, 7)}  # yellow, blue, green

    # Build time → spectral centroid lookup for streak protection
    centroid_by_time: dict[float, float] = {}
    if has_spectral:
        for i, t_ms in enumerate(onset_times_ms):
            centroid_by_time[t_ms] = spectral_centroids[i]

    # Build time→tom lookup for fill detection: if a hit is near other
    # tom hits on DIFFERENT shared lanes, it's likely a fill/roll and
    # should be protected from streak smoothing.
    SHARED_LANES = {2, 3, 4}
    tom_times_by_lane: dict[int, list[float]] = {ln: [] for ln in SHARED_LANES}
    for h in hits:
        if h.lane in SHARED_LANES and not h.is_cymbal:
            tom_times_by_lane[h.lane].append(h.time_ms)
    for ln in SHARED_LANES:
        tom_times_by_lane[ln].sort()

    # Build sorted list of ALL hit times for density-based fill detection.
    all_hit_times = sorted(h.time_ms for h in hits)

    def _in_fill_context(time_ms: float, lane: int, window_ms: float = 200.0) -> bool:
        """True if there are tom hits on OTHER shared lanes within ±window_ms."""
        other_tom_lanes = 0
        for other_lane in SHARED_LANES:
            if other_lane == lane:
                continue
            times = tom_times_by_lane[other_lane]
            # Binary search for nearby toms
            lo = bisect.bisect_left(times, time_ms - window_ms)
            hi = bisect.bisect_right(times, time_ms + window_ms)
            if hi > lo:
                other_tom_lanes += 1
        return other_tom_lanes >= 1

    def _in_high_density_region(time_ms: float, window_ms: float = 500.0,
                                density_ratio: float = 1.5) -> bool:
        """True if local note density is significantly above the song average.

        Fills/rolls have higher note density than normal verse/chorus playing.
        When density is elevated, streak smoothing should be more conservative
        about flipping toms to cymbals.
        """
        if not all_hit_times:
            return False
        # Local density: hits within ±window_ms
        lo = bisect.bisect_left(all_hit_times, time_ms - window_ms)
        hi = bisect.bisect_right(all_hit_times, time_ms + window_ms)
        local_count = hi - lo
        local_density = local_count / (2 * window_ms / 1000.0)  # hits/sec

        # Global average density
        duration_s = (all_hit_times[-1] - all_hit_times[0]) / 1000.0
        if duration_s < 1.0:
            return False
        global_density = len(all_hit_times) / duration_s

        return local_density > global_density * density_ratio

    green_smoothed = 0  # total across all lanes
    spectral_streak_protected = 0
    fill_context_protected = 0
    density_protected = 0
    for smooth_lane, (cym_cls, tom_cls) in LANE_CLASS_MAP.items():
        lane_indices = [idx for idx, h in enumerate(hits) if h.lane == smooth_lane]
        if len(lane_indices) < 3:
            continue

        for _pass in range(5):  # max 5 iterations
            changed_this_pass = 0
            # Build streaks: [(is_cymbal, start_j, end_j), ...]
            streaks = []
            s_start = 0
            for j in range(1, len(lane_indices)):
                if hits[lane_indices[j]].is_cymbal != hits[lane_indices[s_start]].is_cymbal:
                    streaks.append((hits[lane_indices[s_start]].is_cymbal, s_start, j - 1))
                    s_start = j
            streaks.append((hits[lane_indices[s_start]].is_cymbal, s_start, len(lane_indices) - 1))

            # Identify short streaks (≤2) flanked by opposite runs
            for si, (is_cym, j_start, j_end) in enumerate(streaks):
                streak_len = j_end - j_start + 1
                if streak_len > 2:
                    continue
                if si == 0 or si == len(streaks) - 1:
                    continue
                prev_is_cym, _, prev_end = streaks[si - 1]
                next_is_cym, next_start, _ = streaks[si + 1]
                prev_len = prev_end - streaks[si - 1][1] + 1
                next_len = streaks[si + 1][2] - next_start + 1
                if prev_is_cym == is_cym or next_is_cym == is_cym:
                    continue
                # At least one neighbor must be longer than our streak
                if prev_len <= streak_len and next_len <= streak_len:
                    continue

                target_cymbal = prev_is_cym

                # For tom→cymbal flips, require BOTH neighbors to be
                # meaningfully longer (≥3 each).  During fills, streak
                # lengths are naturally short — a single tom between
                # two 2-note cymbal runs shouldn't be flipped.
                if target_cymbal:
                    if prev_len < 3 or next_len < 3:
                        continue

                for j in range(j_start, j_end + 1):
                    idx = lane_indices[j]
                    if hits[idx].is_cymbal != target_cymbal:
                        # Spectral protection: block tom→cymbal flip when
                        # centroid indicates tom-like audio (< 3500 Hz).
                        if target_cymbal and not hits[idx].is_cymbal:
                            centroid = centroid_by_time.get(hits[idx].time_ms, 5000.0)
                            if centroid < 3500:
                                spectral_streak_protected += 1
                                continue
                        # Fill context protection: block tom→cymbal flip
                        # when nearby tom hits exist on other shared lanes
                        # (indicates a fill/roll pattern).
                        if target_cymbal and not hits[idx].is_cymbal:
                            if _in_fill_context(hits[idx].time_ms, smooth_lane):
                                fill_context_protected += 1
                                continue
                        # Density protection: block tom→cymbal flip in
                        # high-density regions (fills/rolls are busier).
                        if target_cymbal and not hits[idx].is_cymbal:
                            if _in_high_density_region(hits[idx].time_ms):
                                density_protected += 1
                                continue
                        hits[idx].is_cymbal = target_cymbal
                        if target_cymbal:
                            class_counts[cym_cls] += 1; class_counts[tom_cls] -= 1
                        else:
                            class_counts[cym_cls] -= 1; class_counts[tom_cls] += 1
                        changed_this_pass += 1
                        green_smoothed += 1

            if changed_this_pass == 0:
                break

    # Dedup: remove near-simultaneous hits on same lane (keep first).
    # Sort so cymbals come before toms at same time — when two nearby
    # onsets produce a cymbal and a tom on the same lane, the cymbal
    # (dominant class) survives.
    hits.sort(key=lambda h: (h.time_ms, h.lane, not h.is_cymbal))
    deduped = []
    for hit in hits:
        if deduped and abs(hit.time_ms - deduped[-1].time_ms) < 15 \
                and hit.lane == deduped[-1].lane:
            continue
        deduped.append(hit)
    hits = deduped

    logger.info(f"  {len(hits)} hits after thresholding:")
    for cls in range(8):
        logger.info(f"    {CLASS_NAMES[cls]}: {class_counts[cls]}")
    if rms_gated > 0:
        logger.info(f"  RMS energy gate: {rms_gated} onsets suppressed (Demucs bleed, threshold={rms_threshold:.4f})")
    if lane_conflicts_resolved > 0:
        logger.info(f"  Lane conflicts resolved (tom won): {lane_conflicts_resolved}")
    if hand_caps_applied > 0:
        logger.info(f"  Hand cap applied (3+ → 2 hands): {hand_caps_applied}")
    if spectral_overrides > 0:
        logger.info(f"  Spectral cymbal→tom overrides: {spectral_overrides}")
    if kick_suppressed_ftom > 0:
        logger.info(f"  Kick-suppressed FloorTom: {kick_suppressed_ftom}")
    if kick_suppress_ens_protected > 0:
        logger.info(f"  Kick-suppress ensemble-protected FloorTom: {kick_suppress_ens_protected}")
    if spectral_conflict_bias > 0:
        logger.info(f"  Spectral-biased lane conflicts: {spectral_conflict_bias}")
    if v14_arbiter_used > 0:
        logger.info(f"  V14 BiLSTM arbiter decisions: {v14_arbiter_used}")
    if v14_fill_protected > 0:
        logger.info(f"  V14 fill-context: {v14_fill_protected} tom→cymbal overrides blocked")
    if v14_fill_rescued > 0:
        logger.info(f"  V14 fill-context: {v14_fill_rescued} cymbal→tom rescues")
    if fill_rescue_count > 0:
        logger.info(f"  Descending fill rescue: {fill_rescue_count} cymbal→tom flips")
    if crash_rescued > 0:
        logger.info(f"  V14 crash rescue: {crash_rescued} crashes recovered (path A)")
    if crash_rescue_b > 0:
        logger.info(f"  V14 crash rescue: {crash_rescue_b} crashes recovered (path B, relaxed)")
    if crash_cofired > 0:
        logger.info(f"  Crash co-fire: {crash_cofired} crashes added (kick/snare co-occurrence, thresh={CRASH_COFIRE_THRESH})")
    if ride_cofired > 0:
        logger.info(f"  Ride co-fire: {ride_cofired} rides added (kick/snare co-occurrence, spectral-gated)")
    if crash_spectral_cofired > 0:
        logger.info(f"  Crash V14 co-fire: {crash_spectral_cofired} crashes added (V14+section gated, thresh={CRASH_V14_COFIRE_THRESH})")
    if crash_post_suppress > 0:
        logger.info(f"  Crash post-suppress: {crash_post_suppress} crashes injected (V14-driven, after FloorTom suppression)")
    if crash_vetoed > 0:
        logger.info(f"  V14 crash veto: {crash_vetoed} false crashes suppressed")
    if green_smoothed > 0:
        logger.info(f"  Streak smoothing: {green_smoothed} flips fixed (bidirectional)")
    if spectral_streak_protected > 0:
        logger.info(f"  Streak smoothing: {spectral_streak_protected} tom→cymbal flips blocked (spectral protection)")
    if fill_context_protected > 0:
        logger.info(f"  Streak smoothing: {fill_context_protected} tom→cymbal flips blocked (fill context)")
    if density_protected > 0:
        logger.info(f"  Streak smoothing: {density_protected} tom→cymbal flips blocked (high density)")

    # Compute tick positions for tempo change events.
    ticks_per_beat = 480
    for idx in range(1, len(tempo_events)):
        prev = tempo_events[idx - 1]
        curr = tempo_events[idx]
        delta_ms = curr.time_ms - prev.time_ms
        ms_per_beat = 60_000.0 / prev.tempo_bpm
        ms_per_tick = ms_per_beat / ticks_per_beat
        delta_ticks = round(delta_ms / ms_per_tick)
        tempo_events[idx] = TempoEvent(
            tick=prev.tick + delta_ticks,
            tempo_bpm=curr.tempo_bpm,
            time_ms=curr.time_ms,
        )

    return DrumChart(
        hits=hits,
        tempo_events=tempo_events,
        time_signatures=[TimeSignature(tick=0, numerator=4, denominator=4, time_ms=0.0)],
        ticks_per_beat=480,
    )


# ══════════════════════════════════════════════════════════════
# IO helpers
# ══════════════════════════════════════════════════════════════

def convert_to_ogg(input_path: Path, output_path: Path, trim_start_ms: float = 0.0) -> None:
    """Copy/convert audio to output as song.ogg.

    If ``trim_start_ms`` > 0, that many milliseconds are removed from the
    beginning so the audio's first detected beat aligns with sample 0
    (= MIDI tick 0 = bar 1 in Clone Hero).
    """
    if trim_start_ms < 1.0 and input_path.suffix.lower() == ".ogg":
        shutil.copy(str(input_path), str(output_path))
        return
    try:
        cmd = ["ffmpeg", "-y"]
        if trim_start_ms >= 1.0:
            cmd += ["-ss", f"{trim_start_ms / 1000.0:.3f}"]
        cmd += ["-i", str(input_path), "-vn", "-c:a", "libvorbis",
                "-q:a", "6", str(output_path)]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        shutil.copy(str(input_path), output_path.parent / f"song{input_path.suffix}")


def create_song_ini(output_dir: Path, artist: str, title: str, tempo_bpm: float, duration_ms: float) -> None:
    """Create song.ini metadata file."""
    ini_content = f"""[song]
name = {title}
artist = {artist}
charter = STRUM
diff_drums = 4
diff_drums_real = 4
diff_drums_real_ps = 4
pro_drums = True
preview_start_time = {int(min(30000, duration_ms * 0.25))}
song_length = {int(duration_ms)}
"""
    (output_dir / "song.ini").write_text(ini_content, encoding="utf-8")


def parse_filename(filepath: Path) -> tuple[str, str]:
    """Extract artist and title from 'Title - Artist' filename."""
    name = filepath.stem
    if " - " in name:
        parts = name.split(" - ", 1)
        # Files are named "Title - Artist"
        return parts[1].strip(), parts[0].strip()
    return "Unknown Artist", name


# ══════════════════════════════════════════════════════════════
# MC hybrid overlay — per-class onset injection from MC head
# ══════════════════════════════════════════════════════════════

def mc_hybrid_overlay(
    chart: DrumChart,
    mc_frame_probs: np.ndarray | None,
    tempo_events: list[TempoEvent],
    drums_path: Path | None = None,
    tom_refinement: dict | None = None,
    device: torch.device | None = None,
) -> DrumChart:
    """Inject MC-detected onsets for classes where MC head beats V14.

    For each class in MC_HYBRID_CLASSES, finds per-class peaks in the
    MC head's frame-level probabilities and adds chart hits for peaks
    that don't already have a matching hit.

    For tom classes (HighTom, LowTom, FloorTom), if a tom refinement CNN
    is provided, candidate peaks are verified/reclassified before injection.

    This does NOT replace existing V14+ensemble hits — only adds missing ones.
    The MC head's per-class output IS the classification (no ensemble needed).

    Args:
        chart: Chart from the V14+ensemble pipeline
        mc_frame_probs: (T, 8) per-frame per-class onset probabilities from MC head
        tempo_events: Tempo events for the song
        drums_path: Path to drums audio (needed for tom refinement)
        tom_refinement: Dict with 'model', 'mel_transform', 'cqt_transform' or None
        device: Torch device

    Returns:
        Modified chart with MC-injected hits
    """
    if mc_frame_probs is None or not MC_HYBRID_ENABLED:
        return chart

    min_dist_frames = max(1, int(20 / 1000 * V14_SR / V14_HOP))
    tolerance_ms = MC_HYBRID_TOLERANCE_MS
    total_added = 0
    class_added = {}

    # Precompute mel/CQT for tom refinement if we have the model
    mel_spec = None
    cqt_spec = None
    tom_model = None
    half_ctx = TOM_REFINEMENT_CONTEXT_FRAMES // 2
    tom_refined = 0
    tom_rejected = 0
    tom_reclassified = 0

    if tom_refinement is not None and drums_path is not None and device is not None:
        tom_model = tom_refinement["model"]
        try:
            y, sr = sf.read(str(drums_path), dtype='float32')
            if y.ndim > 1:
                y = y.mean(axis=1)
            if sr != V14_SR:
                audio_t = torch.from_numpy(y).float().unsqueeze(0)
                audio_t = torchaudio.functional.resample(audio_t, sr, V14_SR)
                y = audio_t.squeeze(0).numpy()
            audio_tensor = torch.from_numpy(y).float().unsqueeze(0)
            mel_spec = torch.log(tom_refinement["mel_transform"](audio_tensor) + 1e-8).squeeze(0)  # (n_mels, T)
            cqt_spec = tom_refinement["cqt_transform"](audio_tensor).squeeze(0)  # (n_cqt, T)
        except Exception as e:
            logger.warning(f"  Tom refinement audio load failed: {e}")
            tom_model = None

    # Index existing chart hits by class for fast lookup
    existing_by_class: dict[int, list[float]] = {cls: [] for cls in range(8)}
    for hit in chart.hits:
        lane = hit.lane
        is_cym = hit.is_cymbal
        # Reverse map: (lane, is_cymbal) → class index
        for cls_idx, (l, c) in CLASS_TO_LANE.items():
            if l == lane and c == is_cym:
                existing_by_class[cls_idx].append(hit.time_ms)
                break
    for cls in range(8):
        existing_by_class[cls].sort()

    new_hits = list(chart.hits)

    # Collect tom candidate peaks for batch refinement
    tom_candidate_peaks: list[tuple[int, int]] = []  # (peak_frame, mc_cls_idx)

    for cls_idx, threshold in MC_HYBRID_CLASSES.items():
        cls_probs = mc_frame_probs[:, cls_idx]
        mc_peaks, _ = find_peaks(cls_probs, height=threshold, distance=min_dist_frames)

        is_tom_class = cls_idx in TOM_CLASSES_MC

        existing_times = existing_by_class[cls_idx]
        cls_added = 0

        for peak_frame in mc_peaks:
            peak_ms = (peak_frame * V14_HOP / V14_SR) * 1000

            # Check if a chart hit already exists within tolerance
            if existing_times:
                insert_pos = bisect.bisect_left(existing_times, peak_ms)
                already_exists = False
                for check_pos in [insert_pos - 1, insert_pos]:
                    if 0 <= check_pos < len(existing_times):
                        if abs(existing_times[check_pos] - peak_ms) <= tolerance_ms:
                            already_exists = True
                            break
                if already_exists:
                    continue

            # For tom classes with refinement: defer to batch CNN classification
            if is_tom_class and tom_model is not None:
                tom_candidate_peaks.append((peak_frame, cls_idx))
                continue

            # Non-tom classes: inject directly (as before)
            lane, is_cymbal = CLASS_TO_LANE[cls_idx]
            new_hits.append(DrumHit(
                time_ms=peak_ms,
                tick=0,
                lane=lane,
                is_cymbal=is_cymbal,
                velocity=100,
            ))
            # Add to existing_times to prevent duplicate injection
            bisect.insort(existing_times, peak_ms)
            cls_added += 1

        if cls_added > 0:
            class_added[cls_idx] = cls_added
            total_added += cls_added

    # Batch tom refinement
    if tom_candidate_peaks and tom_model is not None and mel_spec is not None:
        T = mel_spec.shape[-1]
        mel_windows = []
        cqt_windows = []
        valid_indices = []

        for i, (peak_frame, mc_cls) in enumerate(tom_candidate_peaks):
            if peak_frame < half_ctx or peak_frame >= T - half_ctx:
                continue
            start = peak_frame - half_ctx
            end = peak_frame + half_ctx + 1
            m = mel_spec[:, start:end]
            c = cqt_spec[:, start:end]
            if m.shape[-1] < TOM_REFINEMENT_CONTEXT_FRAMES:
                pad = TOM_REFINEMENT_CONTEXT_FRAMES - m.shape[-1]
                m = torch.nn.functional.pad(m, (0, pad))
                c = torch.nn.functional.pad(c, (0, pad))
            mel_windows.append(m.unsqueeze(0))  # (1, n_mels, ctx)
            cqt_windows.append(c.unsqueeze(0))  # (1, n_cqt, ctx)
            valid_indices.append(i)

        if mel_windows:
            mel_batch = torch.stack(mel_windows).to(device)  # (N, 1, n_mels, ctx)
            cqt_batch = torch.stack(cqt_windows).to(device)  # (N, 1, n_cqt, ctx)

            with torch.no_grad():
                logits = tom_model(mel_batch, cqt_batch)  # (N, 4)
                preds = logits.argmax(dim=1).cpu().numpy()  # 0=NotTom, 1=High, 2=Low, 3=Floor

            for j, orig_idx in enumerate(valid_indices):
                peak_frame, mc_cls_idx = tom_candidate_peaks[orig_idx]
                pred_label = int(preds[j])

                if pred_label == 0:
                    # CNN says NotTom — reject this candidate
                    tom_rejected += 1
                    continue

                # CNN confirmed a tom — use CNN's class prediction
                refined_cls_idx = TOM_LABEL_TO_MC[pred_label]
                if refined_cls_idx != mc_cls_idx:
                    tom_reclassified += 1
                tom_refined += 1

                peak_ms = (peak_frame * V14_HOP / V14_SR) * 1000
                lane, is_cymbal = CLASS_TO_LANE[refined_cls_idx]

                # Check for existing hit at the refined class too
                existing_times_refined = existing_by_class[refined_cls_idx]
                already_exists = False
                if existing_times_refined:
                    insert_pos = bisect.bisect_left(existing_times_refined, peak_ms)
                    for check_pos in [insert_pos - 1, insert_pos]:
                        if 0 <= check_pos < len(existing_times_refined):
                            if abs(existing_times_refined[check_pos] - peak_ms) <= tolerance_ms:
                                already_exists = True
                                break
                if already_exists:
                    continue

                new_hits.append(DrumHit(
                    time_ms=peak_ms,
                    tick=0,
                    lane=lane,
                    is_cymbal=is_cymbal,
                    velocity=100,
                ))
                bisect.insort(existing_times_refined, peak_ms)

                cls_key = refined_cls_idx
                class_added[cls_key] = class_added.get(cls_key, 0) + 1
                total_added += 1

    if total_added > 0:
        detail = ", ".join(f"{CLASS_NAMES[c]}=+{n}" for c, n in sorted(class_added.items()))
        logger.info(f"  MC hybrid overlay: +{total_added} hits ({detail})")
    if tom_refined + tom_rejected > 0:
        logger.info(f"  Tom refinement: {tom_refined} confirmed, {tom_rejected} rejected, {tom_reclassified} reclassified")

    # Sort by time
    new_hits.sort(key=lambda h: h.time_ms)

    return DrumChart(
        hits=new_hits,
        tempo_events=chart.tempo_events,
        time_signatures=chart.time_signatures,
        ticks_per_beat=chart.ticks_per_beat,
    )


def apply_tom_refinement_filter(
    chart: DrumChart,
    drums_path: Path,
    tom_refinement: dict,
    device: torch.device,
) -> DrumChart:
    """Verify all tom hits in the chart using the TomRefinementCNN.

    For every hit classified as HighTom, LowTom, or FloorTom, extract
    a mel+CQT window and run through the CNN.  If the CNN says NotTom,
    flip the hit to its cymbal counterpart on the same lane.
    If the CNN says a different tom type, reclassify.

    Args:
        chart: Complete chart (after MC overlay)
        drums_path: Path to drums audio stem
        tom_refinement: Dict with 'model', 'mel_transform', 'cqt_transform'
        device: Torch device

    Returns:
        Chart with verified/reclassified tom hits
    """
    # Load audio and compute spectrograms
    try:
        y, sr = sf.read(str(drums_path), dtype='float32')
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != V14_SR:
            audio_t = torch.from_numpy(y).float().unsqueeze(0)
            audio_t = torchaudio.functional.resample(audio_t, sr, V14_SR)
            y = audio_t.squeeze(0).numpy()
        audio_tensor = torch.from_numpy(y).float().unsqueeze(0)
        mel_spec = torch.log(tom_refinement["mel_transform"](audio_tensor) + 1e-8).squeeze(0)
        cqt_spec = tom_refinement["cqt_transform"](audio_tensor).squeeze(0)
    except Exception as e:
        logger.warning(f"  Tom refinement filter: audio load failed: {e}")
        return chart

    T = mel_spec.shape[-1]
    half_ctx = TOM_REFINEMENT_CONTEXT_FRAMES // 2
    tom_model = tom_refinement["model"]

    # Tom class indices in 8-class system
    TOM_CLS_SET = {3, 5, 7}  # HighTom, LowTom, FloorTom
    # Cymbal counterpart for each tom (same lane, cymbal=True)
    TOM_TO_CYMBAL = {3: 2, 5: 4, 7: 6}  # HighTom→HiHat, LowTom→Ride, FloorTom→Crash

    # Find tom hits and their indices
    tom_hit_indices = []
    tom_cls_indices = []
    for i, hit in enumerate(chart.hits):
        if hit.is_cymbal:
            continue
        # Check if this is a tom hit (lane 2/3/4, not cymbal)
        for cls_idx in TOM_CLS_SET:
            lane, is_cym = CLASS_TO_LANE[cls_idx]
            if hit.lane == lane and not hit.is_cymbal:
                tom_hit_indices.append(i)
                tom_cls_indices.append(cls_idx)
                break

    if not tom_hit_indices:
        return chart

    # Extract windows for all tom hits
    mel_windows = []
    cqt_windows = []
    valid_batch_indices = []  # maps batch position → tom_hit_indices position

    for j, hit_idx in enumerate(tom_hit_indices):
        hit = chart.hits[hit_idx]
        frame_idx = int(hit.time_ms / 1000 * V14_SR / V14_HOP)
        if frame_idx < half_ctx or frame_idx >= T - half_ctx:
            continue
        start = frame_idx - half_ctx
        end = frame_idx + half_ctx + 1
        m = mel_spec[:, start:end]
        c = cqt_spec[:, start:end]
        if m.shape[-1] < TOM_REFINEMENT_CONTEXT_FRAMES:
            pad = TOM_REFINEMENT_CONTEXT_FRAMES - m.shape[-1]
            m = torch.nn.functional.pad(m, (0, pad))
            c = torch.nn.functional.pad(c, (0, pad))
        mel_windows.append(m.unsqueeze(0))
        cqt_windows.append(c.unsqueeze(0))
        valid_batch_indices.append(j)

    if not mel_windows:
        return chart

    # Batch classify
    mel_batch = torch.stack(mel_windows).to(device)
    cqt_batch = torch.stack(cqt_windows).to(device)

    with torch.no_grad():
        logits = tom_model(mel_batch, cqt_batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()  # (N, 4)
        preds = logits.argmax(dim=1).cpu().numpy()

    # Apply CNN decisions — only flip if confident
    NOT_TOM_CONFIDENCE = float(os.environ.get("STRUM_NOT_TOM_CONF", "0.40"))
    n_kept = 0
    n_flipped_to_cymbal = 0
    n_reclassified = 0
    n_density_protected = 0
    hits_to_modify = {}  # hit_idx → (new_lane, new_is_cymbal)

    # Per-lane tom density: count tom hits in the chart per shared lane.
    # When a lane has many toms (tom-heavy song e.g. tribal/prog-rock),
    # require higher CNN confidence to flip tom→cymbal.
    tom_count_by_lane = {2: 0, 3: 0, 4: 0}
    for hit in chart.hits:
        if hit.lane in tom_count_by_lane and not hit.is_cymbal:
            tom_count_by_lane[hit.lane] += 1
    TOM_DENSE_THRESHOLD = int(os.environ.get("STRUM_TOM_DENSE_TH", "100"))  # ≥100 toms on a single lane = tom-heavy
    TOM_DENSE_NOT_TOM_CONF = 0.70  # Require 70% CNN confidence in tom-heavy lanes

    for batch_j, tom_list_j in enumerate(valid_batch_indices):
        hit_idx = tom_hit_indices[tom_list_j]
        orig_cls = tom_cls_indices[tom_list_j]
        pred_label = int(preds[batch_j])
        not_tom_prob = float(probs[batch_j, 0])

        # Use higher bar for tom-heavy lanes
        hit_lane = CLASS_TO_LANE[orig_cls][0]
        effective_conf = TOM_DENSE_NOT_TOM_CONF if tom_count_by_lane.get(hit_lane, 0) >= TOM_DENSE_THRESHOLD else NOT_TOM_CONFIDENCE

        if pred_label == 0 and not_tom_prob >= effective_conf:
            # CNN is confident it's NotTom → flip to cymbal counterpart
            cymbal_cls = TOM_TO_CYMBAL[orig_cls]
            lane, is_cymbal = CLASS_TO_LANE[cymbal_cls]
            hits_to_modify[hit_idx] = (lane, is_cymbal)
            n_flipped_to_cymbal += 1
        elif pred_label == 0 and not_tom_prob >= NOT_TOM_CONFIDENCE:
            # CNN thinks NotTom but tom-heavy lane protection → keep tom
            n_density_protected += 1
            n_kept += 1
        else:
            # Keep original classification (don't reclassify between tom types
            # — the ensemble/V14 has better lane-specific calibration)
            n_kept += 1

    # Apply modifications
    if hits_to_modify:
        new_hits = []
        for i, hit in enumerate(chart.hits):
            if i in hits_to_modify:
                new_lane, new_is_cymbal = hits_to_modify[i]
                new_hits.append(DrumHit(
                    time_ms=hit.time_ms,
                    tick=hit.tick,
                    lane=new_lane,
                    is_cymbal=new_is_cymbal,
                    velocity=hit.velocity,
                ))
            else:
                new_hits.append(hit)
        chart = DrumChart(
            hits=new_hits,
            tempo_events=chart.tempo_events,
            time_signatures=chart.time_signatures,
            ticks_per_beat=chart.ticks_per_beat,
        )

    total_checked = n_kept + n_flipped_to_cymbal
    if total_checked > 0:
        density_msg = f", {n_density_protected} density-protected" if n_density_protected > 0 else ""
        logger.info(
            f"  Tom refinement filter: {total_checked} tom hits checked, "
            f"{n_kept} kept, {n_flipped_to_cymbal} flipped→cymbal{density_msg}"
        )

    return chart


def apply_cymbal_to_tom_rescue(
    chart: DrumChart,
    drums_path: Path,
    tom_refinement: dict,
    device: torch.device,
) -> DrumChart:
    """Rescue cymbals on shared lanes that are actually toms in a fill.

    Symmetric counterpart to ``apply_tom_refinement_filter``: for every
    CYMBAL hit on a shared lane (Yellow/Blue/Green) that sits inside a
    fill context (i.e. within ±200 ms of a tom hit on a *different*
    shared lane), run the TomRefinementCNN on the audio. If the CNN
    confidently says the audio is a tom, flip cymbal→tom on the same
    lane. This catches the common pattern where a tom roll across
    Y→B→G has one or two intermediate hits the ensemble flipped to
    cymbals.

    Args:
        chart: Chart after ``apply_tom_refinement_filter``.
        drums_path: Path to drums audio stem.
        tom_refinement: Loaded TomRefinementCNN bundle.
        device: Torch device.

    Returns:
        Chart with rescued tom hits.
    """
    if not chart.hits:
        return chart

    # Tunables (env-overridable)
    FILL_WINDOW_MS = float(os.environ.get("STRUM_TOM_RESCUE_WINDOW_MS", "200"))
    TOM_CONF = float(os.environ.get("STRUM_TOM_RESCUE_TOM_CONF", "0.55"))
    NOT_TOM_MAX = float(os.environ.get("STRUM_TOM_RESCUE_NOT_TOM_MAX", "0.30"))

    SHARED_LANES = {2, 3, 4}

    # Sort by time once, then build per-lane time arrays for fast neighbour lookups.
    hits = list(chart.hits)
    hits_sorted_idx = sorted(range(len(hits)), key=lambda i: hits[i].time_ms)

    tom_times_by_lane: dict[int, list[float]] = {2: [], 3: [], 4: []}
    for h in hits:
        if h.lane in SHARED_LANES and not h.is_cymbal:
            tom_times_by_lane[h.lane].append(h.time_ms)
    for lane in tom_times_by_lane:
        tom_times_by_lane[lane].sort()

    def has_other_lane_tom(time_ms: float, this_lane: int) -> bool:
        for lane, times in tom_times_by_lane.items():
            if lane == this_lane or not times:
                continue
            # bisect window
            lo, hi = 0, len(times)
            while lo < hi:
                mid = (lo + hi) // 2
                if times[mid] < time_ms - FILL_WINDOW_MS:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < len(times) and times[lo] <= time_ms + FILL_WINDOW_MS:
                return True
        return False

    # Collect candidate hits: cymbals on shared lanes with fill context
    candidate_indices: list[int] = []
    for i, h in enumerate(hits):
        if h.lane in SHARED_LANES and h.is_cymbal and has_other_lane_tom(h.time_ms, h.lane):
            candidate_indices.append(i)

    if not candidate_indices:
        return chart

    # Load audio + spectrograms (mirror apply_tom_refinement_filter)
    try:
        y, sr = sf.read(str(drums_path), dtype='float32')
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != V14_SR:
            audio_t = torch.from_numpy(y).float().unsqueeze(0)
            audio_t = torchaudio.functional.resample(audio_t, sr, V14_SR)
            y = audio_t.squeeze(0).numpy()
        audio_tensor = torch.from_numpy(y).float().unsqueeze(0)
        mel_spec = torch.log(tom_refinement["mel_transform"](audio_tensor) + 1e-8).squeeze(0)
        cqt_spec = tom_refinement["cqt_transform"](audio_tensor).squeeze(0)
    except Exception as e:
        logger.warning(f"  Cymbal→tom rescue: audio load failed: {e}")
        return chart

    T = mel_spec.shape[-1]
    half_ctx = TOM_REFINEMENT_CONTEXT_FRAMES // 2

    mel_windows, cqt_windows, batch_to_cand = [], [], []
    for j, hit_idx in enumerate(candidate_indices):
        h = hits[hit_idx]
        frame_idx = int(h.time_ms / 1000 * V14_SR / V14_HOP)
        if frame_idx < half_ctx or frame_idx >= T - half_ctx:
            continue
        s, e = frame_idx - half_ctx, frame_idx + half_ctx + 1
        m = mel_spec[:, s:e]
        c = cqt_spec[:, s:e]
        if m.shape[-1] < TOM_REFINEMENT_CONTEXT_FRAMES:
            pad = TOM_REFINEMENT_CONTEXT_FRAMES - m.shape[-1]
            m = torch.nn.functional.pad(m, (0, pad))
            c = torch.nn.functional.pad(c, (0, pad))
        mel_windows.append(m.unsqueeze(0))
        cqt_windows.append(c.unsqueeze(0))
        batch_to_cand.append(j)

    if not mel_windows:
        return chart

    mel_batch = torch.stack(mel_windows).to(device)
    cqt_batch = torch.stack(cqt_windows).to(device)
    tom_model = tom_refinement["model"]
    with torch.no_grad():
        logits = tom_model(mel_batch, cqt_batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()  # (N, 4)

    n_flipped = 0
    new_hits = list(hits)
    for batch_j, cand_j in enumerate(batch_to_cand):
        hit_idx = candidate_indices[cand_j]
        not_tom_p = float(probs[batch_j, 0])
        tom_p = float(max(probs[batch_j, 1:]))  # best-of-tom class probability
        if tom_p >= TOM_CONF and not_tom_p <= NOT_TOM_MAX:
            h = new_hits[hit_idx]
            new_hits[hit_idx] = DrumHit(
                time_ms=h.time_ms,
                tick=h.tick,
                lane=h.lane,
                is_cymbal=False,
                velocity=h.velocity,
            )
            n_flipped += 1

    if n_flipped > 0:
        logger.info(
            f"  Cymbal→tom rescue: {n_flipped}/{len(candidate_indices)} "
            f"shared-lane cymbals in fill context flipped to toms"
        )

    return DrumChart(
        hits=new_hits,
        tempo_events=chart.tempo_events,
        time_signatures=chart.time_signatures,
        ticks_per_beat=chart.ticks_per_beat,
    )


# ══════════════════════════════════════════════════════════════
# Phase 3 reclassification
# ══════════════════════════════════════════════════════════════

# Reverse mapping: (lane, is_cymbal) → 8-class index
LANE_CYM_TO_CLASS = {v: k for k, v in CLASS_TO_LANE.items()}


def phase3_reclassify(
    chart: DrumChart,
    mc_frame_probs: np.ndarray,
) -> DrumChart:
    """Reclassify existing chart hits using Phase 3 frame-level class probs.

    For each hit, looks up Phase 3's class probabilities at that time and
    applies reclassification rules when the Phase 3 model confidently
    suggests a different class than the ensemble assigned.

    Args:
        chart: The classified DrumChart after build_chart + postprocessing.
        mc_frame_probs: (T, 8) Phase 3 per-frame per-class onset probs.

    Returns:
        Modified DrumChart with reclassified hits.
    """
    if not PHASE3_RECLASS_RULES:
        return chart

    total_frames = mc_frame_probs.shape[0]
    reclass_counts = Counter()
    new_hits = []

    for hit in chart.hits:
        # Determine current 8-class index
        from_cls = LANE_CYM_TO_CLASS.get((hit.lane, hit.is_cymbal), -1)
        if from_cls < 0:
            new_hits.append(hit)
            continue

        # Look up Phase 3 probs at this onset time
        frame = int(hit.time_ms * V14_SR / 1000.0 / V14_HOP)
        frame = min(max(frame, 0), total_frames - 1)
        p3 = mc_frame_probs[frame]

        # Try reclassification rules in priority order
        reclassed = False
        for (fc, tc), (min_prob, min_margin) in PHASE3_RECLASS_RULES.items():
            if fc != from_cls:
                continue
            if p3[tc] >= min_prob and (p3[tc] - p3[fc]) >= min_margin:
                new_lane, new_is_cymbal = CLASS_TO_LANE[tc]
                new_hits.append(DrumHit(
                    time_ms=hit.time_ms,
                    tick=hit.tick,
                    lane=new_lane,
                    is_cymbal=new_is_cymbal,
                    velocity=hit.velocity,
                ))
                reclass_counts[(fc, tc)] += 1
                reclassed = True
                break

        if not reclassed:
            new_hits.append(hit)

    if any(reclass_counts.values()):
        parts = []
        for (fc, tc), n in sorted(reclass_counts.items(), key=lambda x: -x[1]):
            parts.append(f"{CLASS_NAMES[fc]}→{CLASS_NAMES[tc]}:{n}")
        logger.info(f"  Phase 3 reclassified {sum(reclass_counts.values())} hits: {', '.join(parts)}")

    return DrumChart(
        hits=new_hits,
        tempo_events=chart.tempo_events,
        time_signatures=chart.time_signatures,
        ticks_per_beat=chart.ticks_per_beat,
    )


def phase3_onset_rescue(
    chart: DrumChart,
    mc_frame_probs: np.ndarray,
) -> DrumChart:
    """Add missing onsets detected by Phase 3 where NO chart hit exists on any lane.

    Unlike mc_hybrid_overlay which checks per-class, this only adds peaks
    at times where the entire chart has no nearby hit — recovering true onset misses.

    Args:
        chart: The DrumChart (post-reclassification).
        mc_frame_probs: (T, 8) Phase 3 per-frame per-class onset probs.

    Returns:
        Modified DrumChart with rescued onsets added.
    """
    if not PHASE3_RESCUE_ENABLED:
        return chart

    threshold = PHASE3_RESCUE_THRESHOLD
    tolerance_ms = PHASE3_RESCUE_TOLERANCE_MS
    min_dist_frames = max(1, int(PHASE3_RESCUE_MIN_DISTANCE_MS / 1000 * V14_SR / V14_HOP))

    # Build sorted list of ALL chart hit times (across all lanes)
    all_times = sorted(h.time_ms for h in chart.hits)

    # For each frame, compute max class prob across all 8 classes
    # then find peaks in this "any class" onset signal
    max_probs = mc_frame_probs.max(axis=1)  # (T,)

    from scipy.signal import find_peaks
    peaks, properties = find_peaks(max_probs, height=threshold, distance=min_dist_frames)

    rescued = Counter()
    new_hits = list(chart.hits)

    for peak_frame in peaks:
        peak_ms = (peak_frame * V14_HOP / V14_SR) * 1000

        # Check if ANY chart hit exists within tolerance
        if all_times:
            insert_pos = bisect.bisect_left(all_times, peak_ms)
            already_exists = False
            for check_pos in [insert_pos - 1, insert_pos]:
                if 0 <= check_pos < len(all_times):
                    if abs(all_times[check_pos] - peak_ms) <= tolerance_ms:
                        already_exists = True
                        break
            if already_exists:
                continue

        # No chart hit nearby — use Phase 3's best class for this frame
        p3 = mc_frame_probs[peak_frame]
        best_cls = int(np.argmax(p3))
        best_prob = float(p3[best_cls])
        if best_prob < threshold:
            continue

        lane, is_cymbal = CLASS_TO_LANE[best_cls]
        new_hits.append(DrumHit(
            time_ms=peak_ms,
            tick=0,
            lane=lane,
            is_cymbal=is_cymbal,
            velocity=100,
        ))
        # Track for dedup in subsequent iterations
        bisect.insort(all_times, peak_ms)
        rescued[best_cls] += 1

    total = sum(rescued.values())
    if total > 0:
        parts = [f"{CLASS_NAMES[c]}=+{n}" for c, n in sorted(rescued.items(), key=lambda x: -x[1])]
        logger.info(f"  Phase 3 onset rescue: +{total} hits ({', '.join(parts)})")

    new_hits.sort(key=lambda h: h.time_ms)
    return DrumChart(
        hits=new_hits,
        tempo_events=chart.tempo_events,
        time_signatures=chart.time_signatures,
        ticks_per_beat=chart.ticks_per_beat,
    )


# ── Phase 3 cymbal co-occurrence rescue ──
# Unlike onset rescue (no chart hit at all), this adds MISSING CYMBALS
# at positions where the chart already has Kick or Snare.
# Error analysis: 88% of missed Rides and 76% of missed Crashes are at
# Kick/Snare positions.  Phase 3 frame probs provide independent evidence
# from the V14 multi-class onset head (different from the 7-model ensemble).
PHASE3_COOCCURRENCE_ENABLED = os.environ.get("STRUM_P3_COOCCURRENCE", "1") == "1"
PHASE3_COOCCURRENCE_RIDE_THRESH = float(os.environ.get("STRUM_P3_RIDE_TH", "0.99"))
PHASE3_COOCCURRENCE_CRASH_THRESH = float(os.environ.get("STRUM_P3_CRASH_TH", "0.55"))


def phase3_cymbal_cooccurrence_rescue(
    chart: DrumChart,
    mc_frame_probs: np.ndarray,
) -> DrumChart:
    """Add missing Ride/Crash cymbals at Kick/Snare positions using Phase 3 probs.

    For each chart position with Kick (lane 0) or Snare (lane 1), checks
    Phase 3's per-class frame probabilities for Ride (cls 4) and Crash (cls 6).
    If Phase 3 confidently predicts Ride/Crash but the chart has no Blue/Green
    hit at that time, add the cymbal.

    This complements the ensemble-based build_chart by using independent evidence
    from V14's multiclass onset head to rescue co-occurrences that the
    ensemble missed due to its 500ms-window isolation.

    Safety gates:
    - Ride requires P3_ride > threshold AND P3_ride > P3_lowtom (not a tom)
    - Crash requires P3_crash > threshold AND P3_crash > P3_floortom
    - Must not already have a hit on the target lane within 50ms
    """
    if not PHASE3_COOCCURRENCE_ENABLED:
        return chart

    total_frames = mc_frame_probs.shape[0]
    tolerance_ms = 50.0

    # Index chart hits by lane and time for fast lookup
    hits_by_lane: dict[int, list[float]] = defaultdict(list)
    for hit in chart.hits:
        hits_by_lane[hit.lane].append(hit.time_ms)
    for lane in hits_by_lane:
        hits_by_lane[lane].sort()

    # Pre-compute tom times (lanes 2/3/4 with is_cymbal=False) so we can
    # detect tom rolls / fills and skip cymbal injection in those regions
    # — co-occurrence rescue inside a tom roll produces phantom crash/ride.
    tom_times = sorted(
        h.time_ms for h in chart.hits
        if h.lane in (2, 3, 4) and not h.is_cymbal
    )
    FILL_WINDOW_MS = 300.0
    FILL_TOM_MIN = 3  # ≥3 toms within ±300ms => treat as fill, skip rescue

    def _is_in_fill(time_ms: float) -> bool:
        if not tom_times:
            return False
        lo = bisect.bisect_left(tom_times, time_ms - FILL_WINDOW_MS)
        hi = bisect.bisect_right(tom_times, time_ms + FILL_WINDOW_MS)
        return (hi - lo) >= FILL_TOM_MIN

    def _has_hit_on_lane(lane: int, time_ms: float) -> bool:
        times = hits_by_lane.get(lane, [])
        if not times:
            return False
        pos = bisect.bisect_left(times, time_ms)
        for check in [pos - 1, pos]:
            if 0 <= check < len(times):
                if abs(times[check] - time_ms) <= tolerance_ms:
                    return True
        return False

    # Collect Kick/Snare positions
    kick_snare_hits = [h for h in chart.hits if h.lane in (0, 1)]

    ride_thresh = PHASE3_COOCCURRENCE_RIDE_THRESH
    crash_thresh = PHASE3_COOCCURRENCE_CRASH_THRESH
    new_hits = list(chart.hits)
    n_ride = 0
    n_crash = 0

    for hit in kick_snare_hits:
        # Skip rescue inside fill regions — phantom cymbals during tom rolls
        if _is_in_fill(hit.time_ms):
            continue
        frame = int(hit.time_ms * V14_SR / 1000.0 / V14_HOP)
        frame = min(max(frame, 0), total_frames - 1)
        p3 = mc_frame_probs[frame]

        # Ride co-occurrence rescue (Blue lane = lane 3)
        if not _has_hit_on_lane(3, hit.time_ms):
            p3_ride = float(p3[4])
            p3_lowtom = float(p3[5])
            if p3_ride > ride_thresh and p3_ride > p3_lowtom:
                new_hits.append(DrumHit(
                    time_ms=hit.time_ms,
                    tick=hit.tick,
                    lane=3,
                    is_cymbal=True,
                    velocity=90,
                ))
                bisect.insort(hits_by_lane[3], hit.time_ms)
                n_ride += 1

        # Crash co-occurrence rescue (Green lane = lane 4)
        if not _has_hit_on_lane(4, hit.time_ms):
            p3_crash = float(p3[6])
            p3_ftom = float(p3[7])
            if p3_crash > crash_thresh and p3_crash > p3_ftom:
                new_hits.append(DrumHit(
                    time_ms=hit.time_ms,
                    tick=hit.tick,
                    lane=4,
                    is_cymbal=True,
                    velocity=90,
                ))
                bisect.insort(hits_by_lane[4], hit.time_ms)
                n_crash += 1

    if n_ride + n_crash > 0:
        logger.info(f"  Phase 3 co-occurrence rescue: Ride=+{n_ride}, Crash=+{n_crash}")
        new_hits.sort(key=lambda h: h.time_ms)

    return DrumChart(
        hits=new_hits,
        tempo_events=chart.tempo_events,
        time_signatures=chart.time_signatures,
        ticks_per_beat=chart.ticks_per_beat,
    )


# ══════════════════════════════════════════════════════════════
# Spectral reclassification (post-chart audio feature rules)
# ══════════════════════════════════════════════════════════════

def _compute_hnr(window: np.ndarray, sr: int) -> float:
    """Compute Harmonic-to-Noise Ratio via autocorrelation in 50-500Hz range."""
    if len(window) < 2 or np.max(np.abs(window)) < 1e-10:
        return 0.0
    w = window - np.mean(window)
    norm = np.dot(w, w)
    if norm < 1e-12:
        return 0.0
    autocorr = np.correlate(w, w, mode="full")
    autocorr = autocorr[len(w) - 1:] / norm
    # Search in 50-500Hz lag range
    min_lag = max(1, int(sr / 500))
    max_lag = min(len(autocorr) - 1, int(sr / 50))
    if min_lag >= max_lag:
        return 0.0
    peak = np.max(autocorr[min_lag:max_lag + 1])
    peak = np.clip(peak, 0.0, 0.999)
    return 10.0 * np.log10(peak / (1.0 - peak + 1e-12))


def spectral_reclassify(chart: DrumChart, drums_path: Path) -> DrumChart:
    """Reclassify HiHat→Snare and Crash→Ride using audio features (HNR, stereo corr, RMS).

    Operates on the chart after Phase 3 rescue. Uses the drums stem audio to
    compute spectral features at each onset position and applies threshold rules.
    """
    if not SPECTRAL_RECLASS_ENABLED or not chart.hits:
        return chart

    sr = SPECTRAL_RECLASS_SR
    win_ms = SPECTRAL_RECLASS_WINDOW_MS
    win_samples = int(sr * win_ms / 1000.0)

    # Load drums stem (preserve stereo if available)
    try:
        audio_raw, orig_sr = sf.read(str(drums_path))
    except Exception as e:
        logger.warning(f"  Spectral reclass: cannot read {drums_path}: {e}")
        return chart

    # Handle mono vs stereo
    if audio_raw.ndim == 1:
        audio_raw = audio_raw[:, np.newaxis]  # (N, 1)
    is_stereo = audio_raw.shape[1] >= 2

    # Resample if needed
    if orig_sr != sr:
        resampled = []
        for ch in range(audio_raw.shape[1]):
            resampled.append(librosa.resample(audio_raw[:, ch], orig_sr=orig_sr, target_sr=sr))
        audio = np.column_stack(resampled)
    else:
        audio = audio_raw

    mono = np.mean(audio, axis=1)
    n_samples = len(mono)

    n_h2s = 0  # HiHat → Snare
    n_c2r = 0  # Crash → Ride
    new_hits = []

    # ── PASS 1: collect all flip CANDIDATES without applying ──
    # Independent per-hit thresholds cause cascading flips: when a section's
    # audio character shifts (e.g. heavier hits), every consecutive HiHat
    # passes the rule and the entire run becomes Snare. Pass 1 marks
    # candidates, Pass 2 suppresses runs of >=3 consecutive same-lane flips
    # (clear sign of section drift, not real per-hit reclassification).
    flip_h2s_idx: set[int] = set()
    flip_c2r_idx: set[int] = set()

    for idx, hit in enumerate(chart.hits):
        if hit.lane == 2 and hit.is_cymbal:
            start = int(hit.time_ms * sr / 1000.0)
            end = min(start + win_samples, n_samples)
            if end - start < 10:
                continue
            win_mono = mono[start:end]
            rms = float(np.sqrt(np.mean(win_mono ** 2)))
            hnr = _compute_hnr(win_mono, sr)
            sc = 1.0
            if is_stereo:
                win_l = audio[start:end, 0]
                win_r = audio[start:end, 1]
                if np.std(win_l) > 1e-10 and np.std(win_r) > 1e-10:
                    sc = float(np.corrcoef(win_l, win_r)[0, 1])
            # Tightened (2026-04-25 ear-test): require BOTH conditions, not OR.
            # OR was triggering on every loud hi-hat, cascading whole sections
            # to snare. AND-rule keeps only true mis-classifications.
            if rms > SPECTRAL_RECLASS_RMS_SNARE and (sc < SPECTRAL_RECLASS_SC_TH and hnr > SPECTRAL_RECLASS_HNR_SNARE):
                flip_h2s_idx.add(idx)
        elif hit.lane == 4 and hit.is_cymbal:
            start = int(hit.time_ms * sr / 1000.0)
            end = min(start + win_samples, n_samples)
            if end - start < 10:
                continue
            win_mono = mono[start:end]
            rms = float(np.sqrt(np.mean(win_mono ** 2)))
            hnr = _compute_hnr(win_mono, sr)
            if hnr > SPECTRAL_RECLASS_HNR_CRASH and rms > SPECTRAL_RECLASS_RMS_CRASH:
                flip_c2r_idx.add(idx)

    # Suppress drift: in any 6-hit window of HiHats, allow at most 1 flip.
    # Catches scattered flips (every 2-3 hihats) that consecutive-only logic missed.
    MAX_CONSEC_FLIPS = 0   # zero consecutive flips allowed
    lane2_indices = [i for i, h in enumerate(chart.hits) if h.lane == 2 and h.is_cymbal]
    # Window-density check: if any 6-hihat window has ≥2 flips, drop them all
    WINDOW = 6
    MAX_PER_WINDOW = 1
    if len(lane2_indices) >= WINDOW:
        to_drop = set()
        for j in range(len(lane2_indices) - WINDOW + 1):
            window_idxs = lane2_indices[j:j + WINDOW]
            flips_in_window = [k for k in window_idxs if k in flip_h2s_idx]
            if len(flips_in_window) > MAX_PER_WINDOW:
                to_drop.update(flips_in_window)
        flip_h2s_idx -= to_drop
    run_start = None
    run_len = 0
    for j, idx in enumerate(lane2_indices):
        if idx in flip_h2s_idx:
            if run_start is None:
                run_start = j
            run_len += 1
        else:
            if run_len > MAX_CONSEC_FLIPS:
                # Drop the whole cascading run — keep as HiHat
                for k in range(run_start, run_start + run_len):
                    flip_h2s_idx.discard(lane2_indices[k])
            run_start = None
            run_len = 0
    if run_len > MAX_CONSEC_FLIPS:
        for k in range(run_start, run_start + run_len):
            flip_h2s_idx.discard(lane2_indices[k])

    # ── PASS 2: apply surviving flips ──
    for idx, hit in enumerate(chart.hits):
        if idx in flip_h2s_idx:
            new_hits.append(DrumHit(
                time_ms=hit.time_ms, tick=hit.tick, lane=1,
                is_cymbal=False, velocity=hit.velocity,
            ))
            n_h2s += 1
            continue
        if idx in flip_c2r_idx:
            new_hits.append(DrumHit(
                time_ms=hit.time_ms, tick=hit.tick, lane=3,
                is_cymbal=True, velocity=hit.velocity,
            ))
            n_c2r += 1
            continue
        new_hits.append(hit)

    if n_h2s + n_c2r > 0:
        logger.info(f"  Spectral reclass: HiHat→Snare={n_h2s}, Crash→Ride={n_c2r}")
    else:
        logger.info(f"  Spectral reclass: no changes")

    return DrumChart(
        hits=new_hits,
        tempo_events=chart.tempo_events,
        time_signatures=chart.time_signatures,
        ticks_per_beat=chart.ticks_per_beat,
    )


# ══════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════

def process_song(
    v14_model: TwoStageDrumsCRNN,
    ensemble: list[dict],
    audio_path: Path,
    output_dir: Path,
    device: torch.device,
    skip_separation: bool = False,
    onset_threshold: float = 0.4,
    class_thresholds: list[float] | None = None,
    two_pass_context: bool = True,
    postprocess: bool = True,
    full_audio_dir: Path | None = None,
    tomcym: dict | None = None,
    tom_refinement: dict | None = None,
    phase3_model: TwoStageDrumsCRNN | None = None,
    v14_only: bool = False,
) -> Path:
    """Process a single song through the hybrid pipeline."""
    artist, title = parse_filename(audio_path)
    safe_name = re.sub(r'[<>:"/\\|?*]', "", f"{artist} - {title}")
    song_folder = output_dir / safe_name
    song_folder.mkdir(parents=True, exist_ok=True)

    logger.info(f"Song folder: {song_folder}")

    # 1. Analyze audio (tempo, beat offset, tempo changes)
    audio_info = analyze_audio(audio_path)

    # 2. Separate drums
    if skip_separation:
        drums_path = audio_path
        logger.info("  Skipping separation (using input as drums stem)")
    else:
        drums_path = separate_drums(audio_path, song_folder)

    # 3. Stage 1: V14 onset detection (also captures class probs + MC head probs)
    t0 = time.time()
    onset_times_ms, v14_class_probs, mc_frame_probs = detect_onsets_v14(
        v14_model, drums_path, device,
        onset_threshold=onset_threshold,
    )
    logger.info(f"  Stage 1 done: {len(onset_times_ms)} onsets ({time.time() - t0:.1f}s)")

    if not onset_times_ms:
        logger.warning("  No onsets detected! Creating empty chart.")
        chart = DrumChart(
            hits=[],
            tempo_events=audio_info["tempo_events"],
            time_signatures=[TimeSignature(tick=0, numerator=4, denominator=4, time_ms=0.0)],
            ticks_per_beat=480,
        )
    else:
        if v14_only:
            # V14-only mode: use Stage 2 classifier directly, skip ensemble
            logger.info("  V14-only: using Stage 2 classifier (no ensemble)")
            probs = v14_class_probs  # Already sigmoid from model output
            probs_pass1 = probs
            windows = {"valid_mask": np.ones(len(onset_times_ms), dtype=bool)}
            spectral_centroids = np.full(len(onset_times_ms), 3000.0)
            spectral_high_pcts = np.full(len(onset_times_ms), 0.3)
            onset_rms = np.ones(len(onset_times_ms), dtype=np.float32)
            n_hard_gated = 0
        else:
            # 4. Extract mel windows at each onset
            t0 = time.time()
            any_needs_cqt = any(e["needs_cqt"] for e in ensemble)
            windows = extract_onset_windows(drums_path, onset_times_ms, needs_cqt=any_needs_cqt)
            logger.info(f"  Window extraction done ({time.time() - t0:.1f}s)")

            # 5. Stage 2: Ensemble classification
            # Pass 1: classify with zero context
            t0 = time.time()
            context = build_context_vectors(onset_times_ms)
            logits_pass1 = classify_onsets_ensemble(ensemble, windows, context, device)
            probs_pass1 = 1.0 / (1.0 + np.exp(-logits_pass1))  # sigmoid for context
            logger.info(f"  Stage 2 pass 1 done ({time.time() - t0:.1f}s)")

            if two_pass_context:
                # Pass 2: re-classify with context from pass 1
                t0 = time.time()
                context = build_context_vectors(onset_times_ms, probs_pass1)
                logits = classify_onsets_ensemble(ensemble, windows, context, device)
                logger.info(f"  Stage 2 pass 2 done ({time.time() - t0:.1f}s)")
            else:
                logits = logits_pass1

        if not v14_only:
            # 5b. Spectral safety net: compute spectral features for cymbal→tom
            # override. When audio at an onset is clearly tom-like (low centroid,
            # no high-frequency shimmer) and kick isn't dominant, override cymbal
            # classification to tom on shared lanes.
            t0 = time.time()
            spectral_centroids, spectral_high_pcts = _compute_spectral_centroid_features(
                drums_path, onset_times_ms
            )
            logger.info(f"  Spectral features computed ({time.time() - t0:.1f}s)")

            # 5b2. Hard spectral gating DISABLED — v17 testing showed no benefit
            n_hard_gated = 0
            # Convert logits to probabilities
            probs = 1.0 / (1.0 + np.exp(-logits))

            # 5c. Compute per-onset RMS for energy gating (suppress Demucs bleed)
            onset_rms = _compute_onset_rms(drums_path, onset_times_ms)

            # 5c2. 6-stem drumsep arbiter (Tom-vs-Cymbal lane disambiguation).
            # Validated 2026-04-27: ≥98% Tom→Cymbal fix-rate, +40dB margins
            # across 3 songs. Default ON. Disable with STRUM_USE_DRUMSEP=0.
            if os.environ.get("STRUM_USE_DRUMSEP", "1") == "1":
                t0 = time.time()
                stems_dir = separate_drums_6stem(drums_path, song_folder)
                logger.info(f"  6-stem drumsep stage ({time.time()-t0:.1f}s)")
                if stems_dir is not None:
                    probs = apply_drumsep_lane_arbiter(
                        probs, onset_times_ms, stems_dir,
                        win_ms=float(os.environ.get("STRUM_DRUMSEP_WIN_MS", "50")),
                        margin_db=float(os.environ.get("STRUM_DRUMSEP_MARGIN_DB", "20")),
                        toms_floor_db=float(os.environ.get("STRUM_DRUMSEP_TOMS_FLOOR_DB", "-40")),
                    )

            # 5d. Fill-aware cymbal→tom rescue (re-wired 2026-04-26).
            # Detects rapid 3+ hit clusters with hihat/ride absent and transfers
            # cymbal probability mass to tom counterpart. Operates on `probs`
            # before build_chart's lane-conflict resolution sees them.
            # NOTE: We deliberately do NOT modify v14_class_probs here. v23
            # experiment showed corrupting V14 with the same fill rescue
            # increases Cym→Tom false positives (Alkaline 7→18) because the
            # rescue's cluster detector flags some legitimate cymbal sections.
            # V14 has stronger song-level context and should remain the
            # authoritative arbiter for lane conflicts.
            # Gated by STRUM_FILL_RESCUE (default on).
            if os.environ.get("STRUM_FILL_RESCUE", "1") == "1":
                t0 = time.time()
                spectral_lr = compute_spectral_features(drums_path, onset_times_ms)
                probs = apply_spectral_reclassification(
                    probs, spectral_lr, onset_times_ms,
                    tom_ratio_threshold=float(os.environ.get("STRUM_FILL_TOM_RATIO", "5.0")),
                    transfer_pct=float(os.environ.get("STRUM_FILL_TRANSFER", "0.4")),
                )
                logger.info(f"  Fill rescue done ({time.time() - t0:.1f}s)")

        # 6. Build chart
        chart = build_chart(
            onset_times_ms, probs, windows["valid_mask"],
            tempo_events=audio_info["tempo_events"],
            thresholds=class_thresholds,
            spectral_centroids=spectral_centroids,
            spectral_high_pcts=spectral_high_pcts,
            onset_rms=onset_rms,
            probs_pass1=probs_pass1,
            v14_class_probs=v14_class_probs,
        )

    _reset_trace()
    _trace_snapshot(chart, "after_build_chart", song_folder)

    # 6a. MC hybrid overlay: MOVED to after postprocessing (6d) to avoid
    # disrupting postprocess heuristics

    # 6b. Binary tom/cymbal reclassification (second pass)
    # DISABLED: CNN reclassification creates more isolated errors than it fixes.
    # The CNN (macro_f1=0.592) shows good test-set discrimination but suffers
    # from domain mismatch on pipeline inference. Needs fine-tuning on
    # human-verified pipeline outputs before enabling.
    # if chart.hits and tomcym is not None:
    #     t0 = time.time()
    #     chart = apply_tomcym_reclassification(
    #         chart, drums_path, tomcym, device,
    #     )
    #     logger.info(f"  Tom/cymbal reclassification done ({time.time() - t0:.1f}s)")

    # 6c. Post-processing
    # Save pre-postprocess snapshot for debugging
    pre_pp_path = song_folder / "notes_pre_postprocess.mid"
    export_all_difficulties(chart, pre_pp_path)
    logger.info(f"  Saved pre-postprocess snapshot ({len(chart.hits)} Expert hits)")

    if chart.hits and postprocess:
        t0 = time.time()
        chart = postprocess_chart(chart)
        logger.info(f"  Post-processing done ({time.time() - t0:.1f}s)")
    _trace_snapshot(chart, "after_postprocess", song_folder)

    # 6d. Phase 3 reclassification: fix misclassified hits using Phase 3 frame probs
    # Runs AFTER postprocessing. Replaces the overlay approach (which added false positives).
    phase3_probs = None
    if phase3_model is not None and (PHASE3_RECLASS_ENABLED or PHASE3_RESCUE_ENABLED):
        t0 = time.time()
        phase3_probs = run_phase3_inference(phase3_model, drums_path, device)
        logger.info(f"  Phase 3 inference done ({time.time() - t0:.1f}s)")

        if PHASE3_RECLASS_ENABLED:
            t0 = time.time()
            chart = phase3_reclassify(chart, phase3_probs)
            logger.info(f"  Phase 3 reclassification done ({time.time() - t0:.1f}s)")
            _trace_snapshot(chart, "after_phase3_reclass", song_folder)

    # 6d2. Phase 3 onset rescue: add missing onsets where no chart hit exists on any lane
    if phase3_probs is not None and PHASE3_RESCUE_ENABLED:
        t0 = time.time()
        chart = phase3_onset_rescue(chart, phase3_probs)
        logger.info(f"  Phase 3 onset rescue done ({time.time() - t0:.1f}s)")
        _trace_snapshot(chart, "after_phase3_rescue", song_folder)

    # 6d2b. Phase 3 cymbal co-occurrence rescue: add missing Ride/Crash at Kick/Snare positions
    if phase3_probs is not None and PHASE3_COOCCURRENCE_ENABLED:
        t0 = time.time()
        chart = phase3_cymbal_cooccurrence_rescue(chart, phase3_probs)
        logger.info(f"  Phase 3 co-occurrence rescue done ({time.time() - t0:.1f}s)")
        _trace_snapshot(chart, "after_phase3_cooccurrence", song_folder)

    # 6d3. Spectral reclassification: fix HiHat→Snare and Crash→Ride using audio features
    if SPECTRAL_RECLASS_ENABLED and chart.hits:
        t0 = time.time()
        chart = spectral_reclassify(chart, drums_path)
        logger.info(f"  Spectral reclassification done ({time.time() - t0:.1f}s)")
        _trace_snapshot(chart, "after_spectral_reclass", song_folder)

    # 6e. MC hybrid overlay: inject MC-detected onsets for winning classes
    # DISABLED by default — reclassification replaces this approach
    elif MC_HYBRID_ENABLED and not PHASE3_RECLASS_ENABLED:
        overlay_probs = mc_frame_probs
        if phase3_model is not None:
            overlay_probs = run_phase3_inference(phase3_model, drums_path, device)
        if overlay_probs is not None:
            t0 = time.time()
            chart = mc_hybrid_overlay(
                chart, overlay_probs, audio_info["tempo_events"],
                drums_path=drums_path, tom_refinement=tom_refinement, device=device,
            )
            logger.info(f"  MC hybrid overlay done ({time.time() - t0:.1f}s)")

    # 6f. Tom refinement filter: verify ALL tom hits in the chart
    # Flips false-positive toms to cymbal counterparts, reclassifies confused toms
    if tom_refinement is not None:
        t0 = time.time()
        chart = apply_tom_refinement_filter(chart, drums_path, tom_refinement, device)
        logger.info(f"  Tom refinement filter done ({time.time() - t0:.1f}s)")
        _trace_snapshot(chart, "after_tom_refine", song_folder)

        # 6f2. Symmetric pass: rescue cymbals on shared lanes that are
        # actually toms in a fill (Y/B/G roll where intermediate hits
        # got flipped to cymbals by the ensemble).
        t0 = time.time()
        chart = apply_cymbal_to_tom_rescue(chart, drums_path, tom_refinement, device)
        logger.info(f"  Cymbal→tom rescue done ({time.time() - t0:.1f}s)")
        _trace_snapshot(chart, "after_cym2tom_rescue", song_folder)

    # 6g. FINAL hand-cap re-pass: phase3_*_rescue, complete_cymbal_patterns and
    # spectral_reclassify can add hits AFTER postprocess_chart's cap, leaving
    # 3+ hand notes at the same tick. Cap to 2 hands per drummer.
    try:
        from scripts.chart_postprocess import resolve_playability, protect_tom_fills, cap_close_hands
        if chart.hits:
            chart = protect_tom_fills(chart)
            chart = resolve_playability(chart)
            chart = cap_close_hands(chart)
            _trace_snapshot(chart, "after_handcap", song_folder)
    except Exception as _e:
        logger.warning(f"  Final hand-cap pass failed: {_e}")

    # 6h. FINAL drumsep arbiter pass on emitted hits.
    # Phase 3 reclassification operates on its own model's probs and can revert
    # the prob-level arbiter applied in step 5c2.  This final pass re-asserts
    # the physical signal: a "cymbal" hit whose toms.wav RMS dominates the
    # paired-cymbal RMS by `margin_db` is flipped to its tom counterpart
    # (same lane, is_cymbal=False).  Validated 2026-04-27 across 3 songs at
    # +40 dB mean margin and ≥98% Tom→Cym correction rate.
    if (
        os.environ.get("STRUM_USE_DRUMSEP", "1") == "1"
        and os.environ.get("STRUM_DRUMSEP_FINAL_PASS", "1") == "1"
        and chart.hits
    ):
        stems_dir = song_folder / "demucs_temp" / "stems_6"
        if stems_dir.exists():
            t0 = time.time()
            chart = apply_drumsep_hits_arbiter(
                chart, stems_dir,
                win_ms=float(os.environ.get("STRUM_DRUMSEP_WIN_MS", "50")),
                margin_db=float(os.environ.get("STRUM_DRUMSEP_MARGIN_DB", "20")),
                toms_floor_db=float(os.environ.get("STRUM_DRUMSEP_TOMS_FLOOR_DB", "-40")),
            )
            logger.info(f"  Drumsep final arbiter done ({time.time() - t0:.1f}s)")
            _trace_snapshot(chart, "after_drumsep_final", song_folder)

            # 6h1. Snare-in-HH-roll veto. Bug #1 (Apr 2026 ear-test):
            # phantom snares stacked with hi-hats during dense rolls.
            # Music-context veto using snare.wav vs hh.wav RMS.
            # OFF by default until 12-song eval validates pure improvement.
            # ON by default (v25.2, Apr 2026). Targets bug #1: phantom
            # snares stacked with hi-hats during dense rolls. Tuned to
            # 100% veto precision on Every Avenue (10/10 correct, 0 false
            # vetoes) by requiring snare-stem RMS to DROP ≥8 dB after
            # onset — a fingerprint of "snare bleed" rather than real hit.
            if os.environ.get("STRUM_SNARE_HH_ROLL_VETO", "1") == "1":
                t1 = time.time()
                chart = apply_snare_in_hh_roll_veto(
                    chart, stems_dir,
                    hh_window_ms=float(os.environ.get("STRUM_SHHR_HH_WIN_MS", "300")),
                    min_hh_neighbours=int(os.environ.get("STRUM_SHHR_MIN_HH", "4")),
                    max_other_snares=int(os.environ.get("STRUM_SHHR_MAX_OTHER", "1")),
                    attack_db=float(os.environ.get("STRUM_SHHR_ATTACK_DB", "-8.0")),
                )
                logger.info(f"  Snare-in-HH-roll veto done ({time.time() - t1:.1f}s)")
                _trace_snapshot(chart, "after_snare_hh_roll_veto", song_folder)

            # 6h1b. Crash+Ride co-stack veto. Bug (May 2026 ear-test):
            # pure crash hits get a phantom Ride charted alongside them.
            # Investigated across all 12 sample songs: 95.7 % of
            # Crash+Ride co-stacks should drop the Ride (only 1/230
            # cases across 12 songs is a legit Both stack). Pure rule:
            # if Ride is within ±30 ms of a Crash, drop it.
            # ON by default. Disable with STRUM_CR_COSTACK_VETO=0.
            if os.environ.get("STRUM_CR_COSTACK_VETO", "1") == "1":
                t1 = time.time()
                chart = apply_crash_ride_costack_veto(
                    chart,
                    stack_tol_ms=float(os.environ.get("STRUM_CRCV_TOL_MS", "30.0")),
                )
                logger.info(f"  Crash+Ride co-stack veto done ({time.time() - t1:.1f}s)")
                _trace_snapshot(chart, "after_cr_costack_veto", song_folder)

            # 6h2. Kick double-rescue. Bug #2 (Apr 2026 ear-test): missing
            # middle of fast 3-kick patterns. Music-context rescue using
            # kick.wav stem onsets at midpoints between chart-kick pairs.
            # OFF by default — prototype validated 100% precision on raw
            # post-drumsep predictions, but post-quantization geometry +
            # offset drift makes the function unreliable in the pipeline.
            # Set STRUM_KICK_DOUBLE_RESCUE=1 to experiment.
            if os.environ.get("STRUM_KICK_DOUBLE_RESCUE", "0") == "1":
                t1 = time.time()
                chart = apply_kick_double_rescue(
                    chart, stems_dir,
                    gap_min_ms=float(os.environ.get("STRUM_KDR_GAP_MIN_MS", "220")),
                    gap_max_ms=float(os.environ.get("STRUM_KDR_GAP_MAX_MS", "400")),
                    midpoint_tol_ms=float(os.environ.get("STRUM_KDR_MID_TOL_MS", "50")),
                    overlap_tol_ms=float(os.environ.get("STRUM_KDR_OVERLAP_TOL_MS", "60")),
                    attack_db=float(os.environ.get("STRUM_KDR_ATTACK_DB", "4.0")),
                )
                logger.info(f"  Kick double-rescue done ({time.time() - t1:.1f}s)")
                _trace_snapshot(chart, "after_kick_double_rescue", song_folder)

    # 6h2. Crash-vs-Ride pairwise arbiter — DISABLED by default.
    # Empirically the pairwise pattern (validated for tom-vs-cymbal) does NOT
    # transfer to crash-vs-ride: crash sustain rings for seconds, so at later
    # real ride hits `crash.wav` RMS > `ride.wav` RMS, causing wrong flips.
    # On Alkaline, 11 flips made Crash→Ride drift slightly worse (58→60).
    # See /memories/repo/strum-stem-veto-findings.md.  Keep code for further
    # experimentation (e.g. with attack-only signal); off by default.
    if (
        os.environ.get("STRUM_USE_DRUMSEP", "1") == "1"
        and os.environ.get("STRUM_DRUMSEP_CR_PASS", "0") == "1"
        and chart.hits
    ):
        stems_dir = song_folder / "demucs_temp" / "stems_6"
        if stems_dir.exists():
            t0 = time.time()
            chart = apply_crash_ride_arbiter(
                chart, stems_dir,
                win_ms=float(os.environ.get("STRUM_DRUMSEP_WIN_MS", "50")),
                margin_db=float(os.environ.get("STRUM_DRUMSEP_CR_MARGIN_DB", "6")),
                floor_db=float(os.environ.get("STRUM_DRUMSEP_CR_FLOOR_DB", "-45")),
            )
            logger.info(f"  Crash/ride arbiter done ({time.time() - t0:.1f}s)")

    # 6i. (DISABLED by default — see /memories/repo/strum-stem-veto-findings.md)
    # The per-hit stem-attack veto fundamentally fails: cymbal sustain and
    # tom-roll decay corrupt the pre/post window comparison, dropping 30%+ of
    # real hits at any threshold.  Per-stem librosa onset detection also fails
    # (only 15-30% of real hits have a stem onset within ±25ms).  Universal
    # vetoes don't work — only pairwise RMS arbiters do (validated tom-vs-cym
    # at -88%).  Keep the function for experimentation; default to OFF.
    if (
        os.environ.get("STRUM_USE_DRUMSEP", "1") == "1"
        and os.environ.get("STRUM_STEM_VETO", "0") == "1"
        and chart.hits
    ):
        stems_dir = song_folder / "demucs_temp" / "stems_6"
        if stems_dir.exists():
            t0 = time.time()
            chart = apply_stem_attack_veto(
                chart, stems_dir,
                attack_db=float(os.environ.get("STRUM_STEM_VETO_ATTACK_DB", "1.0")),
                active_floor_db=float(os.environ.get("STRUM_STEM_VETO_FLOOR_DB", "-50.0")),
                loud_rescue_db=float(os.environ.get("STRUM_STEM_VETO_LOUD_DB", "-25.0")),
            )
            logger.info(f"  Stem-attack veto done ({time.time() - t0:.1f}s)")

    # 6z. v25.7 phase alignment.
    # Trim the leading "pre-beat" silence from song.ogg and shift all hits +
    # tempo events by the same amount so MIDI tick 0 (= bar 1 in CH) coincides
    # with the audio's first detected beat. Without this, bar lines render at
    # chart_time 0, bar_period, ... but audio downbeats are at +phase_offset_ms,
    # so bars never align with notes during gameplay.
    phase_offset_ms = float(audio_info.get("phase_offset_ms", 0.0))
    if os.environ.get("STRUM_PHASE_ALIGN", "1") == "1" and phase_offset_ms >= 1.0:
        # Drop hits that would land before tick 0 after shift.
        kept = []
        for h in chart.hits:
            new_t = h.time_ms - phase_offset_ms
            if new_t >= 0.0:
                h.time_ms = new_t
                kept.append(h)
        chart.hits = kept
        # Shift tempo events similarly; first event stays at time_ms=0.
        for te in chart.tempo_events:
            te.time_ms = max(0.0, te.time_ms - phase_offset_ms)
        # Update reported duration so song.ini matches trimmed audio.
        audio_info["duration_ms"] = max(0.0, audio_info["duration_ms"] - phase_offset_ms)
        logger.info(
            f"  Phase align: trimmed {phase_offset_ms:.1f} ms from audio start; "
            f"{len(chart.hits)} hits remain"
        )

    # 7. Export
    notes_path = song_folder / "notes.mid"
    export_all_difficulties(chart, notes_path)
    logger.info(f"  Created: notes.mid ({len(chart.hits)} Expert hits)")

    # Use full mix audio for song.ogg, not drums stem
    song_ogg = song_folder / "song.ogg"
    ogg_source = audio_path
    if full_audio_dir is not None:
        for ext in (".mp3", ".ogg", ".wav", ".flac"):
            candidate = full_audio_dir / (audio_path.stem + ext)
            if candidate.exists():
                ogg_source = candidate
                break
    convert_to_ogg(ogg_source, song_ogg, trim_start_ms=phase_offset_ms if os.environ.get("STRUM_PHASE_ALIGN", "1") == "1" else 0.0)
    logger.info(f"  Created: song.ogg (from {ogg_source.name})")

    create_song_ini(
        song_folder, artist, title,
        audio_info["tempo_bpm"], audio_info["duration_ms"],
    )
    logger.info(f"  Created: song.ini")

    return song_folder


def main():
    parser = argparse.ArgumentParser(description="Hybrid drum transcription pipeline")
    parser.add_argument("--skip-separation", action="store_true",
                        help="Skip Demucs separation (use input as drums stem)")
    parser.add_argument("--onset-threshold", type=float, default=0.50,
                        help="V14 onset detection threshold (default: 0.35)")
    parser.add_argument("--input-dir", type=str, default="input",
                        help="Input directory with audio files")
    parser.add_argument("--output-dir", type=str, default="output/hybrid",
                        help="Output directory for song folders")
    parser.add_argument("--no-two-pass", action="store_true",
                        help="Disable two-pass context (faster but less accurate)")
    parser.add_argument("--no-postprocess", action="store_true",
                        help="Disable chart post-processing")
    parser.add_argument("--full-audio-dir", type=str, default=None,
                        help="Directory with original full-mix audio (for song.ogg when using --skip-separation)")
    parser.add_argument("--v14-only", action="store_true",
                        help="Use V14 Stage 2 classifier directly (skip ensemble)")
    parser.add_argument("--v14-checkpoint", type=str, default=None,
                        help="Override V14 checkpoint path (e.g. frame_classifier_v3/best.pt)")
    args = parser.parse_args()

    INPUT_DIR = Path(args.input_dir)
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Find input songs
    songs = sorted(
        f for f in INPUT_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not songs:
        songs = sorted(
            f for f in INPUT_DIR.iterdir()
            if f.is_symlink() or (f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS)
        )
    logger.info(f"Found {len(songs)} songs in {INPUT_DIR}")

    # Load models
    logger.info("\n═══ Loading models ═══")
    if args.v14_checkpoint:
        global V14_CHECKPOINT
        V14_CHECKPOINT = args.v14_checkpoint
        logger.info(f"  V14 checkpoint override: {args.v14_checkpoint}")
    v14_model = load_v14_onset_detector(device)
    ensemble = load_ensemble(device)
    tomcym = load_tomcym_classifier(device)
    tom_refinement = load_tom_refinement(device)
    phase3_model = load_phase3_model(device)

    if not ensemble and not args.v14_only:
        logger.error("No ensemble models loaded! Exiting.")
        sys.exit(1)
    if args.v14_only:
        logger.info("  V14-only mode: using Stage 2 classifier directly (no ensemble)")

    # Process each song
    logger.info(f"\n═══ Processing {len(songs)} songs ═══")
    results = []
    for i, song_path in enumerate(songs):
        logger.info(f"\n{'─' * 60}")
        logger.info(f"[{i+1}/{len(songs)}] {song_path.name}")
        logger.info(f"{'─' * 60}")

        t0 = time.time()
        try:
            folder = process_song(
                v14_model, ensemble, song_path, OUTPUT_DIR, device,
                skip_separation=args.skip_separation,
                onset_threshold=args.onset_threshold,
                two_pass_context=not args.no_two_pass,
                postprocess=not args.no_postprocess,
                full_audio_dir=Path(args.full_audio_dir) if args.full_audio_dir else None,
                tomcym=tomcym,
                tom_refinement=tom_refinement,
                phase3_model=phase3_model,
                v14_only=args.v14_only,
            )
            elapsed = time.time() - t0
            results.append({"song": song_path.name, "folder": str(folder), "time": elapsed})
            logger.info(f"  Done in {elapsed:.1f}s")
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append({"song": song_path.name, "error": str(e)})

    # Summary
    logger.info(f"\n{'═' * 60}")
    logger.info("SUMMARY")
    logger.info(f"{'═' * 60}")
    for r in results:
        if "error" in r:
            logger.info(f"  FAIL: {r['song']} — {r['error']}")
        else:
            logger.info(f"  OK:   {r['song']} ({r['time']:.1f}s)")

    logger.info(f"\nOutput: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
