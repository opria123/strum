"""
Chart Enhancer - Adds Star Power, Difficulty Reduction, and Practice Sections.

Based on common approaches from Moonscraper, EOF, and Clone Hero charting tools:
- Star Power: Energy-based detection, placed at musical peaks
- Difficulty Reduction: Note density + chord simplification 
- Practice Sections: Audio structural analysis for verse/chorus detection
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np
import librosa
import mido

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Section:
    """A song section (verse, chorus, etc.)"""
    name: str
    start_time: float
    end_time: float


class ChartEnhancer:
    """
    Enhances Clone Hero/YARG charts with:
    - Star Power phrases based on musical energy
    - Difficulty reduction (Hard/Medium/Easy from Expert)
    - Practice sections from audio structure analysis
    """
    
    # MIDI note numbers for Clone Hero
    SP_NOTE = 116  # Star Power phrase marker
    SOLO_NOTE = 103  # Solo marker
    DRUM_FILL_NOTES = [120, 121, 122, 123, 124]  # Drum fill/activation lanes
    
    # Difficulty note ranges
    EXPERT_RANGE = (96, 100)
    HARD_RANGE = (84, 88)
    MEDIUM_RANGE = (72, 76)
    EASY_RANGE = (60, 64)
    
    def __init__(self, tempo_bpm: float = 120, ticks_per_beat: int = 480):
        self.tempo_bpm = tempo_bpm
        self.ticks_per_beat = ticks_per_beat
        self.ticks_per_sec = ticks_per_beat * tempo_bpm / 60
    
    # =========================================================================
    # STAR POWER DETECTION (Phrase-Aligned)
    # =========================================================================
    
    def add_phrase_aligned_star_power(
        self,
        track: mido.MidiTrack,
        track_type: str = 'instrument',  # 'instrument', 'drums', 'vocals', 'prokeys'
        num_sp_phrases: int = 8,
        sp_phrase_beats: int = 8,  # Length of each SP phrase in beats
    ) -> mido.MidiTrack:
        """
        Add Star Power markers aligned with actual note content in the track.
        
        This ensures SP phrases overlap with playable notes, which is required
        for SP to be earnable in Clone Hero/YARG. The approach:
        
        1. Find all notes in the Expert range for the instrument
        2. Divide song into windows and find note density per window
        3. Select top N densest windows that are well-spaced
        4. Place SP markers exactly covering those note-dense sections
        
        Args:
            track: The instrument track
            track_type: Type of track for note range detection
            num_sp_phrases: Target number of SP phrases (default 8)
            sp_phrase_beats: Length of each SP phrase in beats
        
        Returns:
            Track with SP markers aligned to note phrases
        """
        # Determine note ranges based on track type
        if track_type == 'drums':
            expert_range = (95, 100)  # Drums use 95-100
        elif track_type == 'prokeys':
            expert_range = (48, 72)  # Pro Keys use actual MIDI pitches
        elif track_type == 'vocals':
            # Vocals use lyric phrases (note 105) - delegate to specialized method
            return self.add_vocals_star_power(track, num_sp_phrases)
        else:
            expert_range = (96, 100)  # Standard 5-lane (guitar, bass, keys)
        
        # Collect events and find note start times
        events = []
        note_starts = []  # List of start ticks for notes
        active_notes = {}  # note -> start_tick
        abs_tick = 0
        max_tick = 0
        
        for msg in track:
            abs_tick += msg.time
            max_tick = max(max_tick, abs_tick)
            if msg.type != 'end_of_track':
                events.append((abs_tick, msg))
            
            # Track note ranges
            if msg.type == 'note_on' and expert_range[0] <= msg.note <= expert_range[1]:
                if msg.velocity > 0:
                    note_starts.append(abs_tick)
                    active_notes[msg.note] = abs_tick
        
        if not note_starts:
            logger.debug(f"No notes found in expert range {expert_range} for SP alignment")
            return track
        
        # Calculate window size and step
        sp_phrase_ticks = sp_phrase_beats * self.ticks_per_beat
        min_gap_ticks = sp_phrase_ticks * 2  # At least 2 phrase lengths between SP
        
        # Skip first/last 10 seconds to avoid intro/outro
        skip_ticks = int(10 * self.ticks_per_sec)
        first_note = min(note_starts)
        last_note = max(note_starts)
        
        # Calculate note density in sliding windows
        window_size = sp_phrase_ticks
        step_size = self.ticks_per_beat * 2  # Step by 2 beats
        
        windows = []  # List of (start_tick, end_tick, note_count)
        
        current_start = max(first_note, skip_ticks)
        while current_start + window_size < min(last_note, max_tick - skip_ticks):
            window_end = current_start + window_size
            
            # Count notes in this window
            count = sum(1 for t in note_starts if current_start <= t < window_end)
            
            if count > 0:
                windows.append((current_start, window_end, count))
            
            current_start += step_size
        
        if not windows:
            # Fallback: use any window with notes
            current_start = first_note
            while current_start + window_size < last_note:
                window_end = current_start + window_size
                count = sum(1 for t in note_starts if current_start <= t < window_end)
                if count > 0:
                    windows.append((current_start, window_end, count))
                current_start += step_size
        
        if not windows:
            logger.debug(f"No windows found for SP in {track_type} track")
            return track
        
        # Select best windows - prioritize density but ensure distribution
        # Sort by density (highest first)
        sorted_windows = sorted(windows, key=lambda x: -x[2])
        
        selected_phrases = []
        for start, end, count in sorted_windows:
            # Check gap from already selected phrases
            too_close = False
            for sel_start, sel_end in selected_phrases:
                if abs(start - sel_start) < min_gap_ticks:
                    too_close = True
                    break
            
            if not too_close:
                selected_phrases.append((start, end))
            
            if len(selected_phrases) >= num_sp_phrases:
                break
        
        # If we don't have enough, relax the gap requirement
        if len(selected_phrases) < num_sp_phrases:
            min_gap_ticks = sp_phrase_ticks  # Just one phrase length
            for start, end, count in sorted_windows:
                if (start, end) in selected_phrases:
                    continue
                
                too_close = False
                for sel_start, sel_end in selected_phrases:
                    if abs(start - sel_start) < min_gap_ticks:
                        too_close = True
                        break
                
                if not too_close:
                    selected_phrases.append((start, end))
                
                if len(selected_phrases) >= num_sp_phrases:
                    break
        
        # Sort selected phrases by time
        selected_phrases.sort(key=lambda x: x[0])
        
        logger.debug(f"Selected {len(selected_phrases)} SP phrases for {track_type}")
        
        # Remove any existing SP markers
        new_events = [
            (tick, msg) for tick, msg in events
            if not (msg.type in ('note_on', 'note_off') and msg.note == self.SP_NOTE)
        ]
        
        # Add SP markers for selected phrases
        for start_tick, end_tick in selected_phrases:
            new_events.append((start_tick, mido.Message('note_on', note=self.SP_NOTE, velocity=100, time=0)))
            new_events.append((end_tick, mido.Message('note_off', note=self.SP_NOTE, velocity=0, time=0)))
        
        # Sort by time (note_on before note_off at same tick)
        new_events.sort(key=lambda e: (e[0], 0 if hasattr(e[1], 'velocity') and getattr(e[1], 'velocity', 0) > 0 else 1))
        
        # Rebuild track with delta times
        new_track = mido.MidiTrack()
        prev_tick = 0
        for tick, msg in new_events:
            msg_copy = msg.copy()
            msg_copy.time = tick - prev_tick
            new_track.append(msg_copy)
            prev_tick = tick
        
        new_track.append(mido.MetaMessage('end_of_track', time=0))
        return new_track
    
    def add_vocals_star_power(
        self,
        track: mido.MidiTrack,
        num_sp_phrases: int = 8,
    ) -> mido.MidiTrack:
        """
        Add Star Power markers to vocals aligned with lyric phrases.
        
        For vocals, SP markers (note 116) MUST overlap with lyric phrases
        (note 105) to be earnable. This method finds all lyric phrases and
        distributes SP across a subset of them.
        
        Args:
            track: The vocals track (PART VOCALS, HARM1, etc.)
            num_sp_phrases: Number of phrases to mark as SP (default 8)
        
        Returns:
            Track with SP markers aligned to lyric phrases
        """
        # Collect all events with absolute times
        events = []
        phrase_events = []  # Note 105 events (lyric phrases)
        abs_tick = 0
        
        for msg in track:
            abs_tick += msg.time
            if msg.type != 'end_of_track':
                events.append((abs_tick, msg))
            
            # Track phrase markers (note 105)
            if msg.type == 'note_on' and msg.note == 105:
                if msg.velocity > 0:
                    phrase_events.append(('start', abs_tick))
                else:
                    phrase_events.append(('end', abs_tick))
            elif msg.type == 'note_off' and msg.note == 105:
                phrase_events.append(('end', abs_tick))
        
        # Build phrase ranges from note 105 events
        phrases = []
        phrase_start = None
        for etype, tick in phrase_events:
            if etype == 'start':
                phrase_start = tick
            elif etype == 'end' and phrase_start is not None:
                phrases.append((phrase_start, tick))
                phrase_start = None
        
        if not phrases:
            logger.warning("No lyric phrases (note 105) found in vocals track")
            return track
        
        # Select phrases for SP - distribute evenly across the song
        sp_phrase_indices = []
        if len(phrases) >= num_sp_phrases:
            # Distribute SP across song - pick every Nth phrase
            step = max(2, len(phrases) // num_sp_phrases)
            for i in range(1, len(phrases), step):
                sp_phrase_indices.append(i)
                if len(sp_phrase_indices) >= num_sp_phrases:
                    break
        else:
            # Fewer phrases than desired SP - mark every other phrase
            sp_phrase_indices = list(range(0, len(phrases), 2))[:num_sp_phrases]
        
        logger.debug(f"Marking {len(sp_phrase_indices)} of {len(phrases)} lyric phrases as SP")
        
        # Remove any existing SP markers (note 116)
        new_events = []
        for tick, msg in events:
            if msg.type in ('note_on', 'note_off') and msg.note == self.SP_NOTE:
                continue
            new_events.append((tick, msg))
        
        # Add SP markers aligned with selected lyric phrases
        for idx in sp_phrase_indices:
            if idx < len(phrases):
                start_tick, end_tick = phrases[idx]
                new_events.append((start_tick, mido.Message('note_on', note=self.SP_NOTE, velocity=100, time=0)))
                new_events.append((end_tick, mido.Message('note_off', note=self.SP_NOTE, velocity=0, time=0)))
        
        # Sort by time (note_on before note_off at same time)
        new_events.sort(key=lambda e: (e[0], 0 if hasattr(e[1], 'velocity') and getattr(e[1], 'velocity', 0) > 0 else 1))
        
        # Rebuild track with delta times
        new_track = mido.MidiTrack()
        prev_tick = 0
        for tick, msg in new_events:
            msg_copy = msg.copy()
            msg_copy.time = tick - prev_tick
            new_track.append(msg_copy)
            prev_tick = tick
        
        new_track.append(mido.MetaMessage('end_of_track', time=0))
        return new_track
    
    # =========================================================================
    # DIFFICULTY REDUCTION
    # =========================================================================
    
    def reduce_difficulty(
        self,
        track: mido.MidiTrack,
        source_range: Tuple[int, int] = (96, 100),  # Expert
        target_range: Tuple[int, int] = (84, 88),   # Hard
        keep_ratio: float = 0.8,
        simplify_chords: bool = False,
        max_lanes: int = 5,
    ) -> mido.MidiTrack:
        """
        Reduce difficulty by removing notes and simplifying patterns.
        
        Based on EOF/Moonscraper difficulty reduction approaches:
        - Keep ratio: percentage of notes to keep
        - Simplify chords: reduce multi-note chords to single notes
        - Max lanes: limit to N lanes for easier difficulties
        
        Priority for keeping notes:
        1. Downbeats (beat 1 of each measure)
        2. Backbeats (beats 2 and 4)
        3. Other on-beat notes
        4. Off-beat notes (lowest priority)
        """
        import random
        random.seed(42)  # Deterministic
        
        # Extract notes from source range
        notes = []  # (abs_tick, note, velocity, duration)
        note_starts = {}  # Track note_on events
        
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            
            if msg.type == 'note_on' and source_range[0] <= msg.note <= source_range[1]:
                if msg.velocity > 0:
                    note_starts[(msg.note, abs_tick)] = (abs_tick, msg.note, msg.velocity)
                    
            elif msg.type == 'note_off' and source_range[0] <= msg.note <= source_range[1]:
                # Find matching note_on
                for key, (start_tick, note, vel) in list(note_starts.items()):
                    if key[0] == msg.note:
                        duration = abs_tick - start_tick
                        notes.append((start_tick, note, vel, duration))
                        del note_starts[key]
                        break
        
        # Group notes by time (for chord detection)
        notes_by_time = {}
        for start_tick, note, vel, dur in notes:
            if start_tick not in notes_by_time:
                notes_by_time[start_tick] = []
            notes_by_time[start_tick].append((note, vel, dur))
        
        # Calculate beat positions for priority
        ticks_per_measure = self.ticks_per_beat * 4  # 4/4 time
        
        def get_beat_priority(tick: int) -> int:
            """Higher priority = more important to keep."""
            beat_in_measure = (tick % ticks_per_measure) / self.ticks_per_beat
            
            if beat_in_measure < 0.1:  # Downbeat
                return 4
            elif abs(beat_in_measure - 2) < 0.1 or abs(beat_in_measure - 4) < 0.1:  # Backbeat
                return 3
            elif beat_in_measure % 1.0 < 0.1:  # On-beat
                return 2
            else:  # Off-beat
                return 1
        
        # Decide which notes to keep
        sorted_times = sorted(notes_by_time.keys())
        kept_notes = []
        
        for tick in sorted_times:
            chord_notes = notes_by_time[tick]
            priority = get_beat_priority(tick)
            
            # Higher priority notes more likely to be kept
            keep_prob = keep_ratio * (0.5 + 0.5 * priority / 4)
            
            if random.random() < keep_prob:
                if simplify_chords and len(chord_notes) > 1:
                    # Keep only the lowest note (root)
                    chord_notes = [min(chord_notes, key=lambda x: x[0])]
                
                for note, vel, dur in chord_notes:
                    # Remap to target range
                    lane = note - source_range[0]
                    
                    # Limit lanes for easier difficulties
                    if max_lanes < 5:
                        lane = min(lane, max_lanes - 1)
                    
                    new_note = target_range[0] + lane
                    kept_notes.append((tick, new_note, vel, dur))
        
        # Build new track
        new_track = mido.MidiTrack()
        
        # Copy track name
        for msg in track:
            if msg.type == 'track_name':
                new_track.append(msg.copy())
                break
        
        # Add kept notes
        events = []
        for tick, note, vel, dur in kept_notes:
            events.append((tick, 'on', note, vel))
            events.append((tick + dur, 'off', note, 0))
        
        # Also copy SP markers and other non-note events
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type in ('note_on', 'note_off'):
                # Copy SP markers (note 116) and solo markers (note 103)
                if msg.note in [self.SP_NOTE, self.SOLO_NOTE]:
                    events.append((abs_tick, msg.type.split('_')[1], msg.note, msg.velocity))
        
        # Sort and convert to delta times
        events.sort(key=lambda e: (e[0], 0 if e[1] == 'on' else 1))
        
        prev_tick = 0
        for tick, event_type, note, vel in events:
            delta = tick - prev_tick
            if event_type == 'on':
                new_track.append(mido.Message('note_on', note=note, velocity=vel, time=delta))
            else:
                new_track.append(mido.Message('note_off', note=note, velocity=0, time=delta))
            prev_tick = tick
        
        new_track.append(mido.MetaMessage('end_of_track', time=0))
        
        return new_track
    
    # =========================================================================
    # PRACTICE SECTIONS (Song Structure)
    # =========================================================================
    
    def detect_song_sections(
        self,
        audio_path: str,
    ) -> List[Section]:
        """
        Detect song structure (verse, chorus, bridge, etc.) using audio analysis.
        
        Uses librosa's structural analysis:
        - Self-similarity matrix to find repeating sections
        - Spectral clustering for section boundaries
        - Heuristics to label sections
        """
        logger.info("Detecting song structure...")
        
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        duration = len(y) / sr
        
        # Compute features for structure analysis
        hop_length = 512
        
        # Chromagram for harmonic content
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
        
        # MFCC for timbral content
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
        
        # Combine features
        features = np.vstack([chroma, mfcc])
        
        # Compute self-similarity matrix
        from scipy.spatial.distance import cdist
        sim_matrix = 1 - cdist(features.T, features.T, metric='cosine')
        
        # Detect boundaries using novelty curve
        novelty = librosa.segment.recurrence_to_lag(sim_matrix)
        novelty_curve = np.mean(np.abs(np.diff(novelty, axis=1)), axis=0)
        
        # Find peaks in novelty (section boundaries)
        from scipy.signal import find_peaks
        min_section_frames = int(8 * sr / hop_length)  # Minimum 8 seconds per section
        peaks, _ = find_peaks(novelty_curve, distance=min_section_frames, prominence=0.1)
        
        # Convert to times
        boundary_times = [0.0]
        for peak in peaks:
            t = librosa.frames_to_time(peak, sr=sr, hop_length=hop_length)
            if t > 5 and t < duration - 5:  # Skip boundaries too close to start/end
                boundary_times.append(t)
        boundary_times.append(duration)
        
        # Quantize boundaries to nearest beat
        boundary_times = [
            round(t * self.tempo_bpm / 60) * 60 / self.tempo_bpm
            for t in boundary_times
        ]
        
        # Label sections based on position and repetition
        sections = []
        section_features = []
        
        for i in range(len(boundary_times) - 1):
            start = boundary_times[i]
            end = boundary_times[i + 1]
            
            # Get average features for this section
            start_frame = int(start * sr / hop_length)
            end_frame = int(end * sr / hop_length)
            section_feat = np.mean(features[:, start_frame:end_frame], axis=1)
            section_features.append(section_feat)
        
        # Find repeating sections (chorus candidates)
        section_features = np.array(section_features)
        
        # Cluster similar sections
        if len(section_features) > 2:
            from scipy.cluster.hierarchy import fcluster, linkage
            Z = linkage(section_features, method='ward')
            clusters = fcluster(Z, t=2, criterion='maxclust')
        else:
            clusters = list(range(len(section_features)))
        
        # Label sections based on position and energy
        verse_count = 0
        chorus_count = 0
        bridge_count = 0
        
        # Calculate section energies for chorus detection
        section_energies = []
        for i in range(len(boundary_times) - 1):
            start = boundary_times[i]
            end = boundary_times[i + 1]
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            energy = np.sqrt(np.mean(y[start_sample:end_sample] ** 2))
            section_energies.append(energy)
        
        median_energy = np.median(section_energies)
        
        for i in range(len(boundary_times) - 1):
            rel_position = (boundary_times[i] + boundary_times[i + 1]) / 2 / duration
            section_len = boundary_times[i + 1] - boundary_times[i]
            is_high_energy = section_energies[i] > median_energy * 1.1
            
            if i == 0 and boundary_times[i] < 15:
                name = "Intro"
            elif i == len(boundary_times) - 2 and boundary_times[i + 1] > duration - 20:
                name = "Outro"
            elif 0.4 < rel_position < 0.7 and not is_high_energy and section_len > 15:
                # Bridge: middle of song, lower energy, longer section
                bridge_count += 1
                name = f"Bridge {bridge_count}" if bridge_count > 1 else "Bridge"
            elif is_high_energy:
                # Chorus: high energy sections
                chorus_count += 1
                name = f"Chorus {chorus_count}" if chorus_count > 1 else "Chorus"
            else:
                # Verse: normal energy
                verse_count += 1
                name = f"Verse {verse_count}" if verse_count > 1 else "Verse"
            
            sections.append(Section(
                name=name,
                start_time=boundary_times[i],
                end_time=boundary_times[i + 1],
            ))
        
        logger.info(f"Detected {len(sections)} sections:")
        for s in sections:
            logger.info(f"  {s.name}: {s.start_time:.1f}s - {s.end_time:.1f}s")
        
        return sections
    
    def add_sections_to_events(
        self,
        events_track: mido.MidiTrack,
        sections: List[Section],
    ) -> mido.MidiTrack:
        """Add practice sections to EVENTS track."""
        # Collect existing events
        events = []
        abs_tick = 0
        for msg in events_track:
            abs_tick += msg.time
            if msg.type != 'end_of_track':
                events.append((abs_tick, msg))
        
        # Add section markers
        for section in sections:
            tick = int(section.start_time * self.ticks_per_sec)
            section_text = f"[section {section.name}]"
            events.append((tick, mido.MetaMessage('text', text=section_text, time=0)))
        
        # Sort by time
        events.sort(key=lambda e: e[0])
        
        # Rebuild track with delta times
        new_track = mido.MidiTrack()
        prev_tick = 0
        for tick, msg in events:
            msg_copy = msg.copy()
            msg_copy.time = tick - prev_tick
            new_track.append(msg_copy)
            prev_tick = tick
        
        new_track.append(mido.MetaMessage('end_of_track', time=0))
        return new_track
    
    # =========================================================================
    # MAIN ENHANCE METHOD
    # =========================================================================
    
    def enhance_chart(
        self,
        midi_path: str,
        audio_path: str,
        output_path: str,
    ):
        """
        Enhance a chart with Star Power, difficulty reduction, and sections.
        """
        logger.info(f"Enhancing chart: {midi_path}")
        
        # Load MIDI
        midi = mido.MidiFile(midi_path)
        self.ticks_per_beat = midi.ticks_per_beat
        
        # Get tempo from MIDI
        for track in midi.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    self.tempo_bpm = round(mido.tempo2bpm(msg.tempo))
                    break
        
        self.ticks_per_sec = self.ticks_per_beat * self.tempo_bpm / 60
        logger.info(f"Tempo: {self.tempo_bpm} BPM, Resolution: {self.ticks_per_beat}")
        
        # Note: Star Power is now phrase-aligned per track, not energy-based
        # Each track's SP is placed to overlap actual notes for that instrument
        
        # Detect song structure
        sections = self.detect_song_sections(audio_path)
        
        # Process each track
        new_tracks = []
        
        for track in midi.tracks:
            track_name = None
            for msg in track:
                if msg.type == 'track_name':
                    track_name = msg.name
                    break
            
            if track_name in ['PART GUITAR', 'PART BASS']:
                # Add Star Power aligned with actual note phrases
                enhanced = self.add_phrase_aligned_star_power(track, track_type='instrument')
                new_tracks.append(enhanced)
                
            elif track_name == 'PART DRUMS':
                # Add SP aligned with drum phrases and create drum fills
                enhanced = self.add_phrase_aligned_star_power(track, track_type='drums')
                # Get SP sections for drum fills
                sp_sections = self._extract_sp_sections(enhanced)
                enhanced = self.add_drum_fills(enhanced, sp_sections)
                new_tracks.append(enhanced)
            
            elif track_name in ['PART VOCALS', 'HARM1', 'HARM2', 'HARM3']:
                # Add Star Power aligned with lyric phrases (note 105)
                # Vocals SP (note 116) must overlap lyric phrases to be earnable
                enhanced = self.add_phrase_aligned_star_power(track, track_type='vocals')
                new_tracks.append(enhanced)
            
            elif track_name == 'PART KEYS':
                # 5-lane keys - same as guitar/bass
                enhanced = self.add_phrase_aligned_star_power(track, track_type='instrument')
                new_tracks.append(enhanced)
            
            elif track_name in ['PART REAL_KEYS_E', 'PART REAL_KEYS_M', 
                               'PART REAL_KEYS_H', 'PART REAL_KEYS_X']:
                # Pro Keys - uses actual MIDI pitches
                enhanced = self.add_phrase_aligned_star_power(track, track_type='prokeys')
                new_tracks.append(enhanced)
                
            elif track_name == 'EVENTS':
                # Add practice sections
                enhanced = self.add_sections_to_events(track, sections)
                new_tracks.append(enhanced)
                
            else:
                new_tracks.append(track)
        
        # Now handle difficulty reduction for guitar/bass
        # We need to find the tracks and modify the non-Expert notes
        for i, track in enumerate(new_tracks):
            track_name = None
            for msg in track:
                if msg.type == 'track_name':
                    track_name = msg.name
                    break
            
            if track_name in ['PART GUITAR', 'PART BASS']:
                new_tracks[i] = self.apply_difficulty_reduction(track)
            elif track_name == 'PART DRUMS':
                new_tracks[i] = self.apply_drums_difficulty_reduction(track)
            elif track_name == 'PART KEYS':
                new_tracks[i] = self.apply_keys_difficulty_reduction(track)
        
        # Save enhanced MIDI
        midi.tracks = new_tracks
        midi.save(output_path)
        logger.info(f"Enhanced chart saved to: {output_path}")
        
        # Calculate and update difficulty ratings
        audio_duration = librosa.get_duration(path=audio_path)
        difficulty_ratings = self.calculate_difficulty_ratings(output_path, audio_duration)
        
        # Update song.ini if it exists
        from pathlib import Path
        ini_path = Path(output_path).parent / 'song.ini'
        if ini_path.exists():
            self.update_song_ini(str(ini_path), difficulty_ratings)
    
    def _extract_sp_sections(self, track: mido.MidiTrack) -> List[Tuple[float, float]]:
        """Extract SP sections (note 116) from a track as (start_sec, end_sec) tuples."""
        sp_sections = []
        sp_start_tick = None
        abs_tick = 0
        
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'note_on' and msg.note == self.SP_NOTE:
                if msg.velocity > 0:
                    sp_start_tick = abs_tick
                else:
                    if sp_start_tick is not None:
                        sp_sections.append((
                            sp_start_tick / self.ticks_per_sec,
                            abs_tick / self.ticks_per_sec
                        ))
                        sp_start_tick = None
            elif msg.type == 'note_off' and msg.note == self.SP_NOTE:
                if sp_start_tick is not None:
                    sp_sections.append((
                        sp_start_tick / self.ticks_per_sec,
                        abs_tick / self.ticks_per_sec
                    ))
                    sp_start_tick = None
        
        return sp_sections
    
    def add_drum_fills(
        self,
        track: mido.MidiTrack,
        sp_sections: List[Tuple[float, float]],
    ) -> mido.MidiTrack:
        """Add drum fill markers before SP sections for activation."""
        events = []
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type != 'end_of_track':
                events.append((abs_tick, msg))
        
        # Add drum fill markers 2 beats before each SP section ends
        fill_duration_ticks = self.ticks_per_beat * 2
        
        for start_sec, end_sec in sp_sections:
            fill_start = int(end_sec * self.ticks_per_sec) - fill_duration_ticks
            fill_end = int(end_sec * self.ticks_per_sec)
            
            # Add fill markers on all lanes (120-124)
            for note in self.DRUM_FILL_NOTES:
                events.append((fill_start, mido.Message('note_on', note=note, velocity=100, time=0)))
                events.append((fill_end, mido.Message('note_off', note=note, velocity=0, time=0)))
        
        # Sort and rebuild
        events.sort(key=lambda e: (e[0], 0 if hasattr(e[1], 'velocity') and getattr(e[1], 'velocity', 0) > 0 else 1))
        
        new_track = mido.MidiTrack()
        prev_tick = 0
        for tick, msg in events:
            msg_copy = msg.copy()
            msg_copy.time = tick - prev_tick
            new_track.append(msg_copy)
            prev_tick = tick
        
        new_track.append(mido.MetaMessage('end_of_track', time=0))
        return new_track
    
    def apply_difficulty_reduction(self, track: mido.MidiTrack) -> mido.MidiTrack:
        """
        Apply proper difficulty reduction to Hard/Medium/Easy notes in a track.

        Historically this assumed Hard/Medium/Easy were identical copies of
        Expert and always needed thinning. That's no longer true: the guitar/
        bass track builders already call reduce_to_difficulty() (via
        src/inference/guitar_bass.py) before this runs, so the lower
        difficulties are usually already properly reduced and C3-legal. If we
        don't skip here, we re-thin already-thinned notes and re-apply a
        second, worse chord-shape pass on top of the first one for no
        benefit. Mirrors the equivalent has_hard/has_medium/has_easy guard in
        apply_drums_difficulty_reduction().
        """
        import random
        random.seed(42)

        # Collect all events with absolute times
        events = []
        abs_tick = 0

        for msg in track:
            abs_tick += msg.time
            events.append((abs_tick, msg))

        has_hard = any(84 <= msg.note <= 88 for _, msg in events if msg.type == 'note_on' and msg.velocity > 0)
        has_medium = any(72 <= msg.note <= 76 for _, msg in events if msg.type == 'note_on' and msg.velocity > 0)
        has_easy = any(60 <= msg.note <= 64 for _, msg in events if msg.type == 'note_on' and msg.velocity > 0)

        if has_hard and has_medium and has_easy:
            logger.info("Track already has all difficulties - skipping reduction")
            return track
        
        # Group note events by difficulty and time
        expert_notes = {}  # tick -> list of (note, vel, is_on)
        hard_times_to_remove = set()
        medium_times_to_remove = set()
        easy_times_to_remove = set()
        
        # First pass: find all Expert note times
        for tick, msg in events:
            if msg.type == 'note_on' and 96 <= msg.note <= 100 and msg.velocity > 0:
                if tick not in expert_notes:
                    expert_notes[tick] = []
                expert_notes[tick].append(msg.note)
        
        # Decide which times to thin for each difficulty
        sorted_times = sorted(expert_notes.keys())
        ticks_per_measure = self.ticks_per_beat * 4
        
        for tick in sorted_times:
            beat_pos = (tick % ticks_per_measure) / self.ticks_per_beat
            is_downbeat = beat_pos < 0.1
            is_backbeat = abs(beat_pos - 2) < 0.1 or abs(beat_pos - 4) < 0.1
            is_on_beat = beat_pos % 1.0 < 0.1

            # Hard: Keep 80%, prefer beats
            if not is_downbeat and random.random() > 0.8:
                hard_times_to_remove.add(tick)

            # Medium: Keep 50%, strongly prefer downbeats
            if not is_downbeat and not is_backbeat:
                if random.random() > 0.5:
                    medium_times_to_remove.add(tick)
            elif not is_downbeat and random.random() > 0.7:
                medium_times_to_remove.add(tick)

            # Easy: Keep 30%, only beats
            if not is_on_beat or random.random() > 0.4:
                easy_times_to_remove.add(tick)
        
        # Second pass: filter events
        filtered_events = []
        
        for tick, msg in events:
            if msg.type in ('note_on', 'note_off'):
                note = msg.note
                
                # Hard range (84-88): thin notes
                if 84 <= note <= 88:
                    if tick in hard_times_to_remove:
                        continue
                    # Simplify chords for hard - keep max 2 notes
                    if msg.type == 'note_on' and msg.velocity > 0:
                        expert_tick_notes = expert_notes.get(tick, [])
                        if len(expert_tick_notes) > 2:
                            lane = note - 84
                            expert_lanes = [n - 96 for n in expert_tick_notes]
                            if lane not in expert_lanes[:2]:
                                continue
                
                # Medium range (72-76): more thinning, no chords
                elif 72 <= note <= 76:
                    if tick in medium_times_to_remove:
                        continue
                    # No chords for medium - keep only lowest
                    if msg.type == 'note_on' and msg.velocity > 0:
                        expert_tick_notes = expert_notes.get(tick, [])
                        if len(expert_tick_notes) > 1:
                            lane = note - 72
                            lowest_lane = min(n - 96 for n in expert_tick_notes)
                            if lane != lowest_lane:
                                continue
                
                # Easy range (60-64): heavy thinning, 3 lanes only
                elif 60 <= note <= 64:
                    if tick in easy_times_to_remove:
                        continue
                    # 3 lanes only and no chords
                    lane = note - 60
                    if lane > 2:
                        continue
                    if msg.type == 'note_on' and msg.velocity > 0:
                        expert_tick_notes = expert_notes.get(tick, [])
                        if len(expert_tick_notes) > 1:
                            lowest_lane = min(n - 96 for n in expert_tick_notes)
                            if lane != min(lowest_lane, 2):
                                continue
            
            filtered_events.append((tick, msg))
        
        # Rebuild track
        new_track = mido.MidiTrack()
        prev_tick = 0
        
        for tick, msg in filtered_events:
            if msg.type == 'end_of_track':
                continue
            msg_copy = msg.copy()
            msg_copy.time = tick - prev_tick
            new_track.append(msg_copy)
            prev_tick = tick
        
        new_track.append(mido.MetaMessage('end_of_track', time=0))
        return new_track
    
    def apply_drums_difficulty_reduction(self, track: mido.MidiTrack) -> mido.MidiTrack:
        """
        Apply difficulty reduction to drums track.
        
        Clone Hero Pro Drums note mapping:
        - Expert: 95-100 (but often uses 96-100 like guitar)
        - Hard: 83-88
        - Medium: 71-76
        - Easy: 59-64
        
        Generates lower difficulties from Expert if they don't exist.
        """
        import random
        random.seed(42)
        
        events = []
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            events.append((abs_tick, msg))
        
        # Find Expert drum hits (handle both 95-100 and 96-100 ranges)
        expert_hits = {}  # tick -> list of (note, velocity)
        
        for tick, msg in events:
            if msg.type == 'note_on' and msg.velocity > 0:
                if 95 <= msg.note <= 100 or 96 <= msg.note <= 100:
                    if tick not in expert_hits:
                        expert_hits[tick] = []
                    expert_hits[tick].append((msg.note, msg.velocity))
        
        # Check if lower difficulties exist
        has_hard = any(83 <= msg.note <= 88 for _, msg in events if msg.type == 'note_on' and msg.velocity > 0)
        has_medium = any(71 <= msg.note <= 76 for _, msg in events if msg.type == 'note_on' and msg.velocity > 0)
        has_easy = any(59 <= msg.note <= 64 for _, msg in events if msg.type == 'note_on' and msg.velocity > 0)
        
        if has_hard and has_medium and has_easy:
            logger.info("Drums already has all difficulties - skipping generation")
            return track
        
        sorted_times = sorted(expert_hits.keys())
        ticks_per_measure = self.ticks_per_beat * 4
        
        # Decide which times to keep for each difficulty
        keep_hard = set()
        keep_medium = set()
        keep_easy = set()
        
        for tick in sorted_times:
            beat_pos = (tick % ticks_per_measure) / self.ticks_per_beat
            is_downbeat = beat_pos < 0.1
            is_backbeat = abs(beat_pos - 2) < 0.1 or abs(beat_pos - 4) < 0.1
            is_on_beat = beat_pos % 1.0 < 0.1
            
            # Hard: Keep 85%
            if is_on_beat or random.random() < 0.7:
                keep_hard.add(tick)
            
            # Medium: Keep 50% - mostly on-beat
            if is_on_beat and random.random() < 0.6:
                keep_medium.add(tick)
            
            # Easy: Keep 25% - mostly downbeats and backbeats
            if (is_downbeat or is_backbeat) and random.random() < 0.7:
                keep_easy.add(tick)
        
        # Generate new events with lower difficulties
        # Use a short fixed duration for drum hits (48 ticks = 1/10th beat)
        drum_duration = 48
        new_events = []
        
        # Copy existing events
        for tick, msg in events:
            new_events.append((tick, msg))
        
        # Add lower difficulty notes based on Expert
        for tick, hits in expert_hits.items():
            for note, velocity in hits:
                lane = note - 96  # 0-4 lanes
                if lane < 0:
                    lane = 0  # Handle note 95 (kick)
                
                # Hard (84-88)
                if tick in keep_hard:
                    hard_note = 84 + lane
                    new_events.append((tick, mido.Message('note_on', note=hard_note, velocity=velocity, time=0)))
                    new_events.append((tick + drum_duration, mido.Message('note_off', note=hard_note, velocity=0, time=0)))
                
                # Medium (72-76) - simplify to 4 lanes
                if tick in keep_medium:
                    med_lane = min(lane, 3)  # Max 4 lanes
                    med_note = 72 + med_lane
                    new_events.append((tick, mido.Message('note_on', note=med_note, velocity=velocity, time=0)))
                    new_events.append((tick + drum_duration, mido.Message('note_off', note=med_note, velocity=0, time=0)))
                
                # Easy (60-64) - kick + snare only (lanes 0-1)
                if tick in keep_easy:
                    easy_lane = min(lane, 1)
                    easy_note = 60 + easy_lane
                    new_events.append((tick, mido.Message('note_on', note=easy_note, velocity=velocity, time=0)))
                    new_events.append((tick + drum_duration, mido.Message('note_off', note=easy_note, velocity=0, time=0)))
        
        # Sort and rebuild
        new_events.sort(key=lambda e: (e[0], 0 if hasattr(e[1], 'velocity') and getattr(e[1], 'velocity', 0) > 0 else 1))
        
        new_track = mido.MidiTrack()
        prev_tick = 0
        for tick, msg in new_events:
            if msg.type == 'end_of_track':
                continue
            msg_copy = msg.copy()
            msg_copy.time = tick - prev_tick
            new_track.append(msg_copy)
            prev_tick = tick
        
        new_track.append(mido.MetaMessage('end_of_track', time=0))
        return new_track
    
    def apply_keys_difficulty_reduction(self, track: mido.MidiTrack) -> mido.MidiTrack:
        """
        Apply difficulty reduction to keys track.
        
        Clone Hero 5-lane keys (like guitar):
        - Expert: 96-100
        - Hard: 84-88
        - Medium: 72-76
        - Easy: 60-64
        
        Generates or reduces lower difficulties from Expert.
        """
        import random
        random.seed(42)
        
        events = []
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            events.append((abs_tick, msg))
        
        # Find Expert notes
        expert_notes = {}  # tick -> list of notes
        for tick, msg in events:
            if msg.type == 'note_on' and 96 <= msg.note <= 100 and msg.velocity > 0:
                if tick not in expert_notes:
                    expert_notes[tick] = []
                expert_notes[tick].append((msg.note, msg.velocity))
        
        # Check current note counts per difficulty
        hard_count = sum(1 for _, msg in events if msg.type == 'note_on' and 84 <= msg.note <= 88 and msg.velocity > 0)
        med_count = sum(1 for _, msg in events if msg.type == 'note_on' and 72 <= msg.note <= 76 and msg.velocity > 0)
        easy_count = sum(1 for _, msg in events if msg.type == 'note_on' and 60 <= msg.note <= 64 and msg.velocity > 0)
        expert_count = len(expert_notes)
        
        # If lower difficulties already have notes, just reduce them
        if hard_count > expert_count * 0.5 and med_count > expert_count * 0.3:
            logger.info("Keys already has lower difficulties - reducing existing notes")
            return self._reduce_existing_keys(track, expert_notes)
        
        # Generate lower difficulties from Expert
        logger.info("Keys generating lower difficulties from Expert")
        
        sorted_times = sorted(expert_notes.keys())
        ticks_per_measure = self.ticks_per_beat * 4
        
        keep_hard = set()
        keep_medium = set()
        keep_easy = set()
        
        prev_tick = -9999
        for tick in sorted_times:
            beat_pos = (tick % ticks_per_measure) / self.ticks_per_beat
            is_on_beat = beat_pos % 1.0 < 0.15
            is_downbeat = beat_pos < 0.15
            
            # Preserve melodic runs
            tick_gap = tick - prev_tick
            in_run = tick_gap < self.ticks_per_beat / 2
            prev_tick = tick
            
            # Hard: Keep 75%
            if in_run or is_on_beat or random.random() < 0.6:
                keep_hard.add(tick)
            
            # Medium: Keep 40%
            if is_on_beat and random.random() < 0.5:
                keep_medium.add(tick)
            elif is_downbeat:
                keep_medium.add(tick)
            
            # Easy: Keep 25%
            if is_downbeat or (is_on_beat and random.random() < 0.35):
                keep_easy.add(tick)
        
        # Build new events
        new_events = []
        
        # First pass: Keep track metadata and non-key notes
        for tick, msg in events:
            # Skip existing lower difficulty notes - we'll regenerate
            if msg.type in ('note_on', 'note_off'):
                if 60 <= msg.note <= 64 or 72 <= msg.note <= 76 or 84 <= msg.note <= 88:
                    continue
            new_events.append((tick, msg))
        
        # Second pass: Generate lower difficulties from Expert
        for tick, msg in events:
            if msg.type == 'note_on' and 96 <= msg.note <= 100 and msg.velocity > 0:
                lane = msg.note - 96
                
                # Hard: Keep chords
                if tick in keep_hard:
                    new_events.append((tick, mido.Message('note_on', note=84 + lane, velocity=msg.velocity, time=0)))
                
                # Medium: No chords, keep lowest note only
                if tick in keep_medium:
                    notes_at_tick = expert_notes.get(tick, [])
                    lowest_lane = min(n - 96 for n, _ in notes_at_tick)
                    if lane == lowest_lane:
                        new_events.append((tick, mido.Message('note_on', note=72 + lane, velocity=msg.velocity, time=0)))
                
                # Easy: 3 lanes only, no chords
                if tick in keep_easy:
                    notes_at_tick = expert_notes.get(tick, [])
                    lowest_lane = min(n - 96 for n, _ in notes_at_tick)
                    if lane == lowest_lane:
                        easy_lane = min(lane, 2)
                        new_events.append((tick, mido.Message('note_on', note=60 + easy_lane, velocity=msg.velocity, time=0)))
            
            elif msg.type == 'note_off' and 96 <= msg.note <= 100:
                lane = msg.note - 96
                
                if tick in keep_hard:
                    new_events.append((tick, mido.Message('note_off', note=84 + lane, velocity=0, time=0)))
                if tick in keep_medium:
                    new_events.append((tick, mido.Message('note_off', note=72 + lane, velocity=0, time=0)))
                if tick in keep_easy:
                    new_events.append((tick, mido.Message('note_off', note=60 + min(lane, 2), velocity=0, time=0)))
        
        # Sort and rebuild
        new_events.sort(key=lambda e: (e[0], 0 if hasattr(e[1], 'velocity') and getattr(e[1], 'velocity', 0) > 0 else 1))
        
        new_track = mido.MidiTrack()
        prev_tick = 0
        for tick, msg in new_events:
            if msg.type == 'end_of_track':
                continue
            msg_copy = msg.copy()
            msg_copy.time = tick - prev_tick
            new_track.append(msg_copy)
            prev_tick = tick
        
        new_track.append(mido.MetaMessage('end_of_track', time=0))
        return new_track
    
    def _reduce_existing_keys(self, track: mido.MidiTrack, expert_notes: dict) -> mido.MidiTrack:
        """Reduce existing keys notes if they were duplicated from Expert."""
        import random
        random.seed(42)
        
        events = []
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            events.append((abs_tick, msg))
        
        sorted_times = sorted(expert_notes.keys())
        ticks_per_measure = self.ticks_per_beat * 4
        
        hard_remove = set()
        medium_remove = set()
        easy_remove = set()
        
        for tick in sorted_times:
            beat_pos = (tick % ticks_per_measure) / self.ticks_per_beat
            is_on_beat = beat_pos % 1.0 < 0.15
            is_downbeat = beat_pos < 0.15
            
            # Hard: Keep ~75%
            if not is_on_beat and random.random() > 0.6:
                hard_remove.add(tick)
            
            # Medium: Keep ~45%
            if not is_downbeat and random.random() > 0.45:
                medium_remove.add(tick)
            
            # Easy: Keep ~30%
            if not is_on_beat and random.random() > 0.4:
                easy_remove.add(tick)
        
        filtered = []
        for tick, msg in events:
            if msg.type in ('note_on', 'note_off'):
                note = msg.note
                if 84 <= note <= 88 and tick in hard_remove:
                    continue
                if 72 <= note <= 76 and tick in medium_remove:
                    continue
                if 60 <= note <= 64:
                    if tick in easy_remove:
                        continue
                    # Remap lanes 3-4 to lanes 0-2
                    lane = note - 60
                    if lane > 2:
                        new_note = 60 + (lane % 3)  # Wrap to 3 lanes
                        msg = msg.copy()
                        msg.note = new_note
            filtered.append((tick, msg))
        
        new_track = mido.MidiTrack()
        prev_tick = 0
        for tick, msg in filtered:
            if msg.type == 'end_of_track':
                continue
            msg_copy = msg.copy()
            msg_copy.time = tick - prev_tick
            new_track.append(msg_copy)
            prev_tick = tick
        
        new_track.append(mido.MetaMessage('end_of_track', time=0))
        return new_track
    
    def calculate_difficulty_ratings(self, midi_path: str, duration_sec: float) -> Dict[str, int]:
        """
        Calculate difficulty ratings (0-6 scale) for each instrument.
        
        Based on:
        - Notes per second (NPS) for Expert difficulty
        - Chord complexity
        - Pattern variety
        
        Scale: -1=No part, 0=Warmup, 1-2=Easy, 3-4=Medium, 5=Hard, 6=Impossible
        """
        midi = mido.MidiFile(midi_path)
        ratings = {}
        
        track_stats = {}
        for track in midi.tracks:
            track_name = None
            for msg in track:
                if msg.type == 'track_name':
                    track_name = msg.name
                    break
            
            if not track_name:
                continue
            
            # Count Expert notes
            expert_note_count = 0
            chord_count = 0
            note_times = set()
            
            abs_tick = 0
            for msg in track:
                abs_tick += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    # Expert range varies by instrument
                    if track_name == 'PART DRUMS':
                        is_expert = 95 <= msg.note <= 100
                    elif 'REAL_KEYS' in track_name:
                        # Pro Keys uses actual MIDI pitches 48-72
                        is_expert = 48 <= msg.note <= 72
                    elif track_name == 'PART VOCALS' or track_name.startswith('HARM'):
                        # Vocals use MIDI pitches 36-84 or 0 for pitchless (not 96-100)
                        # Count actual vocal notes, not phrase markers (105)
                        is_expert = (36 <= msg.note <= 84) or msg.note == 0
                    else:
                        is_expert = 96 <= msg.note <= 100
                    
                    if is_expert:
                        if abs_tick in note_times:
                            chord_count += 1
                        note_times.add(abs_tick)
                        expert_note_count += 1
            
            if expert_note_count > 0:
                track_stats[track_name] = {
                    'notes': expert_note_count,
                    'chords': chord_count,
                    'unique_times': len(note_times),
                }
        
        # Calculate ratings
        for track_name, stats in track_stats.items():
            nps = stats['unique_times'] / duration_sec  # Notes per second
            chord_ratio = stats['chords'] / stats['notes'] if stats['notes'] > 0 else 0
            
            # Base rating from NPS (Clone Hero 0–6 scale).
            # Calibrated 2026-05-04 against community-charted Clone Hero
            # songs: typical pop-punk drum chart at 4-6 NPS = 3 (Medium),
            # hard rock at 7-9 NPS = 4 (Hard), prog/metal at 10-12 NPS
            # = 5 (Expert), only constant 16ths at 13+ NPS = 6 (Impossible).
            # Per-instrument adjustment: drums fire faster than guitar in
            # the same song, so they get higher thresholds.
            is_drums = (track_name == 'PART DRUMS')
            if is_drums:
                if nps < 2.5:
                    base = 1
                elif nps < 5.0:
                    base = 2
                elif nps < 7.5:
                    base = 3
                elif nps < 10.0:
                    base = 4
                elif nps < 13.0:
                    base = 5
                else:
                    base = 6
            else:
                if nps < 1.5:
                    base = 1
                elif nps < 3.0:
                    base = 2
                elif nps < 5.0:
                    base = 3
                elif nps < 7.5:
                    base = 4
                elif nps < 10.0:
                    base = 5
                else:
                    base = 6

            # Chord-density bump (only for guitar/bass/keys; drums almost
            # always have kick+other simultaneity which would over-bump).
            if not is_drums and chord_ratio > 0.4:
                base = min(6, base + 1)
            
            # Map track names to song.ini keys
            key_map = {
                'PART DRUMS': 'diff_drums',
                'PART GUITAR': 'diff_guitar',
                'PART BASS': 'diff_bass',
                'PART VOCALS': 'diff_vocals',
                'PART KEYS': 'diff_keys',
                'PART REAL_KEYS_X': 'diff_keys_real',  # Pro Keys
            }
            
            if track_name in key_map:
                ratings[key_map[track_name]] = base
        
        return ratings
    
    def update_song_ini(self, ini_path: str, difficulty_ratings: Dict[str, int]):
        """Update song.ini with calculated difficulty ratings."""
        from pathlib import Path
        
        ini_file = Path(ini_path)
        if not ini_file.exists():
            logger.warning(f"song.ini not found at {ini_path}")
            return
        
        lines = ini_file.read_text(encoding='utf-8').splitlines()
        new_lines = []
        updated_keys = set()
        
        for line in lines:
            # Check if this line is a difficulty setting we should update
            updated = False
            for key, value in difficulty_ratings.items():
                if line.strip().startswith(f'{key} =') or line.strip().startswith(f'{key}='):
                    new_lines.append(f'{key} = {value}')
                    updated_keys.add(key)
                    updated = True
                    break
            
            if not updated:
                new_lines.append(line)
        
        # Add any missing difficulty keys
        for key, value in difficulty_ratings.items():
            if key not in updated_keys:
                # Insert before the last line (usually empty or preview_start_time)
                insert_idx = len(new_lines) - 1
                new_lines.insert(insert_idx, f'{key} = {value}')
        
        ini_file.write_text('\n'.join(new_lines) + '\n', encoding='utf-8')
        logger.info(f"Updated song.ini with difficulty ratings: {difficulty_ratings}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhance Clone Hero/YARG charts")
    parser.add_argument("midi", help="Input MIDI file")
    parser.add_argument("audio", help="Audio file for analysis")
    parser.add_argument("-o", "--output", help="Output MIDI file (default: overwrite input)")
    
    args = parser.parse_args()
    
    output = args.output or args.midi
    
    enhancer = ChartEnhancer()
    enhancer.enhance_chart(args.midi, args.audio, output)


if __name__ == "__main__":
    main()
