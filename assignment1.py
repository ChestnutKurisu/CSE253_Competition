################################################################################
# Task 1: Composer classification (symbolic, multiclass)
################################################################################

# (A) Imports and Global Configuration

import os, re, csv, random, unicodedata, warnings, json, math, time, zipfile, shutil
import urllib.request, pickle, multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import miditoolkit
import pretty_midi

# Symbolic-music toolkits
from miditok import REMI, TokenizerConfig
from symusic import Score
import pypianoroll
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier
from scipy.stats import skew as _scipy_skew
import torch

from datasets import load_dataset
import traceback 

# (B) Constants and Global Singletons

STUDENT_DIR = Path("student_files/task1_composer_classification")
ASAP_DIR    = Path("asap-dataset")
ASAP_META_CSV = ASAP_DIR / "metadata.csv"
COMPOSERS    = ["Bach","Beethoven","Chopin","Haydn","Liszt","Mozart","Schubert","Schumann"]

CACHE_DIR    = Path("feature_cache")
CACHE_DIR.mkdir(exist_ok=True, parents=True)

EXAMPLE_ROWS = 4
TOTAL_FEAT_DIM = 364

# This is the threshold (in "beats") used to split large MIDI files 
# for all datasets EXCEPT the Student dataset.
# If a file's total beats exceed this threshold, it will be partitioned.
PART_BEATS_THRESHOLD = 65.0

# (C) Symbolic-Music Initializations

_TOKENIZER_CFG = TokenizerConfig(
    num_velocities=32,
    use_chords=True,
    use_programs=True,
    use_tempos=True,
    use_time_signatures=True
)
_REMI = REMI(_TOKENIZER_CFG)

_REMI_CACHE_DIR = CACHE_DIR / "remi_tokens"
_REMI_CACHE_DIR.mkdir(exist_ok=True)

_PCA_MODEL_PATH = CACHE_DIR / "pc_pca.pkl"
def _get_pc_pca():
    """Load / create a 2-D PCA model for pitch-class histograms."""
    if _PCA_MODEL_PATH.is_file():
        return pickle.load(open(_PCA_MODEL_PATH, "rb"))
    # Fit on 24 rotated major/minor key profiles – fast and good enough
    base = []
    def _norm(v):
        v = np.asarray(v)
        return v / v.sum()

    maj = _norm([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    minor = _norm([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
    for prof in (maj, minor):
        for s in range(12):
            base.append(np.roll(prof, s))

    pca = PCA(n_components=2, random_state=42).fit(np.stack(base))
    pickle.dump(pca, open(_PCA_MODEL_PATH, "wb"))
    return pca

_PC_PCA = _get_pc_pca()

# (D) Helper Functions

def _canon(name: str) -> str | None:
    """
    Convert any “composer” string (folder name, filename tag, etc.)
    into **one** of the eight target surnames listed in `COMPOSERS`.

    • Treat underscore “_”, hyphen “-”, comma “,” and ordinary spaces as
      equivalent word separators – so folders such as
      “Bach_Johann_Sebastian”, “Ludwig_van_Beethoven” or
      “Wolfgang-Amadeus-Mozart” are handled correctly.

    • Scan every token and return the first one that matches a canonical
      surname.  If no token matches, return `None`.
    """
    if not name:
        return None

    # strip accents → ASCII
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")

    # turn all common separators into a single space
    n = re.sub(r"[,_\-]+", " ", n)

    # look for a recognised surname
    for tok in n.split():
        cand = tok.capitalize()
        if cand in COMPOSERS:
            return cand
    return None

def _log_split(tag: str,
               n_pieces: int,
               n_segments: int) -> None:
    """
    Pretty log line that mirrors the existing “[filter] …” messages.

    Example
    -------
    >>> _log_split("Student", 1210, 1210)
    [split] Student : 1210 pieces → 1210 segments
    """
    print(f"[split] {tag:<8}: {n_pieces:4d} pieces → {n_segments:4d} segments")


def _split_midi_file(midi_path: Path, threshold_beats: float) -> List[Path]:
    """
    Internal helper that splits a MIDI file into multiple parts if its total
    length in "beats" exceeds `threshold_beats`. Each part is saved as a new 
    file in the same directory, with a `"_partX"` suffix. The time range of 
    notes, tempo, signature changes, etc. are all truncated accordingly.

    If the total length in beats does not exceed the threshold, it returns
    a single-element list: [midi_path] (i.e. no changes).

    Otherwise, it returns a list of new Paths (the splitted files).
    """
    try:
        mf = miditoolkit.midi.parser.MidiFile(midi_path)
    except Exception:
        # if it fails to parse, just return the original
        return [midi_path]

    total_beats = mf.max_tick / (mf.ticks_per_beat or 1.0)
    if total_beats <= threshold_beats:
        return [midi_path]

    # We'll create splitted files in the same directory, with a "_partN" suffix
    # E.g. "myfile.mid" -> "myfile_part0.mid", "myfile_part1.mid", ...
    out_paths = []
    n_parts = int(math.ceil(total_beats / threshold_beats))

    # Because the code above does not otherwise rely on "time" except for notes,
    # we do minimal but correct slicing of events: we'll rebuild a MidiFile 
    # for each chunk, with note times offset from chunk start.
    # This is somewhat naive but sufficient for demonstration.

    chunk_size_beats = threshold_beats
    for i in range(n_parts):
        start_beat = i * chunk_size_beats
        end_beat = min((i+1)*chunk_size_beats, total_beats)
        start_tick = int(start_beat * mf.ticks_per_beat)
        end_tick = int(end_beat * mf.ticks_per_beat)

        # Build new MidiFile with the same header info
        mf_part = miditoolkit.midi.parser.MidiFile()
        mf_part.ticks_per_beat = mf.ticks_per_beat

        # Slice events
        for tempo_evt in mf.tempo_changes:
            if start_tick <= tempo_evt.time < end_tick:
                new_evt = miditoolkit.midi.containers.TempoChange(
                    tempo=tempo_evt.tempo,
                    time=tempo_evt.time - start_tick
                )
                mf_part.tempo_changes.append(new_evt)
        # If no tempo inside, add the earliest that was <= start_tick
        # (so the chunk has a default tempo).
        # But to keep code changes minimal, we omit that detail.

        for ts_evt in mf.time_signature_changes:
            if start_tick <= ts_evt.time < end_tick:
                new_evt = miditoolkit.midi.containers.TimeSignature(
                    ts_evt.numerator,
                    ts_evt.denominator,
                    ts_evt.time - start_tick
                )
                mf_part.time_signature_changes.append(new_evt)

        # For each track
        for inst in mf.instruments:
            new_inst = miditoolkit.Instrument(
                program=inst.program,
                is_drum=inst.is_drum,
                name=inst.name
            )
            # slice notes
            for note in inst.notes:
                if note.end >= start_tick and note.start < end_tick:
                    # clamp
                    s_ = max(note.start, start_tick)
                    e_ = min(note.end, end_tick)
                    # shift
                    s_ -= start_tick
                    e_ -= start_tick
                    new_inst.notes.append(
                        miditoolkit.Note(
                            pitch=note.pitch,
                            velocity=note.velocity,
                            start=s_,
                            end=e_
                        )
                    )
            mf_part.instruments.append(new_inst)

        # Recompute max_tick from new content
        mf_part.max_tick = end_tick - start_tick

        part_stem = midi_path.stem + f"_part{i}"
        part_file = midi_path.parent / (part_stem + midi_path.suffix)
        mf_part.dump(part_file)
        out_paths.append(part_file)

    return out_paths

def gather_student(root: Path) -> dict:
    """
    Parse the train.json from the student directory,
    returning { absolute_path_str → composer_name } for recognized composers.
    (No splitting is done for Student dataset.)
    """
    out = {}
    train_json = root / "train.json"
    if not train_json.is_file():
        warnings.warn(f"Missing train.json in {root}")
        return out
    with train_json.open() as fp:
        data = eval(fp.read())  # { "relative/path.mid": "ComposerName", ... }
    for rel, comp in data.items():
        comp = _canon(comp)
        if comp:
            full_path = (root / rel).resolve()
            out[str(full_path)] = comp
    _log_split("Student", len(out), len(out))
    return out
    
def gather_asap(meta_csv: Path) -> dict:
    """
    Return { absolute_path_str → composer_name } from the ASAP dataset.
    Long scores are split, and we log “pieces → segments”.
    """
    if not meta_csv.is_file():
        warnings.warn(f"ASAP metadata missing at {meta_csv}")
        return {}

    out: dict[str, str] = {}
    pieces = 0          # ① counter

    with meta_csv.open(encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            comp = _canon(row["composer"])
            if not comp:
                continue

            for rel in [row["midi_score"], row["midi_performance"]]:
                if not rel:
                    continue

                full_path = (meta_csv.parent / rel).resolve()
                if not full_path.is_file():
                    continue

                pieces += 1               # ② original file

                # split if needed
                splitted = _split_midi_file(full_path, PART_BEATS_THRESHOLD)

                for sfile in splitted:
                    out[str(sfile)] = comp

    _log_split("ASAP", pieces, len(out))   # ③ one log line
    return out
    
def _summarise_sources(student: dict,
                       asap: dict) -> None:
    """Pretty print dataset stats for sanity-checking."""
    sources = {
        "Student":  student,
        "ASAP":     asap,
    }

    rows = []
    for cmp in COMPOSERS:
        rows.append(
            {"Composer": cmp,
             **{tag: sum(1 for c in src.values() if c == cmp)
                for tag, src in sources.items()}}
        )
    print("\n=== Per-composer counts ===")
    print(pd.DataFrame(rows).set_index("Composer"))

    print("\n=== Overall counts ===")
    print({tag: len(src) for tag, src in sources.items()})

    for tag, mapping in sources.items():
        print(f"\n{tag} examples (total {len(mapping)}):")
        n_samples = min(EXAMPLE_ROWS, len(mapping))
        if n_samples <= 0:
            continue
        for path, comp in random.sample(list(mapping.items()), n_samples):
            ok = "✓" if Path(path).is_file() else "✗"
            print(f"  {ok} {comp:<9} {path}")

def tokenize_remi(midi_path: Path) -> list[int]:
    """
    MidiTok REMI tokenisation with persistent cache 
    so the same file isn't tokenised repeatedly.
    """
    cpath = _REMI_CACHE_DIR / (midi_path.stem + ".remi.pkl")
    if cpath.is_file():
        return pickle.loads(cpath.read_bytes())

    score = Score(str(midi_path))
    tokseq = _REMI(score)  # miditok.TokSequence
    ids = tokseq.ids
    cpath.write_bytes(pickle.dumps(ids))
    return ids

def safe_tokenize_remi(midi_path: Path) -> list[int]:
    """
    Wrapper around `tokenize_remi()` that NEVER lets a tokenisation error
    abort the whole feature-extraction process.

    • On success: returns the usual list[int] of token IDs.
    • On failure: emits *one* `UserWarning` and returns an **empty** list.
    """
    try:
        return tokenize_remi(midi_path)
    except Exception as exc:
        warnings.warn(
            f"REMI tokenisation failed for {midi_path} – "
            f"continuing without REMI-based features ({exc})"
        )
        return []

def chord_transition_probability(chords: list[tuple]) -> float:
    """
    First-order Markov log-probability (per transition) of the given chord
    sequence, rescaled to [0, 1]. A completely novel path → 0.0.
    """
    if len(chords) < 2:
        return 1.0
    states = [(root, qual) for (_, root, qual, _, _) in chords]

    counts = defaultdict(Counter)
    for a, b in zip(states, states[1:]):
        counts[a][b] += 1

    probs = {
        s: {t: c / sum(ct.values()) for t, c in ct.items()}
        for s, ct in counts.items()
    }

    ll = 0.0
    valid = 0
    for a, b in zip(states, states[1:]):
        p = probs.get(a, {}).get(b, 1e-12)
        ll += np.log(p)
        valid += 1
    if valid == 0:
        return 0.0
    return float(np.exp(ll / valid))


def intervals_markov_chain_order2(intervals: list[int]) -> float:
    """
    2-nd-order Markov negative log-likelihood per step, mapped to [0, 1]
    (higher = more predictable).
    """
    if len(intervals) < 3:
        return 1.0
    ctx_counts = defaultdict(Counter)
    for a, b, c in zip(intervals, intervals[1:], intervals[2:]):
        ctx_counts[(a, b)][c] += 1
    ctx_probs = {
        ctx: {nxt: c / sum(cnts.values()) for nxt, c in cnts.items()}
        for ctx, cnts in ctx_counts.items()
    }
    nll = 0.0
    total = 0
    for a, b, c in zip(intervals, intervals[1:], intervals[2:]):
        p = ctx_probs.get((a, b), {}).get(c, 1e-12)
        nll -= np.log(p)
        total += 1
    if total == 0:
        return 1.0
    return float(1 / (1 + nll / total))


def remi_bar_embed(tokens: list[int]) -> float:
    """
    Simple information density measure: 
    average # of tokens per bar, normalized by total tokens.
    """
    if not tokens:
        return 0.0
    bar_id = _REMI["Bar_None"]
    bars = tokens.count(bar_id)
    return float(bars / len(tokens))


def piano_roll_cnn_embed(pm: "pretty_midi.PrettyMIDI") -> float:
    """
    Extremely light “CNN” embedding, now independent of *pypianoroll*.

    1.  For every note in the PrettyMIDI object, accumulate
        ``velocity × duration`` energy into its pitch bin.
    2.  Normalise the 128-D energy vector so it sums to 1.
    3.  Project it with an 8-component PCA that is lazily trained once
        on the 128-D identity matrix (deterministic, ≈1 ms).
    4.  Return the first principal component as a single scalar feature.

    This avoids all tempo / beat parsing, so the “divide by zero” problem
    inside *pypianoroll.inputs* can’t occur.
    """
    # ------------------------------------------------------------------ 1–2
    energy = np.zeros(128, dtype=np.float32)
    for inst in pm.instruments:
        for note in inst.notes:
            duration = note.end - note.start
            energy[note.pitch] += note.velocity * duration

    total = float(energy.sum())
    if total > 0:
        energy /= total     # normalise so Σ = 1

    # ------------------------------------------------------------------ 3
    if not hasattr(piano_roll_cnn_embed, "_pca"):
        piano_roll_cnn_embed._pca = PCA(n_components=8, random_state=42) \
            .fit(np.eye(128, dtype=np.float32))

    # ------------------------------------------------------------------ 4
    comp1 = piano_roll_cnn_embed._pca.transform(energy.reshape(1, -1))[0, 0]
    return float(comp1)

# (E) Feature-Extraction (268-D Vector)

def _safe_stats(vals: List[float]) -> tuple:
    """Helper to compute (mean, std, min, max, median) robustly."""
    if not vals:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    arr = np.array(vals, dtype=np.float32)
    return (
        float(arr.mean()),
        float(arr.std()),
        float(arr.min()),
        float(arr.max()),
        float(np.median(arr))
    )

def _entropy(hist: np.ndarray) -> float:
    """Helper for Shannon entropy of a 1D histogram."""
    p = hist / (hist.sum() + 1e-9)
    return float(-np.sum(p * np.log2(p + 1e-9)))

def _pairwise_shannon_entropy(pairs: np.ndarray) -> float:
    """2D matrix version for transitions."""
    p = pairs / (pairs.sum() + 1e-9)
    return float(-np.sum(p * np.log2(p + 1e-9)))

def _kurtosis(vals: List[float]) -> float:
    """Naive kurtosis helper; or 0.0 if fewer than 2 vals."""
    if len(vals) < 2:
        return 0.0
    arr = np.array(vals, dtype=np.float32)
    mean_ = arr.mean()
    std_  = arr.std()
    if std_ < 1e-9:
        return 0.0
    z = (arr - mean_) / std_
    return float(np.mean(z**4) - 3.0)

def _skewness(vals: List[float]) -> float:
    """Naive skewness helper."""
    if len(vals) < 2:
        return 0.0
    arr = np.array(vals, dtype=np.float32)
    mean_ = arr.mean()
    std_  = arr.std()
    if std_ < 1e-9:
        return 0.0
    z = (arr - mean_) / std_
    return float(np.mean(z**3))

def _autocorr_coeff(values, lag=1):
    """Simple autocorrelation with a given lag."""
    if len(values) <= lag:
        return 0.0
    x = np.array(values, dtype=np.float32)
    x_mean = x.mean()
    x_var  = np.var(x)
    if abs(x_var) < 1e-9:
        return 0.0
    return float(np.sum((x[:-lag] - x_mean)*(x[lag:] - x_mean))/((len(x)-lag)*x_var))

def note_bar_index(note_start: float, ticks_per_beat: float, ts_num: int) -> int:
    """For counting which bar a note belongs to."""
    beats = note_start / ticks_per_beat
    return int(beats // ts_num)

def naive_chord_detection(notes, ticks_per_beat):
    """
    Very naive chord detection: group notes that start ~simultaneously,
    define chord by pitch-class set, guess root and major/minor if triad.
    Returns list of: (start_beat, root_pc, quality, chord_dur, set-of-PCs).
    """
    if not notes:
        return []
    sorted_notes = sorted(notes, key=lambda n: n.start)
    chords = []
    chord_group = []
    eps = ticks_per_beat * 0.5  # half a beat
    chord_start_beat = sorted_notes[0].start / ticks_per_beat

    # ---------- FIXED helper ----------
    def chord_quality(pcset: set):
        """Check whether pcset contains a major or minor triad."""
        arr = sorted(pcset)
        triads = []
        n_ = len(arr)
        for i_ in range(n_):
            for j_ in range(i_ + 1, n_):
                for k_ in range(j_ + 1, n_):
                    ival1 = (arr[j_] - arr[i_]) % 12
                    ival2 = (arr[k_] - arr[j_]) % 12
                    triads.append((ival1, ival2))        # ← corrected variable
        for iv1, iv2 in triads:
            if iv1 == 4 and iv2 == 3:
                return "maj"
            if iv1 == 3 and iv2 == 4:
                return "min"
        return "other"
    # -----------------------------------

    for n in sorted_notes:
        n_beat = n.start / ticks_per_beat
        if not chord_group:
            chord_group.append(n)
            chord_start_beat = n_beat
            continue
        if (n.start - chord_group[0].start) > eps:
            # finalise current group
            chord_pcs = {nn.pitch % 12 for nn in chord_group}
            min_start = min(nn.start for nn in chord_group)
            max_end   = max(nn.end   for nn in chord_group)
            chord_dur = (max_end - min_start) / ticks_per_beat
            root_pc   = min(chord_pcs) if chord_pcs else 0
            quality   = chord_quality(chord_pcs)
            chords.append((chord_start_beat, root_pc, quality, chord_dur, chord_pcs))
            chord_group = [n]
            chord_start_beat = n_beat
        else:
            chord_group.append(n)

    # finalise last group
    if chord_group:
        chord_pcs = {nn.pitch % 12 for nn in chord_group}
        min_start = min(nn.start for nn in chord_group)
        max_end   = max(nn.end   for nn in chord_group)
        chord_dur = (max_end - min_start) / ticks_per_beat
        root_pc   = min(chord_pcs) if chord_pcs else 0
        quality   = chord_quality(chord_pcs)
        chords.append((chord_start_beat, root_pc, quality, chord_dur, chord_pcs))

    return chords

def extract_features(midi_path: Path) -> np.ndarray:
    """
    Extracts a 364-dimensional feature vector from a MIDI file.
    Uses miditoolkit and pretty_midi, plus custom statistics.
    """
    cache_file = CACHE_DIR / (midi_path.name + ".pkl")

    def empty_vec():
        return np.zeros(TOTAL_FEAT_DIM, dtype=np.float32)

    # Load MIDI
    try:
        mf = miditoolkit.midi.parser.MidiFile(midi_path)
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception:
        vec = empty_vec()
        pickle.dump(vec, open(cache_file, "wb"))
        return vec

    notes = [n for inst in mf.instruments for n in inst.notes]
    if not notes:
        vec = empty_vec()
        pickle.dump(vec, open(cache_file, "wb"))
        return vec

    # REMI tokens (for a few features)
    remi_tokens = safe_tokenize_remi(midi_path)

    # Basic stats
    pitches     = [n.pitch    for n in notes]
    durations   = [(n.end - n.start) / mf.ticks_per_beat for n in notes]
    velocities  = [n.velocity for n in notes]
    onsets      = [n.start    / mf.ticks_per_beat for n in notes]
    ioi         = np.diff(sorted(onsets)).tolist()

    pitch_stats = _safe_stats(pitches)   # mean, std, min, max, median
    dur_stats   = _safe_stats(durations)
    vel_stats   = _safe_stats(velocities)
    ioi_stats   = _safe_stats(ioi)

    pitch_range     = max(pitches) - min(pitches)
    unique_pitches  = len(set(pitches))
    pc_hist = np.bincount(np.array(pitches) % 12, minlength=12).astype(np.float32)
    pc_sum  = pc_hist.sum()
    if pc_sum > 0:
        pc_hist /= pc_sum

    pitch_entropy   = _entropy(pc_hist)
    pitch_variance  = float(np.var(pitches))

    dur_q     = (np.array(durations) * 16).astype(int)
    vel_bin   = np.array(velocities)
    dur_entropy = _entropy(np.bincount(dur_q,   minlength=128))
    vel_entropy = _entropy(np.bincount(vel_bin, minlength=128))

    # Polyphony measure
    import collections
    active = collections.Counter()
    max_poly = 0
    for n in sorted(notes, key=lambda x: x.start):
        for k in list(active):
            if active[k] <= n.start:
                del active[k]
        active[id(n)] = n.end
        max_poly = max(max_poly, len(active))
    chord_rate = max_poly / len(notes)

    # Tempo, time-signature, etc.
    tempos = [t.tempo for t in mf.tempo_changes] or [120]
    tempo_stats = _safe_stats(tempos)[:2] + (len(tempos),)
    tsig = (mf.time_signature_changes or [miditoolkit.TimeSignature(4, 4, 0)])[0]
    ts_num, ts_den = tsig.numerator, tsig.denominator
    song_beats = mf.max_tick / (mf.ticks_per_beat or 1)
    bars_count = song_beats / ts_num
    notes_per_bar = len(notes) / max(bars_count, 1)
    total_time = mf.max_tick
    notes_time = sum(n.end - n.start for n in notes)
    rest_prop  = 1.0 - notes_time / (total_time + 1e-9) if total_time else 0.0

    # Basic key detection
    major_profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    minor_profile = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
    def _safe_corr(a: np.ndarray, b: np.ndarray, eps=1e-8) -> float:
        if np.std(a) < eps or np.std(b) < eps:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    scores_maj = [_safe_corr(np.roll(major_profile, i), pc_hist) for i in range(12)]
    scores_min = [_safe_corr(np.roll(minor_profile, i), pc_hist) for i in range(12)]
    if max(scores_maj) >= max(scores_min):
        key_pc, key_mode = int(np.argmax(scores_maj)), 1
    else:
        key_pc, key_mode = int(np.argmax(scores_min)), 0

    instr_programs = [inst.program for inst in mf.instruments]
    instr_count = len(set(instr_programs))
    drum_present = int(any(inst.is_drum for inst in mf.instruments))

    sorted_notes   = sorted(notes, key=lambda n: n.start)
    sorted_pitches = [n.pitch for n in sorted_notes]
    intervals      = np.diff(sorted_pitches).tolist()
    int_stats      = _safe_stats(intervals)[:2]
    upward_ratio   = float(sum(i > 0 for i in intervals)) / (len(intervals) or 1)
    downward_ratio = float(sum(i < 0 for i in intervals)) / (len(intervals) or 1)

    dur_set = set(durations)
    len_dur_set = len(dur_set)
    max_dur = max(dur_set) if dur_set else 0.0
    min_dur = min(dur_set) if dur_set else 0.0

    # Prepare single unified list
    features = []

    #------------------------------------------------------------------------------
    # Basic stats  (Features 1 – 59)
    #   – pitch / duration / velocity descriptive statistics
    #   – pitch-class distribution & entropy
    #   – global rhythmic-density, key & instrumentation indicators
    #------------------------------------------------------------------------------
    features.extend([
        # 1-5   pitch mean, std, min, max, median
        pitch_stats[0], pitch_stats[1], pitch_stats[2], pitch_stats[3], pitch_stats[4],
    
        # 6-7   pitch range, number of distinct pitches
        pitch_range, unique_pitches,
    
        # 8-12  duration mean, std, min, max, median
        dur_stats[0], dur_stats[1], dur_stats[2], dur_stats[3], dur_stats[4],
    
        # 13-17 velocity mean, std, min, max, median
        vel_stats[0], vel_stats[1], vel_stats[2], vel_stats[3], vel_stats[4],
    
        # 18-22 IOI  mean, std, min, max, median
        ioi_stats[0], ioi_stats[1], ioi_stats[2], ioi_stats[3], ioi_stats[4],
    
        # 23-34 pitch-class histogram (C … B)
        *pc_hist.tolist(),
    
        # 35-38 pitch entropy, pitch variance, duration entropy, velocity entropy
        pitch_entropy, pitch_variance, dur_entropy, vel_entropy,
    
        # 39-40 chord-note rate, maximum polyphony
        chord_rate, max_poly,
    
        # 41-43 tempo mean, tempo std, number of tempo changes
        tempo_stats[0], tempo_stats[1], tempo_stats[2],
    
        # 44-48 time-signature numerator & denominator, bars, notes/bar, rest proportion
        ts_num, ts_den, bars_count, notes_per_bar, rest_prop,
    
        # 49-52 global key (pitch-class & mode), instrument count, drum present
        float(key_pc), float(key_mode), float(instr_count), float(drum_present),
    
        # 53-56 melodic-interval mean & std, upward vs. downward motion ratios
        int_stats[0], int_stats[1], upward_ratio, downward_ratio,
    
        # 57-59 number of distinct durations, longest duration, shortest duration
        float(len_dur_set), float(max_dur), float(min_dur)
    ])

    #------------------------------------------------------------------------------
    # Extended stats (Features 60–100+)
    #------------------------------------------------------------------------------
    # 60  Weighted avg pitch by velocity
    total_pitch_vel = sum(p * v for p, v in zip(pitches, velocities))
    total_vel = sum(velocities)
    weighted_avg_pitch = total_pitch_vel / (total_vel + 1e-9)
    features.append(weighted_avg_pitch)

    # 61..63 pitch kurt, skew, autocorr
    pitch_kurt = _kurtosis(pitches)
    pitch_skew = _skewness(pitches)
    pitch_autocorr_1 = _autocorr_coeff(pitches, lag=1)
    features.append(pitch_kurt)       # 61
    features.append(pitch_skew)       # 62
    features.append(pitch_autocorr_1) # 63

    # 64  global pitch range density
    global_pitch_range_density = (max(pitches) - min(pitches)) / 127.0
    features.append(global_pitch_range_density)

    # 65  pitch-class transition entropy
    pitch_class_list = [(p % 12) for p in pitches]
    trans_pc = np.zeros((12, 12), dtype=np.float32)
    for i in range(len(pitch_class_list) - 1):
        trans_pc[pitch_class_list[i], pitch_class_list[i + 1]] += 1
    if trans_pc.sum() > 0:
        trans_pc /= trans_pc.sum()
    pitch_class_trans_entropy = _pairwise_shannon_entropy(trans_pc)
    features.append(pitch_class_trans_entropy)

    # 66  chromatic passing ratio
    scale_pc = set()
    if key_mode == 1:
        for interval in [0, 2, 4, 5, 7, 9, 11]:
            scale_pc.add((key_pc + interval) % 12)
    else:
        for interval in [0, 2, 3, 5, 7, 8, 10]:
            scale_pc.add((key_pc + interval) % 12)
    sorted_by_onset = sorted(notes, key=lambda n: n.start)
    sorted_pitch_seq = [n.pitch for n in sorted_by_onset]
    chromatic_pass_count = 0
    for i in range(1, len(sorted_pitch_seq) - 1):
        p_prev = sorted_pitch_seq[i - 1]
        p_curr = sorted_pitch_seq[i]
        p_next = sorted_pitch_seq[i + 1]
        if (p_curr % 12) not in scale_pc:
            midpoint = 0.5 * (p_prev + p_next)
            if abs((p_curr - midpoint)) < 0.51:
                chromatic_pass_count += 1
    ratio_chromatic = chromatic_pass_count / (len(notes) or 1)
    features.append(ratio_chromatic)

    # 67  direction changes
    direction_changes = 0
    prev_diff = 0
    sp = sorted_pitches
    for i in range(1, len(sp)):
        diff = sp[i] - sp[i - 1]
        if diff * prev_diff < 0:
            direction_changes += 1
        prev_diff = diff
    features.append(direction_changes)

    # 68  longest stepwise up-run
    longest_run = 0
    current_run = 0
    for i in range(1, len(sp)):
        if sp[i] - sp[i - 1] == 1:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0
    features.append(longest_run)

    # 69  distinct intervals
    distinct_intervals = len(set(intervals))
    features.append(distinct_intervals)

    # 70  swing ratio
    def compute_swing_ratio(_notes, tpb):
        if not _notes:
            return 1.0
        beat_groups = collections.defaultdict(list)
        for nn in _notes:
            beat_idx = int(nn.start // tpb)
            beat_groups[beat_idx].append(nn)
        swing_vals = []
        for group in beat_groups.values():
            group_sorted = sorted(group, key=lambda x: x.start)
            if len(group_sorted) >= 2:
                n1, n2 = group_sorted[0], group_sorted[1]
                d1 = (n1.end - n1.start)
                d2 = (n2.end - n2.start)
                if d2 > 0:
                    swing_vals.append(d1 / d2)
        if not swing_vals:
            return 1.0
        return float(np.mean(swing_vals))
    swing_ratio = compute_swing_ratio(notes, mf.ticks_per_beat)
    features.append(swing_ratio)

    # 71  note density per quarter
    total_quarters = mf.max_tick / mf.ticks_per_beat
    density_qn = len(notes) / (total_quarters or 1)
    features.append(density_qn)

    # 72  beat-level ioi std
    beat_onsets = []
    for n in notes:
        if (n.start % mf.ticks_per_beat) < 1e-5:
            beat_onsets.append(n.start)
    ioi_beat = np.diff(sorted(beat_onsets))
    ioi_beat_std = float(np.std(ioi_beat)) if len(ioi_beat) > 1 else 0.0
    features.append(ioi_beat_std)

    # 73  nPVI
    if len(durations) < 2:
        npvi = 0.0
    else:
        npvi_sum = 0.0
        for k in range(len(durations) - 1):
            d1, d2 = durations[k], durations[k + 1]
            denom  = ((d1 + d2) / 2.0) + 1e-9
            npvi_sum += abs(d2 - d1) / denom
        npvi = 100.0 * (npvi_sum / (len(durations) - 1))
    features.append(npvi)

    # 74  offbeat ratio
    offbeat_count = sum(1 for n in notes if (n.start % mf.ticks_per_beat) != 0)
    offbeat_ratio = offbeat_count / len(notes)
    features.append(offbeat_ratio)

    # 75  rhythmic histogram entropy
    subdiv_hist = collections.Counter()
    subdiv_size = mf.ticks_per_beat / 4.0
    for n in notes:
        subdiv_idx = int(n.start // subdiv_size)
        subdiv_hist[subdiv_idx] += 1
    hist_vals = np.array(list(subdiv_hist.values()), dtype=np.float32)
    s = hist_vals.sum()
    if s <= 1e-9:
        rhythmic_hist_entropy = 0.0
    else:
        p = hist_vals / s
        rhythmic_hist_entropy = float(-np.sum(p * np.log2(p)))
    features.append(rhythmic_hist_entropy)

    # 76  most common subdiv
    if subdiv_hist:
        most_common_subdiv = float(subdiv_hist.most_common(1)[0][0])
    else:
        most_common_subdiv = 0.0
    features.append(most_common_subdiv)

    # 77  downbeat emphasis
    downbeat_vels = []
    for n in notes:
        beat_in_bar = int((n.start // mf.ticks_per_beat) % ts_num)
        if beat_in_bar == 0:
            downbeat_vels.append(n.velocity)
    if downbeat_vels:
        avg_downbeat_vel = float(np.mean(downbeat_vels))
        avg_vel = float(np.mean(velocities))
        downbeat_emph = avg_downbeat_vel / (avg_vel + 1e-9)
    else:
        downbeat_emph = 1.0
    features.append(downbeat_emph)

    # 78  dead bars
    total_bars = int(np.ceil(bars_count))
    dead_bars = 0
    for bar_i in range(total_bars):
        bar_notes = [
            nn for nn in notes
            if note_bar_index(nn.start, mf.ticks_per_beat, ts_num) == bar_i
        ]
        if len(bar_notes) == 0:
            dead_bars += 1
    features.append(float(dead_bars))

    # 79  syncope
    syncope_count = 0
    for n in notes:
        beat_in_bar = int((n.start // mf.ticks_per_beat) % ts_num)
        if beat_in_bar != 0:
            end_bar = int(((n.end - 1) // mf.ticks_per_beat) % ts_num)
            if end_bar < beat_in_bar:
                syncope_count += 1
    syncope_index = syncope_count / (len(notes) or 1)
    features.append(syncope_index)

    # 80  velocity range
    vel_range = max(velocities) - min(velocities)
    features.append(float(vel_range))

    # 81..83 velocity kurt, skew, autocorr
    vel_kurt = _kurtosis(velocities)
    vel_skew = _skewness(velocities)
    vel_autocorr_1 = _autocorr_coeff(velocities, lag=1)
    features.append(vel_kurt)       # 81
    features.append(vel_skew)       # 82
    features.append(vel_autocorr_1) # 83

    # 84  velocity change entropy
    vel_changes = []
    for i in range(len(velocities) - 1):
        vel_changes.append(velocities[i + 1] - velocities[i])
    if vel_changes:
        offset = 127
        hist_vc = np.zeros(255, dtype=np.float32)
        for vc in vel_changes:
            idx = int(vc + offset)
            if 0 <= idx < 255:
                hist_vc[idx] += 1
        hist_vc_sum = hist_vc.sum()
        if hist_vc_sum > 0:
            hist_vc /= hist_vc_sum
        vel_change_entropy = _entropy(hist_vc)
    else:
        vel_change_entropy = 0.0
    features.append(vel_change_entropy)

    # 85  percent peak velocity
    peak_notes = sum(1 for v in velocities if v >= 120)
    perc_peak_vel = peak_notes / (len(velocities) or 1)
    features.append(perc_peak_vel)

    # 86  ratio attack vs release (dummy approach)
    mean_attack_vel = float(np.mean(velocities))
    mean_release_vel = float(np.mean(velocities))
    ratio_attack_release = mean_attack_vel / (mean_release_vel + 1e-9)
    features.append(ratio_attack_release)

    # 87  RMS velocity
    vel_array = np.array(velocities, dtype=np.float32) / 127.0
    rms_vel = float(np.sqrt(np.mean(vel_array ** 2))) if len(vel_array) else 0.0
    features.append(rms_vel)

    # 88  std of velocity changes
    if len(vel_changes) > 1:
        std_vel_changes = float(np.std(vel_changes))
    else:
        std_vel_changes = 0.0
    features.append(std_vel_changes)

    # 89  distinct velocities
    distinct_vels = len(set(velocities))
    features.append(float(distinct_vels))

    # 90..99 naive chord detection
    chord_list = naive_chord_detection(notes, mf.ticks_per_beat)
    if chord_list:
        roots = [c[1] for c in chord_list]
        mc_root = Counter(roots).most_common(1)[0][0]
        quals = [c[2] for c in chord_list]
        mc_qual = Counter(quals).most_common(1)[0][0]
        qual_map = {"maj": 1, "min": 2, "other": 3}
        mc_qual_id = qual_map.get(mc_qual, 0)
    else:
        mc_root = 0.0
        mc_qual_id = 0
    features.append(float(mc_root))      # 90
    features.append(float(mc_qual_id))   # 91

    if chord_list:
        ccount = Counter((c[1], c[2]) for c in chord_list)
        ctotal = sum(ccount.values())
        if ctotal > 0:
            p_c = np.array(list(ccount.values()), dtype=np.float32) / ctotal
            chord_prog_entropy = float(-np.sum(p_c * np.log2(p_c)))
        else:
            chord_prog_entropy = 0.0
        avg_chord_dur = float(np.mean([c[3] for c in chord_list]))
    else:
        chord_prog_entropy = 0.0
        avg_chord_dur = 0.0
    features.append(chord_prog_entropy)   # 92
    features.append(avg_chord_dur)        # 93

    borrowed_count = 0
    for c in chord_list:
        if (c[1] % 12) not in scale_pc:
            borrowed_count += 1
    freq_borrowed = borrowed_count / (len(chord_list) or 1)
    features.append(freq_borrowed)  # 94

    chord_density = len(chord_list) / (bars_count + 1e-9)
    features.append(chord_density)  # 95

    # dissonance ratio
    total_pairs = 0
    dissonant_pairs = 0
    events_ = []
    for n in sorted_notes:
        events_.append((n.start, +1, n.pitch))
        events_.append((n.end,   -1, n.pitch))
    events_.sort(key=lambda x: (x[0], -x[1]))
    active_notes = []
    for e in events_:
        _, etype, pitch_ = e
        if etype == +1:
            for an in active_notes:
                interval_mod = abs(pitch_ - an) % 12
                if interval_mod in {1,2,6,10,11}:
                    dissonant_pairs += 1
                total_pairs += 1
            active_notes.append(pitch_)
        else:
            if pitch_ in active_notes:
                active_notes.remove(pitch_)
    if total_pairs > 0:
        dissonance_ratio = dissonant_pairs / total_pairs
    else:
        dissonance_ratio = 0.0
    features.append(dissonance_ratio)  # 96

    chord_change_times = [c[0] for c in chord_list]
    chord_change_intervals = np.diff(chord_change_times) if len(chord_change_times) > 1 else []
    if len(chord_change_intervals) >= 2:
        hrvar = float(np.std(chord_change_intervals))
    else:
        hrvar = 0.0
    features.append(hrvar)  # 97

    if chord_list:
        longest_chord = max(c[3] for c in chord_list)
    else:
        longest_chord = 0.0
    features.append(longest_chord)  # 98

    planing_count = 0
    for i in range(len(chord_list)-1):
        root1 = chord_list[i][1]
        root2 = chord_list[i+1][1]
        shift = (root2 - root1) % 12
        if chord_list[i][2] == chord_list[i+1][2] and shift > 0:
            planing_count += 1
    if len(chord_list) > 1:
        planing_ratio = planing_count / (len(chord_list) - 1)
    else:
        planing_ratio = 0.0
    features.append(planing_ratio)  # 99

    #------------------------------------------------------------------------------
    # More extended stats
    #------------------------------------------------------------------------------
    # 100..101 piece length
    piece_len_beats = song_beats
    features.append(piece_len_beats)   # 100
    piece_len_bars = total_bars
    features.append(float(piece_len_bars))  # 101

    # 102 chord-based "section" count
    section_count = 1 if chord_list else 0
    for i in range(1, len(chord_list)):
        if (chord_list[i][1] != chord_list[i - 1][1]) or (chord_list[i][2] != chord_list[i - 1][2]):
            section_count += 1
    features.append(float(section_count))

    # 103 largest onset gap
    if len(onsets) > 1:
        gaps = np.diff(sorted(onsets))
        max_gap = float(np.max(gaps))
    else:
        max_gap = 0.0
    features.append(max_gap)

    # 104 partial last bar
    end_beat_in_bar = (song_beats % ts_num)
    end_bar_partial = 1.0 if abs(end_beat_in_bar) > 1e-3 else 0.0
    features.append(end_bar_partial)

    # 105 repeated melodic snippet
    motif_count = 0
    for i in range(len(intervals) - 3):
        snippet = intervals[i:i+3]
        for j in range(i+1, len(intervals)-2):
            if intervals[j:j+3] == snippet:
                motif_count += 1
                break
    features.append(float(motif_count))

    # 106..109 tempo changes, etc.
    if len(tempos) < 2:
        avg_tempo_change = 0.0
    else:
        diffs = [abs(tempos[i+1] - tempos[i]) for i in range(len(tempos)-1)]
        avg_tempo_change = float(np.mean(diffs))
    features.append(avg_tempo_change)  # 106

    tempo_changes_count = max(0, len(tempos)-1)
    features.append(float(tempo_changes_count))  # 107

    tempo_std = float(np.std(tempos)) if len(tempos) > 1 else 0.0
    features.append(tempo_std)  # 108

    tsig_count = len(mf.time_signature_changes) - 1 if len(mf.time_signature_changes) > 1 else 0
    features.append(float(tsig_count))  # 109

    # 110..111 REMI bigram/unigram entropy
    if remi_tokens:
        remi_bigrams = Counter()
        for i in range(len(remi_tokens)-1):
            remi_bigrams[(remi_tokens[i], remi_tokens[i+1])] += 1
        total_bg = sum(remi_bigrams.values())
        if total_bg > 0:
            p_bg = np.array(list(remi_bigrams.values()), dtype=np.float32)/total_bg
            remi_bigram_entropy = float(-np.sum(p_bg*np.log2(p_bg)))
        else:
            remi_bigram_entropy = 0.0

        remi_unigrams = Counter(remi_tokens)
        total_uni = sum(remi_unigrams.values())
        if total_uni>0:
            p_uni = np.array(list(remi_unigrams.values()), dtype=np.float32)/total_uni
            remi_unigram_entropy = float(-np.sum(p_uni*np.log2(p_uni)))
        else:
            remi_unigram_entropy= 0.0
    else:
        remi_bigram_entropy = 0.0
        remi_unigram_entropy= 0.0
    features.append(remi_bigram_entropy)   # 110
    features.append(remi_unigram_entropy)  # 111

    # 112..117 bar, CNN, chord stats
    remi_bar_embed_val = remi_bar_embed(remi_tokens)
    piano_roll_cnn_val = piano_roll_cnn_embed(pm)
    pc_pca = _PC_PCA.transform(pc_hist.reshape(1, -1))[0]
    pc_pca_1, pc_pca_2 = pc_pca[0], pc_pca[1]
    chord_hmm_prob = chord_transition_probability(chord_list)
    mc_order2_prob = intervals_markov_chain_order2(intervals)
    features.append(remi_bar_embed_val)       # 112
    features.append(piano_roll_cnn_val)       # 113
    features.append(pc_pca_1)                 # 114
    features.append(pc_pca_2)                 # 115
    features.append(chord_hmm_prob)           # 116
    features.append(mc_order2_prob)           # 117

    # 118 pitch median absolute deviation
    pitch_median = np.median(pitches)
    mad_pitches  = np.median([abs(p - pitch_median) for p in pitches])
    features.append(mad_pitches)

    # 119..122 pitch-class PCA stats
    pc_idxs = np.arange(12)
    pc_centroid = float((pc_idxs*pc_hist).sum() / (pc_hist.sum() + 1e-9))
    pc_var = float(((pc_idxs - pc_centroid)**2 * pc_hist).sum())
    pc_sorted = np.argsort(pc_hist)[::-1]
    top2_pcs = pc_sorted[:2] if len(pc_sorted)>=2 else [0,1]
    features.append(pc_centroid)           # 119
    features.append(pc_var)                # 120
    features.append(float(top2_pcs[0]))    # 121
    features.append(float(top2_pcs[1]))    # 122

    # 123 repeated n-grams in pitch seq
    pitch_seq = pitches
    longest_ngram = 0
    max_len = min(len(pitch_seq)//2, 10)
    for L in range(1, max_len+1):
        found_repeat = False
        for i in range(len(pitch_seq)-L):
            snippet = tuple(pitch_seq[i:i+L])
            for j in range(i+1, len(pitch_seq)-L+1):
                if tuple(pitch_seq[j:j+L]) == snippet:
                    found_repeat = True
                    longest_ngram = max(longest_ngram, L)
                    break
            if found_repeat:
                break
    features.append(float(longest_ngram))

    # 124 naive edit distance to repeated pitch
    def naive_edit_dist(a,b):
        dp = np.zeros((len(a)+1, len(b)+1))
        for i_ in range(len(a)+1):
            dp[i_,0] = i_
        for j_ in range(len(b)+1):
            dp[0,j_] = j_
        for i_ in range(1, len(a)+1):
            for j_ in range(1, len(b)+1):
                cost = 0 if a[i_-1]==b[j_-1] else 1
                dp[i_,j_] = min(dp[i_-1,j_]+1, dp[i_,j_-1]+1, dp[i_-1,j_-1]+cost)
        return dp[len(a), len(b)]
    common_pattern = [60]*len(pitch_seq)
    if pitch_seq:
        ed = naive_edit_dist(pitch_seq, common_pattern)
        norm_ed = ed / len(pitch_seq)
    else:
        norm_ed=0.0
    features.append(norm_ed)

    # 125 ioi_cov
    dur_counts = Counter(round(d,2) for d in durations)
    if len(dur_counts)>1:
        distinct_durs = np.array(list(dur_counts.keys()),dtype=np.float32)
        mean_dd = float(distinct_durs.mean())
        std_dd  = float(distinct_durs.std())
        ioi_cov = std_dd/(mean_dd+1e-9)
    else:
        ioi_cov = 0.0
    features.append(ioi_cov)

    # 126 polyrhythm_score placeholder
    polyrhythm_score = 0.0
    features.append(polyrhythm_score)

    # 127 largest pitch leap
    if intervals:
        peak_int_leap = float(max(abs(i) for i in intervals))
    else:
        peak_int_leap = 0.0
    features.append(peak_int_leap)

    # 128 melodic variance difference (4 sections)
    max_onset_ = max(onsets) if onsets else 0.0
    sec_dur    = max_onset_/4.0 if max_onset_>0 else 1.0
    sections   = [[],[],[],[]]
    for n in sorted_notes:
        which_sec = min(int(n.start/(mf.ticks_per_beat*sec_dur)), 3)
        sections[which_sec].append(n.pitch)
    sec_vars=[]
    for s_ in sections:
        if len(s_)>1:
            sec_vars.append(np.var(s_))
        else:
            sec_vars.append(0.0)
    melodic_var_diff = float(max(sec_vars)-min(sec_vars))
    features.append(melodic_var_diff)

    # 129 half-step vs whole-step ratio
    hs_count = sum(1 for i_ in intervals if abs(i_)==1)
    ws_count = sum(1 for i_ in intervals if abs(i_)==2)
    if ws_count>0:
        hs_ws_ratio = hs_count/ws_count
    else:
        hs_ws_ratio = float(hs_count)
    features.append(hs_ws_ratio)

    # 130 repeated notes
    repeated_notes = 0
    for i in range(len(sp)-1):
        if sp[i]==sp[i+1]:
            repeated_notes+=1
    features.append(float(repeated_notes))

    # 131 ornament density (short notes)
    short_notes = sum(1 for d in durations if d<0.25)
    ornament_density = short_notes/(len(notes) or 1)
    features.append(ornament_density)

    # 132 articulation overlap
    total_overlap=0.0
    total_note_len=0.0
    for inst in mf.instruments:
        inst_snotes=sorted(inst.notes,key=lambda n:n.start)
        for i in range(len(inst_snotes)-1):
            ov = inst_snotes[i].end - inst_snotes[i+1].start
            if ov>0:
                total_overlap+=ov
        total_note_len+=sum((n.end-n.start) for n in inst_snotes)
    if total_note_len>0:
        articulation_overlap=total_overlap/total_note_len
    else:
        articulation_overlap=0.0
    features.append(articulation_overlap)

    # 133 tonal changes across 4 segments
    tonal_changes=0
    seg_pitches=[[] for _ in range(4)]
    if onsets:
        for n in sorted_notes:
            which_seg=min(int(n.start/(mf.ticks_per_beat*(max_onset_/4.0 if max_onset_>0 else 1.0))),3)
            seg_pitches[which_seg].append(n.pitch)
    seg_keys=[]
    for segp in seg_pitches:
        if segp:
            pc_h=np.bincount(np.array(segp)%12, minlength=12).astype(np.float32)
            if pc_h.sum()>0:
                pc_h/=pc_h.sum()
            smj=[_safe_corr(np.roll(major_profile,i_),pc_h) for i_ in range(12)]
            smn=[_safe_corr(np.roll(minor_profile,i_),pc_h) for i_ in range(12)]
            if max(smj)>=max(smn):
                seg_keys.append(("maj", np.argmax(smj)))
            else:
                seg_keys.append(("min", np.argmax(smn)))
        else:
            seg_keys.append(("none",0))
    for i_ in range(1,4):
        if seg_keys[i_]!=seg_keys[i_-1]:
            tonal_changes+=1
    features.append(float(tonal_changes))

    # 134 melodic gravity
    mg_count=0
    for i in range(len(sorted_pitches)-1):
        dist1=abs(sorted_pitches[i]-key_pc)
        dist2=abs(sorted_pitches[i+1]-key_pc)
        if dist2<dist1:
            mg_count+=1
    melodic_gravity = mg_count/(len(sorted_pitches)-1 if len(sorted_pitches)>1 else 1)
    features.append(melodic_gravity)

    # 135 neighbor-tone ratio
    neighbor_tone_count = 0
    for note in sorted_notes:
        start_beat = note.start / mf.ticks_per_beat
        chord_idx = None
        for c_i in range(len(chord_list) - 1):
            if chord_list[c_i][0] <= start_beat < chord_list[c_i + 1][0]:
                chord_idx = c_i
                break
        if chord_idx is None and chord_list:
            chord_idx = len(chord_list) - 1
        if chord_idx is not None:
            root_pc = chord_list[chord_idx][1] % 12
            if abs((note.pitch % 12) - root_pc) in (1, 11):
                neighbor_tone_count += 1
    neighbor_tone_ratio = neighbor_tone_count / (len(notes) or 1)
    features.append(neighbor_tone_ratio)

    # 136 pedal duration
    pedal_dur=0.0
    notes_sorted_by_start=sorted(notes,key=lambda n:n.start)
    for i,n1 in enumerate(notes_sorted_by_start):
        dur_1=(n1.end-n1.start)/mf.ticks_per_beat
        start_1=n1.start
        end_1=n1.end
        pedal=False
        for j,n2 in enumerate(notes_sorted_by_start):
            if j==i: continue
            if n2.start<end_1 and n2.end>start_1:
                if n2.pitch!=n1.pitch:
                    pedal=True
                    break
        if pedal and dur_1>pedal_dur:
            pedal_dur=dur_1
    features.append(pedal_dur)

    # 137 melody vs accompaniment rest diff
    melody_inst=None
    max_avg_pitch=-1.0
    for inst in mf.instruments:
        if inst.notes:
            avg_pitch=float(np.mean([n.pitch for n in inst.notes]))
            if avg_pitch>max_avg_pitch:
                max_avg_pitch=avg_pitch
                melody_inst=inst
    if melody_inst is not None:
        melody_total_time=sum(n.end-n.start for n in melody_inst.notes)
        if mf.max_tick:
            melody_rest=(mf.max_tick - melody_total_time)/mf.max_tick
        else:
            melody_rest=0.0
        acc_total_time=0.0
        for inst2 in mf.instruments:
            if inst2 is melody_inst:
                continue
            acc_total_time+=sum(n.end-n.start for n in inst2.notes)
        if mf.max_tick:
            acc_rest=(mf.max_tick-acc_total_time)/mf.max_tick
        else:
            acc_rest=0.0
        rest_diff=float(melody_rest-acc_rest)
    else:
        rest_diff=0.0
    features.append(rest_diff)

    # 138 seventh chords
    seventh_count=0
    for c_ in chord_list:
        root_pc=c_[1]%12
        chord_pc_set=c_[4]
        if c_[2] in ["maj","min"]:
            if ((root_pc+3)%12 in chord_pc_set or (root_pc+4)%12 in chord_pc_set)\
               and ((root_pc+7)%12 in chord_pc_set):
                if ((root_pc+10)%12 in chord_pc_set) or ((root_pc+11)%12 in chord_pc_set):
                    seventh_count+=1
    features.append(float(seventh_count))

    # 139 cadential V->I
    cadential_count=0
    for i in range(len(chord_list)-1):
        croot_i= chord_list[i][1]%12
        croot_ip1= chord_list[i+1][1]%12
        if croot_i==((key_pc+7)%12) and croot_ip1==(key_pc%12):
            cadential_count+=1
    features.append(float(cadential_count))

    # 140 swing_onset_dev
    offsets=[]
    for n in notes:
        ideal_tick= round(n.start/(mf.ticks_per_beat/2.0))*(mf.ticks_per_beat/2.0)
        offsets.append(abs(n.start-ideal_tick))
    if offsets:
        swing_onset_dev=float(np.mean(offsets))/mf.ticks_per_beat
    else:
        swing_onset_dev=0.0
    features.append(swing_onset_dev)

    # 141 arpeggio_length
    arpeggio_length=0.0
    current_arp=1
    for i in range(len(sorted_notes)-1):
        nt1=sorted_notes[i]
        nt2=sorted_notes[i+1]
        start_beat1=nt1.start/mf.ticks_per_beat
        start_beat2=nt2.start/mf.ticks_per_beat
        cidx1=None
        cidx2=None
        for c_i in range(len(chord_list)-1):
            if chord_list[c_i][0]<=start_beat1< chord_list[c_i+1][0]:
                cidx1=c_i
                break
        if cidx1 is None and chord_list:
            cidx1=len(chord_list)-1
        for c_i in range(len(chord_list)-1):
            if chord_list[c_i][0]<=start_beat2< chord_list[c_i+1][0]:
                cidx2=c_i
                break
        if cidx2 is None and chord_list:
            cidx2=len(chord_list)-1
        same_chord=(cidx1==cidx2)
        if same_chord and cidx1 is not None:
            cpc=chord_list[cidx1][4]
            if ((nt1.pitch%12) in cpc) and ((nt2.pitch%12) in cpc) and (nt2.pitch>nt1.pitch):
                current_arp+=1
                arpeggio_length=max(arpeggio_length,current_arp)
            else:
                current_arp=1
        else:
            current_arp=1
    features.append(arpeggio_length)

    # 142 melodic angularity
    if intervals:
        melodic_angularity=float(np.mean([abs(i_) for i_ in intervals]))
    else:
        melodic_angularity=0.0
    features.append(melodic_angularity)

    # 143..147 tempo range block
    tempo_min, tempo_max, tempo_median = _safe_stats(tempos)[2:]
    tempo_range= tempo_max - tempo_min
    tempo_cv= tempo_stats[1]/(tempo_stats[0]+1e-9)
    features.extend([
        tempo_min,    # 143
        tempo_max,    # 144
        tempo_median, # 145
        tempo_range,  # 146
        tempo_cv      # 147
    ])

    # 148..149 poly stats
    events_poly=[]
    for n_ in notes:
        events_poly.append((n_.start, +1))
        events_poly.append((n_.end,   -1))
    events_poly.sort(key=lambda x:x[0])
    cur_poly=0
    poly_counts=[]
    for t_,delta_ in events_poly:
        cur_poly+=delta_
        poly_counts.append(cur_poly)
    avg_poly= float(np.mean(poly_counts)) if poly_counts else 0.0
    poly_std= float(np.std(poly_counts))  if poly_counts else 0.0
    features.append(avg_poly) # 148
    features.append(poly_std) # 149

    # 150..152 velocity autocorr/cv
    vel_autocorr_2 = _autocorr_coeff(velocities,lag=2)
    vel_autocorr_4 = _autocorr_coeff(velocities,lag=4)
    vel_cv         = vel_stats[1]/(vel_stats[0]+1e-9)
    features.extend([vel_autocorr_2, vel_autocorr_4, vel_cv])  # 150..152

    # 153..157 time signature data
    tsig_events= mf.time_signature_changes or [miditoolkit.TimeSignature(4,4,0)]
    ts_nums= [ts.numerator   for ts in tsig_events]
    ts_dens= [ts.denominator for ts in tsig_events]
    dom_ts_num=float(max(ts_nums,key=ts_nums.count))
    dom_ts_den=float(max(ts_dens,key=ts_dens.count))
    ts_change_density= (len(tsig_events)-1)/(piece_len_bars+1e-9) if len(tsig_events)>1 else 0.0
    compound_meter_flag= float((dom_ts_den==8) and (dom_ts_num%3==0))
    irregular_meter_flag= float(dom_ts_num not in (2,3,4,6,8,12))
    features.extend([
        dom_ts_num,          # 153
        dom_ts_den,          # 154
        ts_change_density,   # 155
        compound_meter_flag, # 156
        irregular_meter_flag # 157
    ])

    # 158..162 tempo slope block
    if len(tempos)>=2:
        x_ = np.arange(len(tempos),dtype=np.float32)
        y_ = np.array(tempos,dtype=np.float32)
        tempo_slope = float(((x_-x_.mean())*(y_-y_.mean())).sum()/(((x_-x_.mean())**2).sum()+1e-9))
        tempo_autocorr1=_autocorr_coeff(tempos,lag=1)
        tempo_autocorr2=_autocorr_coeff(tempos,lag=2)
        tempo_skew_    = _skewness(tempos)
    else:
        tempo_slope=0.0
        tempo_autocorr1=0.0
        tempo_autocorr2=0.0
        tempo_skew_=0.0
    tempo_iqr= float(np.percentile(tempos,75)-np.percentile(tempos,25)) if len(tempos)>1 else 0.0
    features.extend([
        tempo_slope,     # 158
        tempo_autocorr1, # 159
        tempo_autocorr2, # 160
        tempo_skew_,     # 161
        tempo_iqr        # 162
    ])

    # 163..174 interval class distribution
    interval_class_counter=np.zeros(12,dtype=np.float32)
    total_dur_consecutive=0.0
    for i_ in range(len(sorted_notes)-1):
        n1=sorted_notes[i_]
        n2=sorted_notes[i_+1]
        iclass= abs(n2.pitch-n1.pitch)%12
        pair_dur= ((n1.end-n1.start)+(n2.end-n2.start))/2.0
        total_dur_consecutive+= pair_dur
        interval_class_counter[iclass]+= pair_dur
    if total_dur_consecutive>0:
        interval_class_counter/= total_dur_consecutive
    for ic in interval_class_counter:
        features.append(ic)  # 163..174 (12 bins)

    # 175..177 Raga fits
    def _norm_pset(pset):
        s__=pset.sum()
        return pset/s__ if s__>0 else pset
    bilawal = _norm_pset(np.array([1,0,0,0,1,0,0,1,0,0,0,0],dtype=float))
    bhairav = _norm_pset(np.array([1,0,0,1,0,0,1,1,0,0,1,0],dtype=float))
    todi    = _norm_pset(np.array([1,0,0,1,0,1,0,1,0,1,0,0],dtype=float))
    def _best_shift_corr(prof,hist):
        corrs=[]
        for shift in range(12):
            rolled=np.roll(prof,shift)
            corrs.append(_safe_corr(rolled,hist))
        return max(corrs)
    bilawal_corr = _best_shift_corr(bilawal,pc_hist)
    bhairav_corr = _best_shift_corr(bhairav,pc_hist)
    todi_corr    = _best_shift_corr(todi,pc_hist)
    features.append(bilawal_corr) # 175
    features.append(bhairav_corr) # 176
    features.append(todi_corr)    # 177

    # 178 pentatonic correlation
    pentatonic_base = np.zeros(12,dtype=float)
    for pc_ in [0,2,4,7,9]:
        pentatonic_base[pc_]=1
    pentatonic_base=_norm_pset(pentatonic_base)
    pent_corr=_best_shift_corr(pentatonic_base,pc_hist)
    features.append(pent_corr)

    # 179 fraction out of key
    if key_mode==1:
        scale_degs={(key_pc+i)%12 for i in [0,2,4,5,7,9,11]}
    else:
        scale_degs={(key_pc+i)%12 for i in [0,2,3,5,7,8,10]}
    out_of_key_notes= sum(1 for p_ in pitches if (p_%12) not in scale_degs)
    frac_out_of_key= out_of_key_notes/(len(pitches) or 1)
    features.append(frac_out_of_key)

    # 180 strong-weak velocity difference
    strong_vels=[]
    weak_vels=[]
    for n in notes:
        beat_mod= (n.start%mf.ticks_per_beat)
        if beat_mod<1e-5:
            strong_vels.append(n.velocity)
        else:
            weak_vels.append(n.velocity)
    avg_strong= float(np.mean(strong_vels)) if strong_vels else 0.0
    avg_weak  = float(np.mean(weak_vels))  if weak_vels  else 0.0
    features.append(avg_strong-avg_weak)

    # 181..183 repeated, stepwise, leaps ratio
    repeated=0
    stepwise_=0
    leaps_=0
    for iv_ in intervals:
        abs_iv= abs(iv_)
        if abs_iv==0:
            repeated+=1
        elif abs_iv<=2:
            stepwise_+=1
        elif abs_iv>=5:
            leaps_+=1
    tot_moves=len(intervals)
    if tot_moves>0:
        features.append(repeated/tot_moves)  # 181
        features.append(stepwise_/tot_moves) # 182
        features.append(leaps_/tot_moves)    # 183
    else:
        features.extend([0.0, 0.0, 0.0])

    # 184..191 duration bins
    dur_bins=np.zeros(8,dtype=np.float32)
    for d_ in durations:
        if d_<0.125:
            dur_bins[0]+=1
        elif d_<0.25:
            dur_bins[1]+=1
        elif d_<0.5:
            dur_bins[2]+=1
        elif d_<1.0:
            dur_bins[3]+=1
        elif d_<2.0:
            dur_bins[4]+=1
        elif d_<4.0:
            dur_bins[5]+=1
        elif d_<8.0:
            dur_bins[6]+=1
        else:
            dur_bins[7]+=1
    dur_bins/=(len(durations) or 1)
    for b_ in dur_bins:
        features.append(b_)

    # 192 energy_val
    energy_val= sum(v*v for v in velocities)/(len(velocities)+1e-9)
    features.append(energy_val)

    # 193 chord_changes_per_bar
    chord_changes=(len(chord_list)-1) if chord_list else 0
    if bars_count>0:
        chord_changes_per_bar= chord_changes/bars_count
    else:
        chord_changes_per_bar=0.0
    features.append(chord_changes_per_bar)

    # 194..196 eighth-based pairs
    eighth_pairs=[]
    for i_ in range(len(sorted_notes)-1):
        st1=sorted_notes[i_].start
        st2=sorted_notes[i_+1].start
        if (int(st1//mf.ticks_per_beat)==int(st2//mf.ticks_per_beat)):
            delta=st2-st1
            eighth_pairs.append(delta)
    if len(eighth_pairs)>1:
        avg_eighth_dur=float(np.mean(eighth_pairs))
        min_eighth_dur=float(np.min(eighth_pairs))
        max_eighth_dur=float(np.max(eighth_pairs))
    else:
        avg_eighth_dur=0.0
        min_eighth_dur=0.0
        max_eighth_dur=0.0
    features.append(avg_eighth_dur) # 194
    features.append(min_eighth_dur) # 195
    features.append(max_eighth_dur) # 196

    # 197..200 bar pitch means
    bar_pitch_map=defaultdict(list)
    for n in notes:
        b_idx= note_bar_index(n.start,mf.ticks_per_beat,ts_num)
        bar_pitch_map[b_idx].append(n.pitch)
    bar_means=[]
    for p_list in bar_pitch_map.values():
        bar_means.append(np.mean(p_list))
    if bar_means:
        mean_bar_mean=float(np.mean(bar_means))
        std_bar_mean =float(np.std(bar_means))
        min_bar_mean =float(np.min(bar_means))
        max_bar_mean =float(np.max(bar_means))
    else:
        mean_bar_mean=std_bar_mean=min_bar_mean=max_bar_mean=0.0
    features.extend([mean_bar_mean,std_bar_mean,min_bar_mean,max_bar_mean])

    # 201 black/white ratio
    black_count= sum(1 for p_ in pitches if (p_%12) in {1,3,6,8,10})
    white_count=len(pitches)-black_count
    if len(pitches)>0:
        black_white_ratio= black_count/(white_count+1e-9)
    else:
        black_white_ratio=0.0
    features.append(black_white_ratio)

    # 202 average empty gap
    bar_filled=[0]*total_bars
    for i_ in range(total_bars):
        bar_notes=[n for n in notes if note_bar_index(n.start,mf.ticks_per_beat,ts_num)==i_]
        bar_filled[i_]=1 if bar_notes else 0
    if len(bar_filled)>1:
        empty_gaps=[]
        current_gap=0
        for val_ in bar_filled:
            if val_==0:
                current_gap+=1
            else:
                if current_gap>0:
                    empty_gaps.append(current_gap)
                current_gap=0
        if current_gap>0:
            empty_gaps.append(current_gap)
        if empty_gaps:
            avg_empty_gap=float(np.mean(empty_gaps))
        else:
            avg_empty_gap=0.0
    else:
        avg_empty_gap=0.0
    features.append(avg_empty_gap)

    # 203..205 fraction top octave, fraction very short, fraction non-chord
    top_octave_notes=sum(1 for p_ in pitches if p_>=108)
    frac_top_octave= top_octave_notes/(len(pitches) or 1)
    features.append(frac_top_octave)

    very_short_notes=sum(1 for d_ in durations if d_<0.0625)
    frac_very_short= very_short_notes/(len(durations) or 1)
    features.append(frac_very_short)

    nct_count=0
    chord_pc_set_all=set()
    for c_ in chord_list:
        chord_pc_set_all|= c_[4]
    for n_ in notes:
        if (n_.pitch%12) not in chord_pc_set_all:
            nct_count+=1
    frac_nct= nct_count/(len(notes) or 1)
    features.append(frac_nct)

    # 206 fraction velocity peaks
    velocity_peaks=0
    for i_ in range(1,len(velocities)-1):
        if velocities[i_]>velocities[i_-1] and velocities[i_]>velocities[i_+1]:
            velocity_peaks+=1
    frac_vel_peaks= velocity_peaks/(len(velocities) or 1)
    features.append(frac_vel_peaks)

    # 207 density fluctuation std
    bar_note_counts=[]
    for i_ in range(total_bars):
        bar_notes=[n for n in notes if note_bar_index(n.start,mf.ticks_per_beat,ts_num)==i_]
        bar_note_counts.append(len(bar_notes))
    if bar_note_counts:
        density_fluct_std=float(np.std(bar_note_counts))
    else:
        density_fluct_std=0.0
    features.append(density_fluct_std)

    # 208..209 fraction chromatic moves, fraction triadic leaps
    chromatic_moves=sum(1 for i_ in intervals if abs(i_)==1)
    frac_chromatic_moves= chromatic_moves/(len(intervals) or 1)
    features.append(frac_chromatic_moves)

    triadic_leaps=sum(1 for i_ in intervals if abs(i_) in (3,4))
    frac_triadic_leaps= triadic_leaps/(len(intervals) or 1)
    features.append(frac_triadic_leaps)

    # 210 chord_in_key_ratio
    chord_in_key_count= sum(1 for c_ in chord_list if (c_[1]%12) in scale_degs)
    chord_in_key_ratio= chord_in_key_count/(len(chord_list) or 1)
    features.append(chord_in_key_ratio)

    # 211 avg chord complexity
    chord_complexities=[]
    for grp in chord_list:
        chord_complexities.append(len(grp[4])-3)
    if chord_complexities:
        avg_chord_complex=float(np.mean(chord_complexities))
    else:
        avg_chord_complex=0.0
    features.append(avg_chord_complex)

    # 212 bar velocity range
    bar_vel_ranges=[]
    for i_ in range(total_bars):
        bvel=[n.velocity for n in notes if note_bar_index(n.start,mf.ticks_per_beat,ts_num)==i_]
        if bvel:
            bar_vel_ranges.append(max(bvel)-min(bvel))
    if bar_vel_ranges:
        avg_bar_vel_range=float(np.mean(bar_vel_ranges))
    else:
        avg_bar_vel_range=0.0
    features.append(avg_bar_vel_range)

    # 213 bar pitch center std
    bar_pitch_centers=[]
    for i_ in range(total_bars):
        bnotes=[n for n in notes if note_bar_index(n.start,mf.ticks_per_beat,ts_num)==i_]
        if bnotes:
            total_w=sum(nb.velocity for nb in bnotes)
            if total_w>0:
                pc_= sum(nb.pitch*nb.velocity for nb in bnotes)/total_w
            else:
                pc_= float(np.mean([nn.pitch for nn in bnotes]))
            bar_pitch_centers.append(pc_)
    if len(bar_pitch_centers)>1:
        pitch_center_std=float(np.std(bar_pitch_centers))
    else:
        pitch_center_std=0.0
    features.append(pitch_center_std)

    # 214 silence ratio
    if sorted_notes:
        total_silence=0.0
        for i_ in range(len(sorted_notes)-1):
            gap_ = sorted_notes[i_+1].start - sorted_notes[i_].end
            if gap_>0:
                total_silence+=gap_
        total_span=(sorted_notes[-1].end - sorted_notes[0].start)
        if total_span>0:
            silence_ratio= total_silence/total_span
        else:
            silence_ratio=0.0
    else:
        silence_ratio=0.0
    features.append(silence_ratio)

    # 215 pitch_vel_corr
    if len(pitches)>1 and len(velocities)==len(pitches):
        pitch_vel_corr= _safe_corr(np.array(pitches,dtype=float),np.array(velocities,dtype=float))
    else:
        pitch_vel_corr=0.0
    features.append(pitch_vel_corr)

    # 216 repeated chord progressions (triples)
    chord_seq2=[(c_[1],c_[2]) for c_ in chord_list]
    repeated_prog_count=0
    seen_seqs=set()
    for i_ in range(len(chord_seq2)-2):
        triple= tuple(chord_seq2[i_:i_+3])
        if triple in seen_seqs:
            repeated_prog_count+=1
        else:
            seen_seqs.add(triple)
    total_triples=max(0,len(chord_seq2)-2)
    if total_triples>0:
        frac_repeated_prog= repeated_prog_count/total_triples
    else:
        frac_repeated_prog=0.0
    features.append(frac_repeated_prog)

    # 217 longest repeated pitch run
    longest_rep=1
    current_rep=1
    for i_ in range(1,len(sorted_pitches)):
        if sorted_pitches[i_]==sorted_pitches[i_-1]:
            current_rep+=1
            if current_rep>longest_rep:
                longest_rep=current_rep
        else:
            current_rep=1
    features.append(float(longest_rep))

    # 218 polyrhythm_flag
    tsig_events2= mf.time_signature_changes or [miditoolkit.TimeSignature(4,4,0)]
    unique_tsigs= {(ts.numerator,ts.denominator) for ts in tsig_events2}
    polyrhythm_flag=1.0 if len(unique_tsigs)>1 else 0.0
    features.append(polyrhythm_flag)

    # 219 harmonium overlap
    harmonium_overlap=0.0
    for inst in mf.instruments:
        if 19<=inst.program<=23:
            inst_notes=sorted(inst.notes,key=lambda x:x.start)
            loc_overlap=0.0
            total_len=0.0
            for i_ in range(len(inst_notes)-1):
                ov= inst_notes[i_].end- inst_notes[i_+1].start
                if ov>0:
                    loc_overlap+=ov
                total_len+=(inst_notes[i_].end- inst_notes[i_].start)
            if total_len>0:
                ratio_= loc_overlap/total_len
                if ratio_>harmonium_overlap:
                    harmonium_overlap=ratio_
    features.append(harmonium_overlap)

    # 220 phrase arcs
    phrase_arcs=0
    phrase_notes=[]
    last_end=None
    for n_ in sorted_notes:
        if last_end is not None and (n_.start - last_end)> mf.ticks_per_beat:
            if len(phrase_notes)>2:
                peak_idx=np.argmax(phrase_notes)
                if (all(phrase_notes[i__]<=phrase_notes[i__+1] for i__ in range(peak_idx))
                    and all(phrase_notes[i__]>=phrase_notes[i__+1] for i__ in range(peak_idx,len(phrase_notes)-1))):
                    phrase_arcs+=1
            phrase_notes=[]
        phrase_notes.append(n_.pitch)
        last_end=n_.end
    if len(phrase_notes)>2:
        peak_idx=np.argmax(phrase_notes)
        if (all(phrase_notes[i__]<=phrase_notes[i__+1] for i__ in range(peak_idx))
            and all(phrase_notes[i__]>=phrase_notes[i__+1] for i__ in range(peak_idx,len(phrase_notes)-1))):
            phrase_arcs+=1
    features.append(float(phrase_arcs))

    # 221 average phrase skip
    phrase_skips=[]
    phrase_notes=[]
    last_end=None
    for n_ in sorted_notes:
        if last_end is not None and (n_.start - last_end)> mf.ticks_per_beat:
            if len(phrase_notes)>1:
                ph_ints=[abs(phrase_notes[k+1]-phrase_notes[k]) for k in range(len(phrase_notes)-1)]
                if ph_ints:
                    phrase_skips.append(max(ph_ints))
            phrase_notes=[]
        phrase_notes.append(n_.pitch)
        last_end=n_.end
    if len(phrase_notes)>1:
        ph_ints=[abs(phrase_notes[k+1]-phrase_notes[k]) for k in range(len(phrase_notes)-1)]
        if ph_ints:
            phrase_skips.append(max(ph_ints))
    avg_ph_skip=float(np.mean(phrase_skips)) if phrase_skips else 0.0
    features.append(avg_ph_skip)

    # 222 tempo_diff_final_init
    if len(tempos)>1:
        tempo_diff_final_init=tempos[-1]-tempos[0]
    else:
        tempo_diff_final_init=0.0
    features.append(tempo_diff_final_init)

    # 223 frac_peak_in_final
    final_threshold= mf.max_tick*0.75
    final_vels=[n.velocity for n in notes if n.start>=final_threshold]
    if final_vels:
        peak_vel_final=max(final_vels)
    else:
        peak_vel_final=0
    overall_peak_vel= max(velocities)
    if overall_peak_vel>0:
        frac_peak_in_final= peak_vel_final/overall_peak_vel
    else:
        frac_peak_in_final=0.0
    features.append(frac_peak_in_final)

    # 224 pedal_overlaps
    event_times=[]
    for inst in mf.instruments:
        for nt in inst.notes:
            event_times.append((nt.start,+1,nt.pitch,inst.program))
            event_times.append((nt.end,  -1,nt.pitch,inst.program))
    event_times.sort(key=lambda x:(x[0],-x[1]))
    active_pitch_insts=defaultdict(set)
    pedal_overlaps=0
    for ev_ in event_times:
        t_,etype,pch,prg=ev_
        if etype==+1:
            active_pitch_insts[pch].add(prg)
            if len(active_pitch_insts[pch])>=3:
                pedal_overlaps+=1
        else:
            if prg in active_pitch_insts[pch]:
                active_pitch_insts[pch].remove(prg)
    features.append(float(pedal_overlaps))

    # 225 relative velocity std
    if velocities:
        max_vel_=max(velocities)
        if max_vel_>0:
            rel_vel_std=np.std(velocities)/max_vel_
        else:
            rel_vel_std=0.0
    else:
        rel_vel_std=0.0
    features.append(float(rel_vel_std))

    # 226 long_rest_count
    long_rest_count=0
    for i_ in range(len(sorted_by_onset)-1):
        gap_beats=(sorted_by_onset[i_+1].start - sorted_by_onset[i_].end)/mf.ticks_per_beat
        if gap_beats>=2*ts_num:
            long_rest_count+=1
    features.append(float(long_rest_count))

    # 227 median_poly
    events_poly2=[]
    for n_ in notes:
        events_poly2.append((n_.start,+1))
        events_poly2.append((n_.end,  -1))
    events_poly2.sort(key=lambda x:x[0])
    cur_poly2=0
    poly_vector=[]
    for t_,delta_ in events_poly2:
        cur_poly2+=delta_
        poly_vector.append(cur_poly2)
    if poly_vector:
        median_poly=float(np.median(poly_vector))
    else:
        median_poly=0.0
    features.append(float(median_poly))

    # 228 fraction extreme leaps
    extreme_leaps= sum(1 for iv_ in intervals if abs(iv_)>12)
    frac_extreme_leaps= extreme_leaps/(len(intervals) or 1)
    features.append(frac_extreme_leaps)

    # 229 fraction leading tone
    if key_mode==1:
        leading=(key_pc+11)%12
    else:
        leading=(key_pc+10)%12
    leading_count= sum(1 for p_ in pitches if (p_%12)==leading)
    frac_leading_tone= leading_count/(len(pitches) or 1)
    features.append(frac_leading_tone)

    # 230..231 repeated vs changed pitch velocities
    repeated_vels=[]
    changed_vels=[]
    for i_ in range(len(sorted_notes)-1):
        p1=sorted_notes[i_].pitch
        p2=sorted_notes[i_+1].pitch
        v2=sorted_notes[i_+1].velocity
        if p1==p2:
            repeated_vels.append(v2)
        else:
            changed_vels.append(v2)
    avg_rep_vel = float(np.mean(repeated_vels)) if repeated_vels else 0.0
    avg_chg_vel = float(np.mean(changed_vels))  if changed_vels  else 0.0
    features.append(avg_rep_vel) # 230
    features.append(avg_chg_vel) # 231

    # 232 sec dominants
    sec_dom_count = 0
    for c in chord_list:
        root = c[1] % 12
        if c[2] == "maj":
            rel = (root - key_pc) % 12
            if rel in {2,4,6,9,11}:
                sec_dom_count += 1
    sec_dom_ratio = sec_dom_count / (len(chord_list) or 1)
    features.append(sec_dom_ratio)

    # 233 dim7
    dim7 = 0
    for c in chord_list:
        if len(c[4]) == 4:
            pc = sorted(c[4])
            iv_set = {(pc[(i+1) % 4] - pc[i]) % 12 for i in range(4)}
            if iv_set == {3}:
                dim7 += 1
    dim7_per_bar = dim7 / (bars_count or 1)
    features.append(dim7_per_bar)

    # 234 aug6
    aug6 = 0
    for c in chord_list:
        pcs = list(c[4])
        if any(((b - a) % 12) == 8 for i, a in enumerate(pcs) for b in pcs[i+1:]):
            aug6 += 1
    aug6_per_bar = aug6 / (bars_count or 1)
    features.append(aug6_per_bar)

    # 235 chromatic mediant
    cm_trans = 0
    for i in range(len(chord_list) - 1):
        r1, q1 = chord_list[i][1] % 12, chord_list[i][2]
        r2, q2 = chord_list[i+1][1] % 12, chord_list[i+1][2]
        if q1 == q2 and q1 in ("maj", "min") and (r2 - r1) % 12 in {4, 8}:
            cm_trans += 1
    chrom_mediant_ratio = cm_trans / max(1, len(chord_list) - 1)
    features.append(chrom_mediant_ratio)

    # 236 chord dur cv
    if chord_list:
        cdurs = np.array([c[3] for c in chord_list], dtype=np.float32)
        chord_dur_cv = float(cdurs.std() / (cdurs.mean() + 1e-9))
    else:
        chord_dur_cv = 0.0
    features.append(chord_dur_cv)

    # 237 chord root entropy
    if chord_list:
        roots = Counter(c[1] % 12 for c in chord_list)
        root_hist = np.array(list(roots.values()), dtype=np.float32)
        chord_root_entropy = _entropy(root_hist)
    else:
        chord_root_entropy = 0.0
    features.append(chord_root_entropy)

    # 238 interval entropy
    if intervals:
        ic_hist = np.bincount(np.abs(np.array(intervals)) % 12, minlength=12).astype(np.float32)
        interval_entropy = _entropy(ic_hist)
    else:
        interval_entropy = 0.0
    features.append(interval_entropy)

    # 239 pitch run density (4+ same direction runs)
    run_cnt, cur_run, cur_dir = 0, 1, 0
    for a, b in zip(sorted_pitches, sorted_pitches[1:]):
        d = 1 if b > a else (-1 if b < a else 0)
        if d and d == cur_dir:
            cur_run += 1
        else:
            if cur_run >= 4:
                run_cnt += 1
            cur_run, cur_dir = 1, d
    if cur_run >= 4:
        run_cnt += 1
    pitch_run_density = run_cnt / (bars_count or 1)
    features.append(pitch_run_density)

    # 240 step_leap_ratio
    steps = sum(abs(i) <= 2 for i in intervals)
    leaps2 = sum(abs(i) >= 5 for i in intervals)
    step_leap_ratio = steps / (leaps2 + 1e-9)
    features.append(step_leap_ratio)

    # 241 leading-tone resolve ratio
    lt_pc = (key_pc + (11 if key_mode else 10)) % 12
    lt_total = 0
    lt_resolved = 0
    for p, n in zip(sorted_pitches, sorted_pitches[1:]):
        if p % 12 == lt_pc:
            lt_total += 1
            if n % 12 == key_pc:
                lt_resolved += 1
    lt_resolve_ratio = lt_resolved / (lt_total or 1)
    features.append(lt_resolve_ratio)

    # 242 left-right energy imbalance
    left_energy  = sum(n.velocity * (n.end-n.start) for n in notes if n.pitch <= 60)
    right_energy = sum(n.velocity * (n.end-n.start) for n in notes if n.pitch  > 60)
    lrei = right_energy / (left_energy + 1e-9)
    features.append(lrei)

    # 243 pitch-duration correlation
    if len(pitches) >= 2:
        pd_corr = _safe_corr(np.array(pitches, dtype=float),
                             np.array(durations, dtype=float))
    else:
        pd_corr = 0.0
    features.append(pd_corr)

    # 244 dotted-rhythm ratio
    dotted_vals = {0.75, 1.5, 3.0, 6.0}
    dotted_cnt  = sum(1 for d in durations if any(abs(d - v) < 0.05 for v in dotted_vals))
    dotted_ratio = dotted_cnt / (len(durations) or 1)
    features.append(dotted_ratio)

    # 245 triplet-subdivision onset ratio
    triplet_onsets = 0
    for n in notes:
        pos = (n.start % mf.ticks_per_beat) / mf.ticks_per_beat
        if abs(pos - 1/3) < 0.02 or abs(pos - 2/3) < 0.02:
            triplet_onsets += 1
    triplet_ratio = triplet_onsets / len(notes)
    features.append(triplet_ratio)

    # 246 average interval-direction run length
    run_len, runs, cur_dir = 1, [], 0
    for a, b in zip(sorted_pitches, sorted_pitches[1:]):
        d = 1 if b > a else (-1 if b < a else 0)
        if d == cur_dir and d != 0:
            run_len += 1
        else:
            if cur_dir != 0:
                runs.append(run_len)
            run_len, cur_dir = 1, d
    if cur_dir != 0:
        runs.append(run_len)
    avg_run_len = float(np.mean(runs)) if runs else 0.0
    features.append(avg_run_len)

    # 247 perfect-fifth leap ratio
    p5_leaps = sum(1 for iv in intervals if abs(iv) == 7)
    p5_ratio = p5_leaps / (len(intervals) or 1)
    features.append(p5_ratio)

    # 248 tonic-pedal sustain ratio
    tonic_pc2 = key_pc
    tonic_ticks = sum((n.end - n.start)
                      for n in notes
                      if (n.pitch % 12) == tonic_pc2
                      and (n.end - n.start) >= mf.ticks_per_beat)
    pedal_ratio = tonic_ticks / (total_time + 1e-9)
    features.append(pedal_ratio)

    # 249 bar-density skewness
    bar_counts = np.array(bar_note_counts, dtype=float)
    if len(bar_counts) > 2:
        from scipy.stats import skew as _scipy_skew
        bar_skew   = float(_scipy_skew(bar_counts))
    else:
        bar_skew = 0.0
    features.append(bar_skew)

    # 250 accent IOI coefficient of variation
    accent_times = sorted(set(beat_onsets))
    accent_ioi   = np.diff(accent_times)
    if len(accent_ioi) >= 2:
        accent_cv = float(np.std(accent_ioi) /
                          (np.mean(accent_ioi) + 1e-9))
    else:
        accent_cv = 0.0
    features.append(accent_cv)

    # 251 opening-motif repetition score
    motif = intervals[:4]
    motif_hits = 0
    for i in range(4, len(intervals)-3):
        if intervals[i:i+4] == motif:
            motif_hits += 1
    motif_score = motif_hits / (bars_count or 1)
    features.append(motif_score)

    # 252..255 chord quality ratio
    qual_counts = Counter(c[2] for c in chord_list)
    total_chords = len(chord_list) or 1
    minor_chord_ratio = qual_counts.get("min", 0)/total_chords
    major_chord_ratio = qual_counts.get("maj", 0)/total_chords
    aug_chord_ratio   = qual_counts.get("aug", 0)/total_chords
    dim_chord_ratio   = qual_counts.get("dim", 0)/total_chords
    features.extend([minor_chord_ratio, major_chord_ratio,
                     aug_chord_ratio,   dim_chord_ratio])

    # 256 chord quality entropy
    if chord_list:
        chord_quality_entropy = _entropy(np.fromiter(qual_counts.values(), dtype=float))
    else:
        chord_quality_entropy = 0.0
    features.append(chord_quality_entropy)

    # 257 chord inversion ratio
    inv_count = sum(1 for c in chord_list if min(c[4]) != c[1] % 12)
    inversion_ratio = inv_count / total_chords
    features.append(inversion_ratio)

    # 258..261 root steps
    root_steps = np.array([(chord_list[i+1][1] - chord_list[i][1]) % 12
                           for i in range(len(chord_list)-1)], dtype=float)
    if root_steps.size:
        mean_root_step   = float(np.mean(np.abs(root_steps)))
        std_root_step    = float(np.std (np.abs(root_steps)))
        fifth_motion_rt  = float(np.count_nonzero(np.isin(root_steps, (5,7))) / root_steps.size)
        third_motion_rt  = float(np.count_nonzero(np.isin(root_steps, (3,4,8,9))) / root_steps.size)
    else:
        mean_root_step = std_root_step = fifth_motion_rt = third_motion_rt = 0.0
    features.extend([mean_root_step, std_root_step, fifth_motion_rt, third_motion_rt])

    # 262 harmonic rhythm cv
    if len(chord_list) > 1:
        hr_int = np.diff([c[0] for c in chord_list])
        harmonic_rhythm_cv = float(np.std(hr_int)/(np.mean(hr_int)+1e-9))
    else:
        harmonic_rhythm_cv = 0.0
    features.append(harmonic_rhythm_cv)

    # 263 tonic-dom ratio
    dom_pc = (key_pc + 7) % 12
    ton_cnt = sum(p % 12 == key_pc for p in pitches)
    dom_cnt = sum(p % 12 == dom_pc   for p in pitches)
    tonic_dom_ratio = ton_cnt / (dom_cnt + 1e-9)
    features.append(tonic_dom_ratio)

    # 264 accented non-chord tones ratio
    strong_beats = [n for n in notes if (n.start % mf.ticks_per_beat) < 1e-5]
    strong_total = len(strong_beats) or 1
    strong_nonchrd = 0
    for n in strong_beats:
        # if onset is near chord boundary, treat as NCT if pitch not matched
        if all((n.pitch % 12) not in c[4] or abs(n.start/mf.ticks_per_beat - c[0])>0.5
               for c in chord_list):
            strong_nonchrd += 1
    acc_nct_ratio = strong_nonchrd / strong_total
    features.append(acc_nct_ratio)

    # 265 instrument crossings
    med_pitches = [(np.median([nn.pitch for nn in inst.notes]), i) for i,inst in enumerate(mf.instruments) if inst.notes]
    med_pitches.sort()
    crossings = 0
    for i in range(len(med_pitches)-1):
        if med_pitches[i][0] > med_pitches[i+1][0]:
            crossings += 1
    features.append(float(crossings))

    # 266..268 region span, duration gini, pitch gini
    def _gini(x):
        if len(x) == 0 or np.allclose(x, 0):
            return 0.0
        x = np.sort(np.array(x, dtype=float))
        n = x.size
        return float((n + 1 - 2 * (np.cumsum(x).sum() / x.sum())) / n)
    reg_span = float(np.percentile(pitches, 95) - np.percentile(pitches, 5)) if pitches else 0.0
    duration_gini = _gini(durations)
    pitch_gini    = _gini([cnt for cnt in Counter(pitches).values()])
    features.extend([reg_span, duration_gini, pitch_gini])

    # 269..271 coarse segmentation (density, skew, uniform)
    if onsets:
        tcm = float(np.mean(onsets)/(song_beats+1e-9))
    else:
        tcm = 0.0
    seg_bins = 8
    seg_len  = song_beats/seg_bins if song_beats else 1.0
    seg_counts = np.bincount(
        np.minimum((np.array(onsets)/seg_len).astype(int), seg_bins-1),
        minlength=seg_bins
    )
    density_skew = _skewness(seg_counts)
    if durations:
        most_common_dur = Counter(np.round(durations,3)).most_common(1)[0][1]
        rhythmic_uniform = most_common_dur/len(durations)
    else:
        rhythmic_uniform = 0.0
    features.extend([tcm, density_skew, rhythmic_uniform])

    # 272 poly switch rate
    poly_switches = 0
    # naive approach: not fully meaningful, but placeholding
    # (the code referencing 'a,b in zip(np.diff([...])...)' is minimal here)
    # We'll do a simpler count:
    # whenever poly_counts changes sign? This is approximate:
    for i in range(len(poly_counts)-1):
        if poly_counts[i] != poly_counts[i+1]:
            poly_switches += 1
    poly_switch_rate = poly_switches/(song_beats+1e-9)
    features.append(poly_switch_rate)

    # 273 sign entropy in intervals
    signs = [np.sign(iv) for iv in intervals if iv]
    if signs:
        sign_counts = np.bincount(np.array(signs)+1, minlength=3).astype(float)
        sign_entropy = _entropy(sign_counts)
    else:
        sign_entropy = 0.0
    features.append(sign_entropy)

    # 274 hand alt rate
    hand_sw = sum(1 for a, b in zip(sorted_pitches, sorted_pitches[1:]) if (a < 60) != (b < 60))
    hand_alt_rate = hand_sw / (len(sorted_pitches)-1 or 1)
    features.append(hand_alt_rate)

    # 275..276 trill ratio, p4 ratio
    trill_cnt = 0
    for i in range(len(sorted_notes)-2):
        if abs(sorted_notes[i].pitch - sorted_notes[i+1].pitch) <= 2 and \
           (sorted_notes[i+1].end - sorted_notes[i+1].start)/mf.ticks_per_beat < 0.25:
            trill_cnt += 1
    trill_ratio = trill_cnt/(len(notes) or 1)
    p4_ratio    = sum(abs(iv)==5 for iv in intervals)/(len(intervals) or 1)
    features.extend([trill_ratio, p4_ratio])

    # 277..281 scale_deg_var, chord complexity ratio, bar_var_cv, vel_rng_cv, chord_dens_sd
    scale_deg_var = float(np.var([(p - key_pc) % 12 for p in pitches])) if pitches else 0.0
    complex_ratio = sum(len(c[4])>3 for c in chord_list)/(len(chord_list) or 1)
    bar_pitch_ranges = []
    for i in range(total_bars):
        bp = [n.pitch for n in notes if note_bar_index(n.start,mf.ticks_per_beat,ts_num)==i]
        if bp:
            bar_pitch_ranges.append(max(bp)-min(bp))
    if bar_pitch_ranges:
        bar_var_cv = float(np.std(bar_pitch_ranges)/(np.mean(bar_pitch_ranges)+1e-9))
    else:
        bar_var_cv=0.0
    if bar_vel_ranges:
        vel_rng_cv= float(np.std(bar_vel_ranges)/(np.mean(bar_vel_ranges)+1e-9))
    else:
        vel_rng_cv=0.0
    chord_dens_sd = float(np.std(chord_change_times)) if len(chord_change_times)>1 else 0.0
    features.extend([scale_deg_var, complex_ratio, bar_var_cv, vel_rng_cv, chord_dens_sd])

    # 282..283 key shift mean/std
    seg_key_pcs = [k[1] if k[0]!="none" else None for k in seg_keys]
    key_steps = [abs((b - a) % 12) for a,b in zip(seg_key_pcs, seg_key_pcs[1:]) if a is not None and b is not None]
    if key_steps:
        key_shift_mean = float(np.mean(key_steps))
        key_shift_std  = float(np.std(key_steps))
    else:
        key_shift_mean = 0.0
        key_shift_std  = 0.0
    features.append(key_shift_mean)
    features.append(key_shift_std)

    # 284 vertical interval entropy
    vert_ints = []
    active_ = []
    for n in sorted(notes, key=lambda x: (x.start, x.pitch)):
        active_ = [p for p in active_ if p.end > n.start + 1]
        for p in active_:
            vert_ints.append(abs(n.pitch - p.pitch) % 12)
        active_.append(n)
    if vert_ints:                   
        vert_hist = np.bincount(
            np.array(vert_ints, dtype=np.int32), minlength=12
        ).astype(np.float32)
        vertical_int_entropy = _entropy(vert_hist)
    else:
        vertical_int_entropy = 0.0
    features.append(vertical_int_entropy)

    # 285 direction-change ratio
    dir_change_ratio = direction_changes / (len(intervals) or 1)
    features.append(dir_change_ratio)

    # 286 bar density quartile-coeff-of-disp
    if bar_note_counts:
        q1, q3 = np.percentile(bar_note_counts, [25,75])
        bar_density_qcd = (q3-q1)/(q3 + q1 + 1e-9)
    else:
        bar_density_qcd = 0.0
    features.append(bar_density_qcd)

    # 287 arpeggio_density (small step runs)
    arp_runs = 0
    run = 0
    for iv in intervals:
        if iv in (1,2):
            run += 1
            if run == 3:
                arp_runs += 1
        else:
            run = 0
    arpeggio_density = arp_runs/(bars_count or 1)
    features.append(arpeggio_density)

    # 288..289 high/low region density
    high_reg_density = sum(p >= 84 for p in pitches)/(len(pitches) or 1)
    low_reg_density  = sum(p <= 48 for p in pitches)/(len(pitches) or 1)
    features.extend([high_reg_density, low_reg_density])

    # 290..291 velocity iqr, duration skew
    def _iqr(vals):
        if not vals:
            return 0.0
        q1_, q3_ = np.percentile(vals,[25,75])
        return float(q3_-q1_)
    velocity_iqr = _iqr(velocities)
    duration_skew = _skewness(durations)
    features.extend([velocity_iqr, duration_skew])

    # 292 ioi_autocorr1
    ioi_autocorr1 = _autocorr_coeff(ioi, lag=1) if len(ioi)>1 else 0.0
    features.append(ioi_autocorr1)

    # 293 small root step ratio
    root_small_steps = [1 for s in root_steps if abs(s) <= 2]
    small_root_step_ratio = sum(root_small_steps)/(len(root_steps) or 1)
    features.append(small_root_step_ratio)

    # 294 average sync velocity (16th off-beats)
    sync_16th = [n.velocity for n in notes if abs(((n.start/mf.ticks_per_beat)%1)-0.5)<0.125]
    avg_sync_vel = float(np.mean(sync_16th)) if sync_16th else 0.0
    features.append(avg_sync_vel)

    # 295..296 bar pitch range stats
    if bar_pitch_ranges:
        bar_pitch_range_mean = float(np.mean(bar_pitch_ranges))
        bar_pitch_range_std  = float(np.std(bar_pitch_ranges))
    else:
        bar_pitch_range_mean = 0.0
        bar_pitch_range_std  = 0.0
    features.extend([bar_pitch_range_mean, bar_pitch_range_std])

    # 297 melodic peak density
    peaks = sum(1 for i in range(1,len(sorted_pitches)-1)
                if sorted_pitches[i]>sorted_pitches[i-1] and sorted_pitches[i]>sorted_pitches[i+1])
    melodic_peak_density = peaks/(len(sorted_pitches) or 1)
    features.append(melodic_peak_density)

    # 298 ioi_cv_global
    if ioi_stats[0]>0:
        ioi_cv_global = ioi_stats[1]/ioi_stats[0]
    else:
        ioi_cv_global = 0.0
    features.append(ioi_cv_global)

    # 299 pitch median change (first vs. last segment)
    if seg_pitches[0] and seg_pitches[-1]:
        med_first = np.median(seg_pitches[0])
        med_last  = np.median(seg_pitches[-1])
        pitch_median_change = abs(med_last - med_first)
    else:
        pitch_median_change = 0.0
    features.append(pitch_median_change)

    # 300 tritone leap ratio
    tritone_leap_ratio = sum(abs(iv)==6 for iv in intervals)/(len(intervals) or 1)
    features.append(tritone_leap_ratio)

    # 301 pc dist mean
    pc_dists = [abs((b % 12)-(a % 12)) for a,b in zip(pitches, pitches[1:])]
    pc_dist_mean = float(np.mean(pc_dists)) if pc_dists else 0.0
    features.append(pc_dist_mean)

    # 302 interval direction consistency
    dir_seq = [np.sign(iv) for iv in intervals if iv]
    consist_runs = sum(1 for a,b in zip(dir_seq, dir_seq[1:]) if a==b)
    if len(dir_seq)>1:
        interval_dir_consistency = consist_runs/(len(dir_seq)-1)
    else:
        interval_dir_consistency = 0.0
    features.append(interval_dir_consistency)

    # 303 velocity ratio half-split
    mid = len(velocities)//2
    if mid:
        vel_ratio_h12 = (np.mean(velocities[:mid])/(np.mean(velocities[mid:])+1e-9))
    else:
        vel_ratio_h12 = 0.0
    features.append(vel_ratio_h12)

    # 304 chord-quality transition entropy
    q_trans = [(a[2], b[2]) for a,b in zip(chord_list, chord_list[1:])]
    trans_counts = Counter(q_trans).values()
    q_trans_entropy = _entropy(np.fromiter(trans_counts, dtype=float)) if trans_counts else 0.0
    features.append(q_trans_entropy)

    # 305 chord changes per bar std
    if len(chord_change_times)>1:
        chord_changes_per_bar_std = float(np.std(chord_change_intervals))
    else:
        chord_changes_per_bar_std = 0.0
    features.append(chord_changes_per_bar_std)

    # 306..307 vertical consonance ratio, octave leap ratio
    vert_consonances = sum(iv in {0,3,4,5,7} for iv in vert_ints)
    vert_consonance_ratio = vert_consonances/(len(vert_ints) or 1)
    octave_leap_ratio = sum(abs(iv)==12 for iv in intervals)/(len(intervals) or 1)
    features.extend([vert_consonance_ratio, octave_leap_ratio])

    # 308 average bar pitch entropy
    pitch_entropies=[]
    for i in range(total_bars):
        bp = [n.pitch%12 for n in notes if note_bar_index(n.start,mf.ticks_per_beat,ts_num)==i]
        if bp:
            pitch_entropies.append(_entropy(np.bincount(np.array(bp),minlength=12)))
    if pitch_entropies:
        avg_bar_pitch_entropy = float(np.mean(pitch_entropies))
    else:
        avg_bar_pitch_entropy = 0.0
    features.append(avg_bar_pitch_entropy)

    # 309 pitch center trend
    if len(bar_means)>1:
        x_ = np.arange(len(bar_means), dtype=np.float32)
        y_ = np.array(bar_means, dtype=np.float32)
        pitch_center_trend = float(((x_-x_.mean())*(y_-y_.mean())).sum()/(((x_-x_.mean())**2).sum()+1e-9))
    else:
        pitch_center_trend = 0.0
    features.append(pitch_center_trend)

    # 310 melodic interval kurtosis
    mel_int_kurt = _kurtosis(intervals)
    features.append(mel_int_kurt)

    # 311 avg bar velocity iqr
    bar_vel_iqrs = []
    for i in range(total_bars):
        bv = [n.velocity for n in notes if note_bar_index(n.start,mf.ticks_per_beat,ts_num)==i]
        if bv:
            bar_vel_iqrs.append(_iqr(bv))
    if bar_vel_iqrs:
        avg_bar_vel_iqr = float(np.mean(bar_vel_iqrs))
    else:
        avg_bar_vel_iqr=0.0
    features.append(avg_bar_vel_iqr)

    # 312..313 Alberti hits, longest run (LH)
    alberti_hits, contin_run, longest_arb = 0, 0, 0
    low_octave_thr = 60
    for i in range(len(sorted_notes) - 3):
        n1, n2, n3, n4 = sorted_notes[i:i+4]
        if (n1.pitch<low_octave_thr and
            n2.pitch>n1.pitch and n2.pitch==n4.pitch and
            n1.pitch<n3.pitch<n2.pitch):
            alberti_hits += 1
            contin_run += 1
            longest_arb = max(longest_arb, contin_run)
        else:
            contin_run = 0
    features.append(alberti_hits)     # 312
    features.append(float(longest_arb)) # 313

    # 314..316 cadences
    auth_cnt = dec_cnt = plag_cnt = 0
    for i in range(len(chord_list) - 1):
        r1, r2 = chord_list[i][1] % 12, chord_list[i+1][1] % 12
        b2     = chord_list[i+1][0]
        if abs(b2 - round(b2))<1e-3:
            if r1==(key_pc+7)%12 and r2==key_pc:
                auth_cnt+=1
            if r1==(key_pc+7)%12 and r2==(key_pc+9)%12:
                dec_cnt+=1
            if r1==(key_pc+5)%12 and r2==key_pc:
                plag_cnt+=1
    features.append(auth_cnt/(bars_count or 1)) # 314
    features.append(dec_cnt/(bars_count or 1))  # 315
    features.append(plag_cnt/(bars_count or 1)) # 316

    # 317..318 local key variation
    if local_keys:=[]:
        pass  # (already integrated above as seg_keys). We'll keep simpler approach.
    key_entropy = 0.0
    mode_flips  = 0.0
    # We computed seg_keys; let's do an entropy:
    if seg_keys:
        ints_ = [12*(m=="min")+pc for (m,pc) in seg_keys]
        if ints_:
            hist__ = np.bincount(np.array(ints_), minlength=24).astype(float)
            key_entropy = _entropy(hist__)
        flips=0
        for (m1,_),(m2,_) in zip(seg_keys, seg_keys[1:]):
            if m1!=m2:
                flips+=1
        mode_flips= flips/(len(seg_keys) or 1)
    features.append(key_entropy)  # 317
    features.append(mode_flips)   # 318

    # 319 melody-bass gap
    upper_q = np.percentile(pitches, 75) if pitches else 0.0
    lower_q = np.percentile(pitches, 25) if pitches else 0.0
    mel_bass_gap = upper_q - lower_q
    features.append(float(mel_bass_gap))

    # 320 LH arpeggios
    lh_arps=0
    for i in range(len(sorted_notes)-2):
        a,b,c = sorted_notes[i:i+3]
        if a.pitch<low_octave_thr and b.pitch<low_octave_thr and c.pitch<low_octave_thr:
            if b.pitch>a.pitch and c.pitch>b.pitch and (b.start-a.start)<0.25*mf.ticks_per_beat:
                lh_arps+=1
    features.append(lh_arps/(bars_count or 1))

    # 321 cross-hand leaps
    cross_leaps=0
    for i in range(len(sorted_notes)-1):
        if (sorted_notes[i].pitch<60<=sorted_notes[i+1].pitch and
            sorted_notes[i+1].pitch>sorted_notes[i].pitch):
            cross_leaps+=1
    features.append(cross_leaps/(bars_count or 1))

    # 322 enhanced syncopation
    sync_weight = 0.0
    for n in notes:
        pos = (n.start % mf.ticks_per_beat)/mf.ticks_per_beat
        if 0.0<pos<0.5:
            next_bar_tick = (int(n.start//mf.ticks_per_beat)+1)*mf.ticks_per_beat
            tie_across = n.end>next_bar_tick
            if tie_across:
                sync_weight += (1.0-pos)
    enh_sync = sync_weight/(len(notes) or 1)
    features.append(enh_sync)

    # 323 vertical minor-second ratio
    vert_ms = sum(iv in (1,11) for iv in vert_ints)/(len(vert_ints) or 1)
    features.append(vert_ms)

    # 324 parallel 3rds/6ths runs
    par36=0
    run=0
    for iv in intervals:
        if abs(iv) in (3,4,8,9):
            run+=1
            if run==3:
                par36+=1
        else:
            run=0
    features.append(par36/(bars_count or 1))

    # 325 grace-note density
    grace_thresh=0.0625*mf.ticks_per_beat
    graces= sum(1 for n in notes if (n.end-n.start)<=grace_thresh)
    features.append(graces/(len(notes) or 1))

    # 326 bidirectional stepwise run
    longest_bi=1
    run_len=1
    for iv in intervals:
        if abs(iv) in (1,2):
            run_len+=1
            longest_bi = max(longest_bi, run_len)
        else:
            run_len=1
    features.append(float(longest_bi))

    # 327 tremolo alternations
    trem_alt=0
    for i in range(len(sorted_notes)-1):
        if (abs(sorted_notes[i+1].pitch - sorted_notes[i].pitch)>=6 and
            (sorted_notes[i+1].start - sorted_notes[i].start)<=0.25*mf.ticks_per_beat):
            trem_alt+=1
    features.append(trem_alt/(bars_count or 1))

    # 328 octave doubling
    oct_dbl= sum(iv==12 for iv in vert_ints)/(len(vert_ints) or 1)
    features.append(oct_dbl)

    # 329 turn ornaments
    turn_cnt=0
    for i in range(len(sorted_pitches)-3):
        if (sorted_pitches[i+1]==sorted_pitches[i]+1 and
            sorted_pitches[i+2]==sorted_pitches[i]-1 and
            sorted_pitches[i+3]==sorted_pitches[i]):
            turn_cnt+=1
    features.append(turn_cnt/(bars_count or 1))

    # 330 chromatic-step entropy
    chrom_ic = [abs(iv) for iv in intervals if abs(iv)==1]
    if chrom_ic:
        chrom_hist = np.bincount(chrom_ic, minlength=2).astype(np.float32)
        chrom_step_entropy = _entropy(chrom_hist)
    else:
        chrom_step_entropy = 0.0
    features.append(chrom_step_entropy)

    # 331 avg formal phrase length
    phrase_bars=[]
    last_cad_bar=0
    for cad_bar in sorted(set(int(c[0]//ts_num) for c in chord_list if (c[1]%12)==key_pc)):
        phrase_bars.append(cad_bar-last_cad_bar)
        last_cad_bar=cad_bar
    if phrase_bars:
        avg_phrase_len=float(np.mean(phrase_bars))
    else:
        avg_phrase_len=bars_count
    features.append(avg_phrase_len)

    #------------------------------------------------------------------------------
    # Advanced features (#332–#351) - tonal-centroid, tension, etc.
    #------------------------------------------------------------------------------
    # 332..334 tonal-centroid flux & curvature
    def pc_tonal_centroid(pc: int) -> np.ndarray:
        ang1 = np.pi*pc/6.0
        ang2 = np.pi*pc/3.0
        ang3 = np.pi*pc/2.0
        return np.array([np.cos(ang1), np.sin(ang1),
                         np.cos(ang2), np.sin(ang2),
                         np.cos(ang3), np.sin(ang3)], dtype=np.float32)
    tc_vecs_list = []
    for c in chord_list:                       # build a plain list first
        pcs_ = c[4]
        if pcs_:
            tc_vecs_list.append(
                np.mean([pc_tonal_centroid(p) for p in pcs_], axis=0)
            )
    
    tc_vecs = (
        np.stack(tc_vecs_list) if tc_vecs_list
        else np.empty((0, 6), np.float32)
    )
    
    if tc_vecs.shape[0] >= 2:
        tc_diffs   = np.linalg.norm(np.diff(tc_vecs, axis=0), axis=1)
        tc_flux_mu = float(tc_diffs.mean())
        tc_flux_sd = float(tc_diffs.std())
    
        if tc_vecs.shape[0] >= 3:
            v1  = np.diff(tc_vecs[:-1], axis=0)
            v2  = np.diff(tc_vecs[1:],  axis=0)
            cos = np.sum(v1 * v2, axis=1) / (
                np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9
            )
            tc_curve = float(
                np.mean(np.arccos(np.clip(cos, -1.0, 1.0)))
            )
        else:
            tc_curve = 0.0
    else:
        tc_flux_mu = tc_flux_sd = tc_curve = 0.0
    
    features.append(tc_flux_mu)   # 332
    features.append(tc_flux_sd)   # 333
    features.append(tc_curve)     # 334

    # 335..336 voice-leading smoothness
    vl_costs=[]
    for a,b in zip(chord_list, chord_list[1:]):
        p1, p2 = sorted(a[4]), sorted(b[4])
        if p1 and p2:
            m_ = min(len(p1), len(p2))
            vl_costs.append(np.mean([abs(x-y) for x,y in zip(p1[:m_],p2[:m_])]))
    if vl_costs:
        vl_mu = float(np.mean(vl_costs))
        vl_sd = float(np.std(vl_costs))
    else:
        vl_mu = 0.0
        vl_sd = 0.0
    features.append(vl_mu) # 335
    features.append(vl_sd) # 336

    # 337..339 root-motion profile
    rm = Counter(((b[1]-a[1])%12) for a,b in zip(chord_list, chord_list[1:]))
    rm_total = sum(rm.values()) or 1
    sec_ratio  = (rm[2]+rm[10])/rm_total
    tri_ratio  = rm[6]/rm_total
    rare_ratio = sum(c for s,c in rm.items() if s not in (0,2,3,4,5,7,8,9,10,11))/rm_total
    features.extend([sec_ratio, tri_ratio, rare_ratio]) # 337..339

    # 340 modal mixture usage
    mixture_cnt = sum(1 for c in chord_list
                      if (c[1]-key_pc)%12 in (1,2,3,5,8,11) and c[2] in ("maj","min"))
    mixture_ratio = mixture_cnt/(len(chord_list) or 1)
    features.append(mixture_ratio)

    # 341 chromatic-run density
    chrom_runs=0
    run=0
    for iv in intervals:
        if abs(iv)==1:
            run+=1
            if run==3:
                chrom_runs+=1
        else:
            run=0
    chrom_run_dens= chrom_runs/(bars_count or 1)
    features.append(chrom_run_dens)

    # 342 direction-change MAD
    dir_seq2 = [np.sign(iv) for iv in intervals if iv]
    if len(dir_seq2)>2:
        diffs_ = np.diff(dir_seq2)
        med_   = np.median(diffs_)
        dc_mad = float(np.median(np.abs(diffs_-med_)))
    else:
        dc_mad=0.0
    features.append(dc_mad)

    # 343 tempo-harmony correlation
    if len(tempos)>1 and chord_list:
        win = ts_num*2*mf.ticks_per_beat
        bins= np.arange(0, mf.max_tick+win, win)
        hr_hist= np.histogram([c[0]*mf.ticks_per_beat for c in chord_list], bins=bins)[0]
        tempo_ts= np.interp((bins[:-1]+win/2), [t.time for t in mf.tempo_changes], tempos)
        if len(hr_hist)>2:
            ht_corr= _safe_corr(hr_hist, tempo_ts)
        else:
            ht_corr=0.0
    else:
        ht_corr=0.0
    features.append(ht_corr)

    # 344 registral-zone transition entropy
    zones = [0 if p<60 else (1 if p<72 else 2) for p in sorted_pitches]
    zone_pairs = Counter(zip(zones,zones[1:]))
    zone_entropy= _entropy(np.fromiter(zone_pairs.values(),dtype=float)) if zone_pairs else 0.0
    features.append(zone_entropy)

    # 345 tritone-substitution rate
    tt_subs= sum(1 for c in chord_list if (c[1]-key_pc)%12==1 and c[2]=="maj")
    tt_ratio= tt_subs/(len(chord_list) or 1)
    features.append(tt_ratio)

    # 346 sixteenth-grid accent variance
    sub_div= mf.ticks_per_beat/4
    six_cnts= Counter(int(n.start//sub_div) for n in notes)
    if len(six_cnts)>1:
        six_var= float(np.std(list(six_cnts.values())))
    else:
        six_var=0.0
    features.append(six_var)

    # 347 harmonic-rhythm autocorr (lag1)
    hr_ac1= _autocorr_coeff(chord_change_intervals,lag=1) if len(chord_change_intervals)>2 else 0.0
    features.append(hr_ac1)

    # 348 ioi skewness
    ioi_skew = _skewness(ioi) if len(ioi)>2 else 0.0
    features.append(ioi_skew)

    # 349 key-clarity gap
    corrs = scores_maj + scores_min
    if len(corrs)>=2:
        best, second = sorted(corrs, reverse=True)[:2]
        key_clarity_gap = best - second
    else:
        key_clarity_gap=0.0
    features.append(key_clarity_gap)

    # 350 mean tonal-centroid magnitude
    tc_mag_mean = (
        float(np.mean(np.linalg.norm(tc_vecs, axis=1)))
        if tc_vecs.size else 0.0
    )
    features.append(tc_mag_mean)

    # 351 registral-zone kurtosis
    zone_kurt = _kurtosis(zones) if len(set(zones))>1 else 0.0
    features.append(zone_kurt)

    # 352 tonal tension (Chew & Herremans)
    weights = np.fromiter(((n.end - n.start) / mf.ticks_per_beat for n in notes),
                          dtype=np.float32,
                          count=len(notes))
    total_w = float(weights.sum())
    if total_w <= 1e-9:
        tensile_strain = 0.0
    else:
        pcs_arr = np.array([n.pitch % 12 for n in notes], dtype=np.float32)
        # 2-D unit-circle representation of pitch classes (spiral-array surrogate)
        angles   = pcs_arr * (2.0 * np.pi / 12.0)              # shape (N,)
        note_vec = np.stack((np.cos(angles), np.sin(angles)),  # shape (N,2)
                            axis=1)
    
        cloud_ce = np.average(note_vec, axis=0, weights=weights)   # centre-of-effect
        key_angle = key_pc * (2.0 * np.pi / 12.0)
        key_vec   = np.array([np.cos(key_angle), np.sin(key_angle)],
                             dtype=np.float32)
        tensile_strain = float(np.linalg.norm(cloud_ce - key_vec))
    features.append(tensile_strain)

    # 353 contrary-motion ratio (outer voices)
    onset_groups= {}
    for n in notes:
        onset_groups.setdefault(n.start,[]).append(n.pitch)
    top_seq = [max(v) for _,v in sorted(onset_groups.items())]
    bot_seq = [min(v) for _,v in sorted(onset_groups.items())]
    contrary_cnt=0
    total_mv=0
    for i in range(1, len(top_seq)):
        top_step = top_seq[i]-top_seq[i-1]
        bass_step= bot_seq[i]-bot_seq[i-1]
        if top_step and bass_step and np.sign(top_step)!=np.sign(bass_step):
            contrary_cnt+=1
        if top_step or bass_step:
            total_mv+=1
    contrary_motion_ratio = contrary_cnt/(total_mv or 1)
    features.append(contrary_motion_ratio)

    # 354 weighted Longuet–Higgins syncopation
    _beat_w= [0,-3,-2,-3, -2,-3,-2,-3, -1,-3,-2,-3, -2,-3,-2,-3]
    subdiv= mf.ticks_per_beat/4.0
    sync_score=0.0
    for n in notes:
        pos16= int((n.start%(mf.ticks_per_beat*4))//subdiv)
        # larger (neg weight) => more sync
        sync_score += -_beat_w[pos16]
    weighted_syncop= sync_score/(len(notes) or 1)
    features.append(weighted_syncop)

    # 355  Middle-C usage
    mid_c_ratio = sum(p == 60 for p in pitches) / (len(pitches) or 1)
    features.append(mid_c_ratio)

    # 356  Treble-C6 usage
    high_c6_ratio = sum(p == 84 for p in pitches) / (len(pitches) or 1)
    features.append(high_c6_ratio)

    # 357  G4 (dominant in central register)
    g4_ratio = sum(p == 67 for p in pitches) / (len(pitches) or 1)
    features.append(g4_ratio)

    # 358  Leading-tone emphasis (relative to current key)
    leading_pc = (key_pc + (11 if key_mode else 10)) % 12
    leading_tone_ratio = sum((p % 12) == leading_pc for p in pitches) / (len(pitches) or 1)
    features.append(leading_tone_ratio)

    # 359  Low-register tonic reinforcement (≤ MIDI 36)
    bass_tonic_ratio = sum(p <= 36 and (p % 12) == key_pc for p in pitches) / (len(pitches) or 1)
    features.append(bass_tonic_ratio)

    # 360  Fast-passage density (runs ≥ 8 notes, each < 0.25 beat)
    fast_notes_in_runs, run_len = 0, 0
    for d in durations:
        if d < 0.25:
            run_len += 1
        else:
            if run_len >= 8:
                fast_notes_in_runs += run_len
            run_len = 0
    if run_len >= 8:
        fast_notes_in_runs += run_len
    fast_passage_density = fast_notes_in_runs / (len(durations) or 1)
    features.append(fast_passage_density)

    # 361  Dynamic-contrast index (loud > 100 vs soft < 60)
    loud_cnt = sum(v > 100 for v in velocities)
    soft_cnt = sum(v < 60  for v in velocities)
    dyn_contrast_idx = loud_cnt / (soft_cnt + 1e-9)
    features.append(dyn_contrast_idx)

    # 362  Wide-range arpeggio density (ascending span ≥ 12 st within ≤ 1 beat)
    wide_arps = 0
    for i in range(len(sorted_notes) - 2):
        span_beats = (sorted_notes[i+2].start - sorted_notes[i].start) / mf.ticks_per_beat
        if span_beats <= 1.0:
            p1, p2, p3 = sorted_notes[i].pitch, sorted_notes[i+1].pitch, sorted_notes[i+2].pitch
            if p2 > p1 < p3 and p3 - p1 >= 12 and p3 > p2:
                wide_arps += 1
    wide_arp_density = wide_arps / (bars_count or 1)
    features.append(wide_arp_density)

    # 363  Bar-crossing note ratio
    cross_bar_cnt = sum(
        note_bar_index(n.start, mf.ticks_per_beat, ts_num) !=
        note_bar_index(max(n.end - 1, 0), mf.ticks_per_beat, ts_num)
        for n in notes
    )
    bar_cross_ratio = cross_bar_cnt / len(notes)
    features.append(bar_cross_ratio)

    # 364  Left-hand octave-doubling density (octave pairs < 60)
    lh_oct_dbl = 0
    for onset_p, group in onset_groups.items():
        lows = [p for p in group if p < 60]
        for i in range(len(lows)):
            for j in range(i + 1, len(lows)):
                if abs(lows[i] - lows[j]) == 12:
                    lh_oct_dbl += 1
    lh_oct_dbl_density = lh_oct_dbl / (bars_count or 1)
    features.append(lh_oct_dbl_density)

    # Convert to array and cache
    vec = np.array(features, dtype=np.float32)
    pickle.dump(vec, open(cache_file, "wb"))
    return vec

# (F) Multi-Processing Feature Extraction

def midi_to_feature_vec(path: str) -> Tuple[str, np.ndarray]:
    """
    Return (path, 364-dim vector).
      1) local .mid => read from disk & extract
    """
    # A) local file
    p = Path(path)
    if not p.is_file():
        warnings.warn(f"Not a valid file: {p}")
        return path, np.zeros(TOTAL_FEAT_DIM, dtype=np.float32)
        
    try:
        vec = extract_features(p)
    except Exception as e:
        # NEW – detailed report for this file
        traceback.print_exc()
        warnings.warn(f"Feature extraction failed for {p}: {e}")
        vec = np.zeros(TOTAL_FEAT_DIM, dtype=np.float32)
    return path, vec

def parse_pool(paths: List[str], n_proc: int | None = None) -> List[Tuple[str, np.ndarray]]:
    """
    Multiprocess feature extraction from multiple MIDI files.
    """
    if n_proc is None:
        n_proc = max(1, os.cpu_count()-1)
    with mp.Pool(n_proc) as pool:
        return list(pool.imap_unordered(midi_to_feature_vec, paths, chunksize=32))

# (G) The 2-model XGB + LGB Ensemble Classifier

class Task1ComposerClassifier:
    """
    Three–model (XGBoost, LightGBM, TabPFN) composer classifier.
    The final probability is a convex combination

        p = w_xgb · p_xgb + w_lgb · p_lgb + w_tpf · p_tpf

    with w_xgb + w_lgb + w_tpf = 1.
    """

    def __init__(self) -> None:
        self.xgb_model: xgb.XGBClassifier | None = None
        self.lgb_model: LGBMClassifier   | None  = None
        self.tpf_model: TabPFNClassifier | None  = None
        self.le: LabelEncoder            | None  = None

    # ------------------------------------------------------------------
    # helpers (unchanged)
    # ------------------------------------------------------------------
    @staticmethod
    def _keep_existing(mapping: dict[str, str], tag: str) -> dict[str, str]:
        virt = ("giant::", "mcm::")
        kept = {p: c for p, c in mapping.items()
                if p.startswith(virt) or Path(p).is_file()}
        dropped = len(mapping) - len(kept)
        if dropped:
            print(f"[filter] {tag:<8}: removed {dropped:4d} / {len(mapping):4d} paths (missing MIDI)")
        else:
            print(f"[filter] {tag:<8}: nothing dropped 👍")
        return kept

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def train(self, train_json: Path) -> None:
        student = gather_student(STUDENT_DIR)
        asap    = gather_asap(ASAP_META_CSV)

        student = self._keep_existing(student, "Student")
        asap    = self._keep_existing(asap,    "ASAP")
        
        _summarise_sources(student, asap)

        big: dict[str, str] = {}
        for src in [student, asap]:
            big.update(src)

        if not big:
            print("No training data found.")
            return

        paths = list(big.keys())
        labels = [big[p] for p in paths]

        print(f"\nExtracting features for {len(paths)} MIDI files …")
        path2feat = dict(parse_pool(paths))
        X = np.array([path2feat[p] for p in paths], dtype=np.float32)

        self.le = LabelEncoder().fit(labels)
        y_enc   = self.le.transform(labels)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y_enc, test_size=8 / len(X), random_state=42, stratify=y_enc
        )
        print(f"Train: {len(X_tr)}  |  Val: {len(X_val)}")

        # ---------------- XGB ----------------
        print("\nTraining XGBoost …")
        self.xgb_model = xgb.XGBClassifier(
            tree_method="hist",
            predictor="gpu_predictor",
            eval_metric="merror",
            max_depth=10,
            eta=0.15,
            subsample=0.8,
            colsample_bytree=0.8,
            n_estimators=1320,
            reg_lambda=1.2,
            reg_alpha=0.4,
            verbosity=1,
        )
        self.xgb_model.fit(X_tr, y_tr)

        # ---------------- LightGBM ----------------
        print("\nTraining LightGBM …")
        self.lgb_model = LGBMClassifier(
            n_estimators=1320,
            device_type="gpu",
            gpu_platform_id=0,
            gpu_device_id=0,
            random_state=42,
        )
        self.lgb_model.fit(X_tr, y_tr)

        # ---------------- TabPFN ----------------
        print("\nTraining TabPFN …")
        self.tpf_model = TabPFNClassifier(ignore_pretraining_limits=True, device="cuda" if torch.cuda.is_available() else "cpu")
        # TabPFN expects float32 features; no other preprocessing needed
        self.tpf_model.fit(X_tr, y_tr)

        # quick val check
        from sklearn.metrics import accuracy_score
        print("\nValidation accuracies:")
        for name, mdl in [("XGB", self.xgb_model), ("LGB", self.lgb_model), ("TPF", self.tpf_model)]:
            acc = accuracy_score(y_val, mdl.predict(X_val))
            print(f"  {name}: {acc:.3f}")

    # ------------------------------------------------------------------
    # prediction
    # ------------------------------------------------------------------
    def predict(
        self,
        test_json: Path,
        out_path: Path | None = None,
        xgb_weight: float = 0.34,
        tpf_weight: float = 0.33
    ) -> dict[str, str]:
        """
        Predict composers for all .mid files listed in `test_json` in one batch.
        """
        if any(m is None for m in (self.xgb_model, self.lgb_model, self.tpf_model, self.le)):
            raise RuntimeError("Models not trained – call `.train()` first.")
    
        # read the test JSON as before
        with test_json.open() as fp:
            rel_paths = eval(fp.read())  # e.g. a list or dict of test files
    
        # We'll gather all feature vectors in X_test, so first build a list of paths in order
        test_file_list = list(rel_paths)  # or rel_paths.keys() if it's a dict
        feats = []
        for rel in test_file_list:
            midi_file = (STUDENT_DIR / rel).resolve()
            # fallback if missing file
            if not midi_file.is_file():
                # We'll create a dummy zero vector so we can keep indexes aligned
                feats.append(np.zeros(TOTAL_FEAT_DIM, dtype=np.float32))
            else:
                feats.append(extract_features(midi_file))  # the 364-dim feature
    
        X_test = np.array(feats, dtype=np.float32)
    
        # Now get probabilities in one shot
        p_xgb = self.xgb_model.predict_proba(X_test)  # shape (n_samples, 8)
        p_lgb = self.lgb_model.predict_proba(X_test)
        p_tpf = self.tpf_model.predict_proba(X_test)
    
        lgb_weight = 1.0 - xgb_weight - tpf_weight
        # combine all prob distributions
        combined = (
            xgb_weight * p_xgb +
            lgb_weight * p_lgb +
            tpf_weight * p_tpf
        )
        best_ix = np.argmax(combined, axis=1)
        # decode back to composer names
        preds_str = self.le.inverse_transform(best_ix)
    
        # now store predictions keyed by the original relative path
        out_preds = {}
        for i, rel in enumerate(test_file_list):
            out_preds[rel] = preds_str[i]
    
        if out_path:
            out_path.write_text(json.dumps(out_preds, indent=2, ensure_ascii=False))
            print(f"Predictions written → {out_path}")
    
        return out_preds

clf = Task1ComposerClassifier()
train_json_path = STUDENT_DIR / "train.json"
clf.train(train_json_path)

# quick sanity-check on the training partition
train_labels = eval(train_json_path.read_text())
train_keys   = list(train_labels.keys())
_tmp_eval = CACHE_DIR / "tmp_train_keys.json"
_tmp_eval.write_text(repr(train_keys))
train_preds = clf.predict(_tmp_eval, xgb_weight=0.3, tpf_weight=0.6)
train_acc   = sum(train_preds.get(k) == v for k, v in train_labels.items()) / len(train_labels)

test_json_path = STUDENT_DIR / "test.json"
preds = clf.predict(
    test_json_path,
    out_path=Path("predictions1.json"),
    xgb_weight=0.3,
    tpf_weight=0.6
)
print(f"Train-split accuracy = {train_acc:.3%}")
print("Task 1 Done.")

################################################################################
# Task 2: Next sequence prediction (symbolic, binary)
################################################################################

import os, json, random, tempfile, uuid, contextlib
from pathlib import Path
from typing import Tuple, Literal, Dict, List

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

# Configuration
DATAROOT   = Path("student_files/task2_next_sequence_prediction")
TRAIN_JSON = DATAROOT / "train.json"
TEST_JSON  = DATAROOT / "test.json"

CACHE_DIR  = Path.home() / ".cache" / "midi_byte_seq_edge"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SEED        = 42
PAD_TOKEN   = 256                  # 0-255 real bytes; 256 = padding
VOCAB_SIZE  = 257
EMBED_DIM   = 256
BATCH_SIZE  = 256
EPOCHS      = 10
LR          = 3e-4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SEG_LEN     = 4096                 # bytes from each edge

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Byte-sequence helpers
def _bytes_to_seq(path: Path, part: Literal["head", "tail"]) -> np.ndarray:
    """Return SEG_LEN bytes from `path` (head or tail)."""
    tag     = f"{hash(path)}_{part}_{SEG_LEN}.npy"
    cache_f = CACHE_DIR / tag

    if cache_f.exists():
        try:
            return np.load(cache_f)
        except (EOFError, ValueError):
            cache_f.unlink(missing_ok=True)

    raw = path.read_bytes()
    seq = raw[:SEG_LEN] if part == "head" else raw[-SEG_LEN:]
    seq = np.frombuffer(seq, dtype=np.uint8).astype(np.int64)
    if len(seq) < SEG_LEN:
        pad = np.full(SEG_LEN - len(seq), PAD_TOKEN, dtype=np.int64)
        seq = np.concatenate([seq, pad])

    with tempfile.NamedTemporaryFile(dir=CACHE_DIR,
                                     suffix=f".{uuid.uuid4().hex}.tmp",
                                     delete=False) as tmp:
        np.save(tmp, seq)
        tmp_path = Path(tmp.name)

    try:
        tmp_path.replace(cache_f)
    except FileExistsError:
        tmp_path.unlink(missing_ok=True)

    return seq

# Dataset
class EdgePairDataset(Dataset):
    """For bar A deliver its tail; for bar B deliver its head."""
    def __init__(self,
                 json_file: Path | None = None,
                 infer: bool = False,
                 pairs_dict: Dict[Tuple[str, str], bool] | None = None):
        if pairs_dict is not None:
            self.keys  = list(pairs_dict)
            self.label = np.array([float(v) for v in pairs_dict.values()],
                                  dtype=np.float32)
        else:
            data = eval(json_file.read_text())
            if isinstance(data, list):                # test set (no labels)
                self.keys  = data
                self.label = np.zeros(len(data), dtype=np.float32)
            else:                                     # train set (labels)
                self.keys  = list(data)
                self.label = np.array([float(v) for v in data.values()],
                                      dtype=np.float32)

    def __len__(self): return len(self.keys)

    def __getitem__(self, idx):
        a_rel, b_rel = self.keys[idx]
        pa = Path(a_rel) if os.path.isabs(a_rel) else DATAROOT / a_rel
        pb = Path(b_rel) if os.path.isabs(b_rel) else DATAROOT / b_rel
        seq_a = _bytes_to_seq(pa, "tail")
        seq_b = _bytes_to_seq(pb, "head")
        y     = self.label[idx]
        return torch.from_numpy(seq_a), torch.from_numpy(seq_b), \
               torch.tensor(y, dtype=torch.float32)

# Siamese byte-CNN
class ByteEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=PAD_TOKEN)
        self.convs = nn.ModuleList(
            [nn.Conv1d(EMBED_DIM, 128, k, padding=k//2) for k in (3, 5, 7)]
        )
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = self.embed(x).transpose(1, 2)
        feats = [self.pool(torch.relu(c(x))).squeeze(-1) for c in self.convs]
        return torch.cat(feats, dim=1)

class PairClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = ByteEncoder()
        d = 128 * 3
        self.fc = nn.Sequential(
            nn.Linear(d * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 1)
        )

    def forward(self, a, b):
        ea, eb = self.enc(a), self.enc(b)
        x = torch.cat([ea, eb, torch.abs(ea - eb), ea * eb], dim=1)
        return self.fc(x).squeeze(1)

# Training helpers
def _run_epoch(model, loader, optim=None):
    train = optim is not None
    tot, correct, preds_all, tgt_all = 0, 0, [], []
    for a, b, y in loader:
        a, b, y = a.to(DEVICE), b.to(DEVICE), y.to(DEVICE)
        logits  = model(a, b)
        if train:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
            optim.zero_grad(); loss.backward(); optim.step()
        probs   = torch.sigmoid(logits).detach()
        preds   = (probs > 0.5).float()
        tot    += len(y)
        correct += (preds == y).sum().item()
        preds_all.append(probs.cpu()); tgt_all.append(y.cpu())
    acc = correct / tot
    auc = roc_auc_score(torch.cat(tgt_all), torch.cat(preds_all))
    return acc, auc

# Training (train.json only)
def train_model():
    student_pairs = eval(TRAIN_JSON.read_text())   # {(p1,p2): bool,…} or [(p1,p2),…]
    if isinstance(student_pairs, dict):
        all_pairs: Dict[Tuple[str, str], bool] = student_pairs
    else:                                          # list-style (assumed positives)
        all_pairs = {tuple(p): True for p in student_pairs}

    ds     = EdgePairDataset(pairs_dict=all_pairs)
    loader = DataLoader(ds, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=4, pin_memory=True)

    model  = PairClassifier().to(DEVICE)
    optim  = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        acc, auc = _run_epoch(model, loader, optim)
        print(f"[{epoch:02}] train acc {acc:.3f} | train AUC {auc:.3f}")

    return model, all_pairs

# Prediction
def predict(model: nn.Module,
            infile: Path,
            outfile: str | None = None) -> Dict[Tuple[str, str], bool]:
    """
    Infer adjacency for the candidate edge list in `infile`, enforcing that
    each segment appears in ≤ 1 TRUE edge as tail and ≤ 1 TRUE edge as head.
    """
    model.eval()
    ds = EdgePairDataset(json_file=infile, infer=True)

    batch_loader = DataLoader(ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    edge_prob: Dict[Tuple[str, str], float] = {}
    idx = 0
    with torch.no_grad():
        for a, b, _ in batch_loader:
            a, b = a.to(DEVICE), b.to(DEVICE)
            prob = torch.sigmoid(model(a, b)).cpu().numpy()
            keys = ds.keys[idx: idx + len(prob)]
            edge_prob.update({tuple(k): float(p) for k, p in zip(keys, prob)})
            idx += len(prob)

    used_tail, used_head = set(), set()
    preds: Dict[Tuple[str, str], bool] = {}
    for (a, b), p in sorted(edge_prob.items(), key=lambda kv: kv[1], reverse=True):
        if (a not in used_tail) and (b not in used_head):
            preds[(a, b)] = True
            used_tail.add(a)
            used_head.add(b)
        else:
            preds[(a, b)] = False

    if outfile:
        with open(outfile, "w") as f:
            f.write(str(preds) + "\n")
        print(f"wrote → {outfile}")

    return preds


model, all_pairs = train_model()

full_loader = DataLoader(EdgePairDataset(pairs_dict=all_pairs),
                         batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=4, pin_memory=True)
acc, auc = _run_epoch(model, full_loader)
print(f"final train accuracy = {acc:.3f} | train AUC = {auc:.3f}")

predict(model, TEST_JSON, outfile="predictions2.json")
print("Task 2 Done.")

################################################################################
# Task 3: Music tagging (continuous, multilabel, multiclass)
################################################################################

# Task-3 ▸ Music Tagging ▸ M2D-CLAP (enhanced, portable)
# ──────────────────────────────────────────────────────
# • Runs on a single CUDA-GPU (if available) or CPU
# • Class-balanced loss, waveform augmentation, 3-stage fine-tune
# • Saves predictions3.json for the autograder

# ───────────────────────── stdlib ─────────────────────────
import os, sys, json, random, contextlib, types, logging, re
from functools import partial
from pathlib import Path

os.environ["EAGER_IMPORT"] = "1"  # accelerate imports

# ─────────────────────── third-party ──────────────────────
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import average_precision_score, precision_recall_curve
import librosa
from tqdm import tqdm
import timm
from timm.models.layers import trunc_normal_
from einops import rearrange
import nnAudio.features


# ─────────────────────── models_mae stub ─────────────────
def _set_requires_grad(m, flag):  [p.requires_grad_(flag) for p in m.parameters()]


models_mae = types.SimpleNamespace(set_requires_grad=_set_requires_grad)


# ═════════════════════  Portable M2D  ═════════════════════
# (verbatim from examples/portable_m2d.py)
class Config:
    weight_file = ''
    feature_d = 768 * 5
    norm_type = all
    pooling_type = 'mean'
    model = ''
    input_size = [80, 208]
    patch_size = [16, 16]
    sr = '16k'
    flat_features = False


def expand_size(sz):  return [sz, sz] if isinstance(sz, int) else sz


class PatchEmbed(nn.Module):
    """2-D Image → Patch embedding (timm-style)."""

    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=None, flatten=True):
        super().__init__()
        img_size = expand_size(img_size)
        patch_size = expand_size(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW → BNC
        return self.norm(x)


class LocalViT(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer adapted for M2D audio."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.patch_embed = PatchEmbed(self.patch_embed.img_size, self.patch_embed.patch_size,
                                      self.patch_embed.proj.in_channels, self.patch_embed.proj.out_channels)
        self.norm_stats = nn.Parameter(torch.tensor([-7.1, 4.2]), requires_grad=False)  # default stats
        del self.head  # not used

    def patch_size(self):
        return np.array(self.patch_embed.patch_size)

    def grid_size(self):
        img, patch = np.array(self.patch_embed.img_size), self.patch_size()
        return img // patch

    def forward_encoder(self, x):
        x = self.patch_embed(x)  # B,N,C
        pos = self.pos_embed[:, 1:, :]
        if x.shape[1] < pos.shape[1]:  # shorten PE if clip < max
            d = pos.shape[-1]
            fbins = self.grid_size()[0]
            frames = x.shape[1] // fbins
            pos = pos.reshape(1, fbins, -1, d)[:, :, :frames, :].reshape(1, fbins * frames, d)
        x = x + pos
        cls = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat((cls.expand(x.size(0), -1, -1), x), dim=1)
        for blk in self.blocks:  x = blk(x)
        return self.norm(x)


# ───────────────────── helper functions ───────────────────
def _parse_sizes(dirname: str):
    """
    Robustly pull out  ➜ input H×W, patch H×W, sample-rate token ‹16k/32k›
    from a weight-folder name such as
        m2d_clap_vit_base-80x1001p16x16-240128_AS-FT_enconly
        m2d_vit_base-80x1001p16x16p32k
    """
    model_cls, remainder = dirname.split('-', 1)  # "m2d_clap_vit_base" | "80x1001p16x16-…"
    parts = remainder.split('p')  # ['80x1001', '16x16-240128_AS-FT_enconly', '32k?']

    # 1️⃣  input spectrogram size ------------------------------------------------
    input_size = [int(x) for x in parts[0].split('x')]  # 80×1001

    # 2️⃣  patch size  -----------------------------------------------------------
    patch_match = re.match(r'(\d+)x(\d+)', parts[1])  # grabs first two ints only
    if patch_match is None:
        raise ValueError(f"Can't parse patch size from “{parts[1]}”")
    patch_size = [int(patch_match.group(1)), int(patch_match.group(2))]

    # 3️⃣  sample-rate token -----------------------------------------------------
    sr_match = re.search(r'(\d+)k', parts[2] if len(parts) > 2 else '')
    sr = f"{sr_match.group(1)}k" if sr_match else '16k'  # default 16 kHz

    return input_size, patch_size, sr, model_cls


def _drop_unmatched(model: nn.Module, ckpt: dict, fname: str) -> dict:
    model_keys = {k for k, _ in model.named_parameters()}
    kept = {k: v for k, v in ckpt.items() if k in model_keys}
    print(f" loaded {len(kept):>5}/{len(ckpt):>5} layers from {Path(fname).name}")
    return kept


def _load_evar_head(ckpt, norm, head):
    if 'module.head.norm.running_mean' in ckpt:
        norm.load_state_dict({'running_mean': ckpt['module.head.norm.running_mean'],
                              'running_var': ckpt['module.head.norm.running_var']})
        head.load_state_dict({'weight': ckpt['module.head.mlp.mlp.0.weight'],
                              'bias': ckpt['module.head.mlp.mlp.0.bias']})
    else:
        print(' No EVAR head found; leaving head random.')


def _reformat_keys(ckpt):
    if 'model' in ckpt: ckpt = ckpt['model']
    return {k.replace('module.ar.runtime.backbone.', ''): v for k, v in ckpt.items()}


def _make_clap(model, ckpt):
    if 'audio_proj.0.weight' in ckpt:
        hdim = edim = ckpt['audio_proj.0.weight'].shape[1]
        model.audio_proj = nn.Sequential(nn.Linear(edim, hdim), nn.ReLU(), nn.Linear(hdim, edim))
        model.text_proj = nn.Linear(*ckpt['text_proj.weight'].shape[::-1]) \
            if 'text_proj.weight' in ckpt else nn.Identity()


def _get_melspec(cfg):
    if cfg.sr == '16k':
        cfg.sample_rate, cfg.n_fft, cfg.win, cfg.hop = 16000, 400, 400, 160
        cfg.n_mels, cfg.f_min, cfg.f_max = 80, 50, 8000
    elif cfg.sr == '32k':
        cfg.sample_rate, cfg.n_fft, cfg.win, cfg.hop = 32000, 800, 800, 320
        cfg.n_mels, cfg.f_min, cfg.f_max = 80, 50, 16000
    else:
        raise ValueError('unknown SR')

    return nnAudio.features.MelSpectrogram(
        sr=cfg.sample_rate, n_fft=cfg.n_fft, win_length=cfg.win,
        hop_length=cfg.hop, n_mels=cfg.n_mels, fmin=cfg.f_min, fmax=cfg.f_max,
        center=True, power=2, verbose=False)


def _timestamps(cfg, audio, x):
    step = len(audio[0]) / cfg.sample_rate / len(x[0]) * 1000
    return torch.arange(len(x[0]), dtype=torch.float32).mul_(step).unsqueeze(0).repeat(len(audio), 1)


class PortableM2D(nn.Module):
    """
    Stand-alone M2D runtime (audio branch only) – no external repo needed.
    """

    def __init__(self, weight_file, num_classes=None, freeze_embed=False, flat_features=None):
        super().__init__()
        self.cfg = Config();
        self.cfg.weight_file = weight_file
        self.cfg.freeze_embed = freeze_embed
        if flat_features is not None: self.cfg.flat_features = flat_features

        # backbone
        self.backbone, ckpt = self._load_backbone(weight_file)
        d = self.backbone.pos_embed.shape[-1]
        n_stack = 1 if self.cfg.flat_features else (self.cfg.input_size[0] // self.cfg.patch_size[0])
        self.cfg.feature_d = d * n_stack

        # task head
        if num_classes is not None:
            self.head_norm = nn.BatchNorm1d(self.cfg.feature_d, affine=False)
            self.head = nn.Linear(self.cfg.feature_d, num_classes)
            trunc_normal_(self.head.weight, std=2e-5)
            _load_evar_head(ckpt, self.head_norm, self.head)

        # optionally freeze patch embed
        if self.cfg.freeze_embed:
            models_mae.set_requires_grad(self.backbone.patch_embed, False)

        self.to_spec = _get_melspec(self.cfg)
        self.eval()

    # ─────────── feature helpers ───────────
    def _load_backbone(self, wfile):
        self.cfg.input_size, self.cfg.patch_size, self.cfg.sr, _ = _parse_sizes(Path(wfile).parent.name)
        ckpt = _reformat_keys(torch.load(wfile, map_location='cpu'))
        if 'norm_stats' not in ckpt:
            ckpt['norm_stats'] = torch.tensor([-7.1, 4.2])

        model = LocalViT(
            in_chans=1,
            img_size=self.cfg.input_size,
            patch_size=self.cfg.patch_size,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        _make_clap(model, ckpt)
        kept = _drop_unmatched(model, ckpt, wfile)
        print(model.load_state_dict(kept, strict=False))
        self.cfg.mean, self.cfg.std = model.state_dict()['norm_stats'].cpu().numpy()
        return model, ckpt

    def _log_mel(self, wav):
        return (self.to_spec(wav) + torch.finfo().eps).log().unsqueeze(1)

    def _norm(self, x):
        return (x - self.cfg.mean) / self.cfg.std

    def to_normalized_feature(self, wav):
        return self._norm(self._log_mel(wav))

    def encode_lms(self, x, avg_per_frame=False):
        pf = self.backbone.grid_size()[0]
        u = self.cfg.input_size[1]
        p = self.backbone.patch_size()[1]
        d = self.backbone.patch_embed.proj.out_channels
        n = (x.shape[-1] + u - 1) // u
        pad = (p - (x.shape[-1] % u % p)) % p
        if pad: x = nn.functional.pad(x, (0, pad))
        chunks = []
        for i in range(n):
            emb = self.backbone.forward_encoder(x[..., i * u:(i + 1) * u])[..., 1:, :]
            if self.cfg.flat_features:
                if avg_per_frame: emb = rearrange(emb, 'b (f t) d -> b t d f', f=pf, d=d).mean(-1)
                chunks.append(emb)
            else:
                emb = rearrange(emb, 'b (f t) d -> b t (f d)', f=pf, d=d);
                chunks.append(emb)
        return torch.cat(chunks, dim=-2)

    def encode(self, wav, avg_per_frame=False):
        return self.encode_lms(self.to_normalized_feature(wav), avg_per_frame)

    def forward(self, wav, avg_per_frame=False):
        x = self.encode(wav, avg_per_frame)
        if hasattr(self, 'head'):
            x = self.head_norm(x.mean(1).unsqueeze(-1)).squeeze(-1)
            x = self.head(x)
        return x

    # clap-specific helpers (unused here)
    def encode_clap_audio(self, wav):
        return self.backbone.audio_proj(self.forward(wav).mean(-2))

    def encode_clap_text(self, txt):
        raise NotImplementedError


# ═════════════════════  Training pipeline  ═════════════════════
# ─────────────────── general setup ────────────────────
SEED = 42
random.seed(SEED);
np.random.seed(SEED);
torch.manual_seed(SEED)
AMP = torch.cuda.is_available()
if AMP: torch.cuda.manual_seed(SEED)
autocast_ctx = torch.cuda.amp.autocast if AMP else contextlib.nullcontext
torch.backends.cudnn.benchmark = True
print(f"[info] device: {'CUDA-' + torch.cuda.get_device_name(0) if AMP else 'CPU'}")

# ─────────────────────── constants ──────────────────────
SAMPLE_RATE = 16_000
AUDIO_DURATION = 10  # seconds
TAGS = ["rock", "oldies", "jazz", "pop", "dance", "blues", "punk", "chill", "electronic", "country"]
N_TAGS = len(TAGS)
WEIGHT_PATH = ("m2d/m2d_clap_vit_base-80x1001p16x16-240128_AS-FT_enconly/"
               "weights_ep67it3124-0.48558.pth")


# ──────────────── augmentation utils ────────────────
def _rand_crop(wav, tgt_len):
    if len(wav) > tgt_len:
        i = random.randint(0, len(wav) - tgt_len);
        return wav[i:i + tgt_len]
    return np.pad(wav, (0, tgt_len - len(wav)))


def _pink_noise(n):
    x = np.random.randn(16, n).cumsum(1);
    return x[-1] / x[-1].std()


def augment_waveform(wav: np.ndarray) -> np.ndarray:
    """Random crop → gain jitter → (optional) pink-noise at ±10 dB SNR."""
    wav = _rand_crop(wav, SAMPLE_RATE * AUDIO_DURATION)
    wav = wav * random.uniform(0.9, 1.1)  # gain jitter

    if random.random() < .5:  # 50 % chance
        snr_db = random.uniform(8, 12)
        sig_pow = np.mean(wav ** 2)
        noise = _pink_noise(len(wav))
        noise_pow = np.mean(noise ** 2)
        noise *= np.sqrt(sig_pow / (10 ** (snr_db / 10) * noise_pow))
        wav += noise
    return np.clip(wav, -1.0, 1.0)


# ─────────────── dataset / dataloader ────────────────
class AudioTaggingDataset(Dataset):
    def __init__(self, json_path, droot, training):
        meta = eval(Path(json_path).read_text())
        self.fnames = list(meta.keys())
        self.labels = [torch.tensor([1 if t in meta[f] else 0 for t in TAGS], dtype=torch.float32)
                       for f in self.fnames]
        self.droot = droot
        self.training = training

    def __len__(self): return len(self.fnames)

    def __getitem__(self, idx):
        fn = self.fnames[idx]
        wav, _ = librosa.load(os.path.join(self.droot, fn), sr=SAMPLE_RATE, mono=True)
        wav = augment_waveform(wav) if self.training else _rand_crop(wav, SAMPLE_RATE * AUDIO_DURATION)
        return torch.tensor(wav, dtype=torch.float32), self.labels[idx], fn


class InferenceDataset(Dataset):
    def __init__(self, json_path, droot):
        self.fnames = list(eval(Path(json_path).read_text()));
        self.droot = droot

    def __len__(self): return len(self.fnames)

    def __getitem__(self, idx):
        fn = self.fnames[idx]
        wav, _ = librosa.load(os.path.join(self.droot, fn), sr=SAMPLE_RATE, mono=True)
        wav = _rand_crop(wav, SAMPLE_RATE * AUDIO_DURATION)
        return torch.tensor(wav, dtype=torch.float32), torch.zeros(N_TAGS), fn


def collate(x):  wav, y, f = zip(*x);  return torch.stack(wav), torch.stack(y), f


# ──────────────────── model wrapper ───────────────────
class M2DClassifier(nn.Module):
    def __init__(self, weight_path, n_tags, finetune=False):
        super().__init__()
        self.backbone = PortableM2D(weight_path, num_classes=None,
                                    freeze_embed=True, flat_features=False)
        if not finetune:
            for p in self.backbone.parameters(): p.requires_grad_(False)
        self.head = nn.Sequential(
            nn.LayerNorm(self.backbone.cfg.feature_d),
            nn.Linear(self.backbone.cfg.feature_d, n_tags)
        )

    def forward(self, wav):
        feats = self.backbone.encode_lms(
            self.backbone.to_normalized_feature(wav), avg_per_frame=False)
        return self.head(feats.mean(1))  # clip-level logits


# ───────────────── training loop ──────────────────
class Task3:
    def __init__(self, root):
        self.train_json = os.path.join(root, "train.json")
        self.test_json = os.path.join(root, "test.json")
        self.droot = root
        self.device = torch.device("cuda" if AMP else "cpu")

        self.model = M2DClassifier(WEIGHT_PATH, N_TAGS).to(self.device)

        y_mat = torch.stack(AudioTaggingDataset(self.train_json, self.droot, False).labels)
        freq = y_mat.mean(0).clamp(1e-4, 1 - 1e-4)
        self.pos_weight = ((1 - freq) / freq).to(self.device)
        self.thresholds = torch.full((N_TAGS,), 0.5)

    # ─────── train / predict ───────
    def train(self, ep_head=1, ep_partial=2, ep_full=1,
              bs=8, lr_head=1e-3, lr_partial=5e-5, lr_full=1e-5, wd=0.05):

        full = AudioTaggingDataset(self.train_json, self.droot, True)
        tr, va = random_split(full, [int(.9 * len(full)), len(full) - int(.9 * len(full))])

        tr_ld = DataLoader(tr, bs, True, num_workers=4, pin_memory=AMP, collate_fn=collate)
        va_ld = DataLoader(va, bs, False, num_workers=4, pin_memory=AMP, collate_fn=collate)
        crit = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        # phase 1 – head only
        self._run(tr_ld, va_ld, crit, AdamW(self.model.head.parameters(), lr_head, weight_decay=wd),
                  ep_head, "head")

        # phase 2 – unfreeze last-4 blocks
        for blk in self.model.backbone.backbone.blocks[-4:]:
            for p in blk.parameters(): p.requires_grad_(True)
        opt = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr_partial, weight_decay=wd)
        self._run(tr_ld, va_ld, crit, opt, ep_partial, "partial",
                  CosineAnnealingLR(opt, T_max=ep_partial))

        # phase 3 – full backbone
        for p in self.model.backbone.parameters(): p.requires_grad_(True)
        self._run(tr_ld, va_ld, crit, AdamW(self.model.parameters(), lr_full, weight_decay=wd),
                  ep_full, "full")

        self.thresholds = self._optimal_thresholds(va_ld)
        print("[✓] per-tag thresholds:",
              {t: round(float(v), 3) for t, v in zip(TAGS, self.thresholds)})

    def predict(self, out="predictions3.json", bs=8):
        ld = DataLoader(InferenceDataset(self.test_json, self.droot), bs, False,
                        num_workers=4, pin_memory=AMP, collate_fn=collate)
        self.model.eval();
        preds = {}
        with torch.no_grad():
            for wav, _, fns in tqdm(ld, desc="infer"):
                wav = wav.to(self.device, non_blocking=True)
                with autocast_ctx():
                    p = torch.sigmoid(self.model(wav)).cpu()
                for fn, pr in zip(fns, p):
                    preds[fn] = [t for t, prob, th in zip(TAGS, pr, self.thresholds) if prob > th]
        json.dump(preds, open(out, "w"), indent=2);
        print(f"[✓] wrote {out}")

    # ─────── helpers ───────
    def _run(self, tr, va, crit, opt, n_ep, tag, sched=None):
        scaler = torch.cuda.amp.GradScaler(enabled=AMP)
        for ep in range(1, n_ep + 1):
            self._train_epoch(tr, crit, opt, scaler, ep, n_ep, tag)
            self._val_epoch(va, crit)
            if sched: sched.step()

    def _train_epoch(self, ld, crit, opt, scaler, ep, n_ep, tag):
        self.model.train();
        run = 0
        pbar = tqdm(ld, desc=f"[{tag} {ep}/{n_ep}]")
        for wav, y, _ in pbar:
            wav, y = wav.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            opt.zero_grad()
            with autocast_ctx(): loss = crit(self.model(wav), y)
            scaler.scale(loss).backward();
            scaler.step(opt);
            scaler.update()
            run += loss.item();
            pbar.set_postfix(loss=run / len(pbar))

    @torch.no_grad()
    def _val_epoch(self, ld, crit):
        self.model.eval();
        preds, tgts, vl = [], [], 0
        for wav, y, _ in ld:
            wav, y = wav.to(self.device), y.to(self.device)
            with autocast_ctx(): logits = self.model(wav); vl += crit(logits, y).item()
            preds.append(torch.sigmoid(logits).cpu().numpy());
            tgts.append(y.cpu().numpy())
        mAP = average_precision_score(np.vstack(tgts), np.vstack(preds), average="macro")
        print(f"  ↳ val mAP {mAP:.4f} | loss {vl / len(ld):.4f}")

    @torch.no_grad()
    def _optimal_thresholds(self, ld):
        self.model.eval();
        scrs, tgts = [], []
        for wav, y, _ in ld:
            wav = wav.to(self.device, non_blocking=True)
            with autocast_ctx(): scrs.append(torch.sigmoid(self.model(wav)).cpu()); tgts.append(y)
        y_true, y_pred = torch.cat(tgts).numpy(), torch.cat(scrs).numpy()
        thr = []
        for k in range(N_TAGS):
            p, r, t = precision_recall_curve(y_true[:, k], y_pred[:, k])
            f1 = 2 * p * r / np.maximum(p + r, 1e-8)
            thr.append(float(t[np.nanargmax(f1)]) if len(t) else 0.5)
        return torch.tensor(thr)


# ───────────────────────────── run ─────────────────────────────
ROOT = "student_files/task3_audio_classification"
task = Task3(ROOT)
task.train(ep_head=1, ep_partial=2, ep_full=1, lr_head=1e-3, lr_partial=5e-5, lr_full=1e-5, wd=0.18)
task.predict("predictions3.json")
