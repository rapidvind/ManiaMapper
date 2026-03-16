"""
ManiaNNTrainer.py  —  osu!mania 4K trainer  (v4)
=================================================
What's new in v4
-----------------
  Features (114 audio + 16 cross-diff context = 130 total):
    mel spectrogram (80)        — tonal content, main signal
    onset strength (1)          — WHEN hits occur
    onset_harm (1)              — onset on harmonic component
    onset_perc (1)              — onset on percussive component
    spectral contrast (7)       — peak/valley texture
    chroma CQT (12)             — harmonic/melodic content
    RMS energy (1)              — loudness
    spectral flatness (1)       — noisy vs tonal
    spectral centroid (1)       — brightness, normalized 0-1
    zero crossing rate (1)      — noisiness indicator
    phases @ 4 temporal scales  — 8 dims (sin/cos × 4 scales: 1,2,4,8 beat)
    cross-diff context (16)     — sibling map hit_labels for same audio

  Architecture: PatternConvBlock(k=3,5,9,13) + Transformer + 4 DiffHeads
    + pattern auxiliary head (9 pattern classes)
    Long-note (LN) support: ln_labels, ln_dur regression

  Training:
    Multi-task loss: hit placement + LN classification + LN duration + pattern
    Sibling diff auxiliary (weight=0.2, hit loss only)
    Warmup (3 epochs) + cosine decay, gradient clipping 1.0
    AdamW lr=2e-4, weight_decay=2e-4

Usage
-----
    python ManiaNNTrainer.py
    python ManiaNNTrainer.py --maps-dir C:\\ManiaStyles --epochs 50
    python ManiaNNTrainer.py --epochs 50 --max-maps 3000
"""

import os, sys, argparse, warnings
import numpy as np

warnings.filterwarnings("ignore")

# ── Feature / grid config ──────────────────────────────────────────────────────
N_MEL       = 80
N_SC        = 7           # spectral contrast bands
N_CHROMA    = 12          # chroma CQT features
SUBDIV      = 8           # 32nd-note training grid
HOP         = 512
SR          = 22050
# Audio features: mel(80) + onset(1) + onset_harm(1) + onset_perc(1) + sc(7) +
#                 chroma(12) + rms(1) + flatness(1) + centroid(1) + zcr(1) +
#                 phases_4scales(8) = 114
FEAT_DIM_AUDIO = 114
FEAT_DIM_CTX   = 16       # cross-diff context (4 diffs × 4 cols)
FEAT_DIM       = FEAT_DIM_AUDIO + FEAT_DIM_CTX   # = 130
SEQ_LEN     = 256
DIFF_LEVELS = 4
DIFF_EMB    = 32
MAX_LN_BEATS = 16.0       # max LN duration in beats
NUM_PATTERNS = 9          # pattern type classes

# Transformer hyper-parameters
D_MODEL  = 384
NHEAD    = 8
N_LAYERS = 6
DIM_FF   = 1536
DROPOUT  = 0.20

# Pattern constants
PATTERN_REST    = 0
PATTERN_SINGLE  = 1
PATTERN_CHORD   = 2
PATTERN_JACK    = 3
PATTERN_TRILL   = 4
PATTERN_STAIR_U = 5
PATTERN_STAIR_D = 6
PATTERN_STREAM  = 7
PATTERN_LN      = 8

DEFAULT_MAPS_DIR = r"C:\Users\Aravind Dora\Desktop\ManiaMapper\ManiaStyles"
DEFAULT_OUT      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mania_model.pt")


# ── osu! parser ───────────────────────────────────────────────────────────────

def x_to_col(x):
    return min(3, max(0, x * 4 // 512))


def version_to_diff(version_str):
    v = version_str.lower()
    if any(k in v for k in ("easy", "beginner", "basic", "intro", "lv1", "lv2")):
        return 0
    if any(k in v for k in ("normal", "standard", "medium", "lv3", "lv4")):
        return 1
    if any(k in v for k in ("hard", "advanced", "hyper", "lv5", "lv6", "lv7")):
        return 2
    if any(k in v for k in ("insane", "another", "extra", "extreme", "maximum",
                              "expert", "master", "lunatic", "mx", "sc", "lv8",
                              "lv9", "lv10", "lv11", "lv12", "legend")):
        return 3
    return -1


def parse_osu(path):
    """
    Parse an osu!mania 4K beatmap file.
    raw_notes entries: (time_ms, col, is_ln: bool, ln_end_ms: int)
    groups: list of (time_ms, [(col, is_ln, ln_end_ms), ...])
    Returns (audio_file, groups, version_str) or None.
    """
    mode = audio_file = version = None
    keys = None
    in_hitobjects = False
    raw_notes = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line.startswith("AudioFilename:"):
                    audio_file = line.split(":", 1)[1].strip()
                elif line.startswith("Version:"):
                    version = line.split(":", 1)[1].strip()
                elif line.startswith("Mode:"):
                    try: mode = int(line.split(":")[1].strip())
                    except ValueError: pass
                elif line.startswith("CircleSize:"):
                    try: keys = float(line.split(":")[1].strip())
                    except ValueError: pass
                elif line == "[HitObjects]":
                    in_hitobjects = True
                elif line.startswith("[") and line.endswith("]") and in_hitobjects:
                    break
                elif in_hitobjects and line and not line.startswith("//"):
                    parts = line.split(",")
                    if len(parts) < 4:
                        continue
                    try:
                        x         = int(parts[0])
                        time_ms   = int(parts[2])
                        note_type = int(parts[3])
                        col       = x_to_col(x)
                        if note_type == 128:
                            # Long note: endTime encoded in parts[5] before ':'
                            ln_end_ms = 0
                            if len(parts) >= 6:
                                try:
                                    ln_end_ms = int(parts[5].split(":")[0])
                                except (ValueError, IndexError):
                                    ln_end_ms = 0
                            raw_notes.append((time_ms, col, True, ln_end_ms))
                        else:
                            raw_notes.append((time_ms, col, False, 0))
                    except ValueError:
                        continue
    except Exception:
        return None

    if mode != 3 or keys != 4.0 or not audio_file or not raw_notes:
        return None

    raw_notes.sort(key=lambda n: n[0])

    groups, i = [], 0
    while i < len(raw_notes):
        t0 = raw_notes[i][0]
        note_group = []
        j = i
        while j < len(raw_notes) and raw_notes[j][0] - t0 <= 10:
            _, col, is_ln, ln_end_ms = raw_notes[j]
            note_group.append((col, is_ln, ln_end_ms))
            j += 1
        groups.append((t0, note_group))
        i = j

    return audio_file, groups, (version or "")


# ── Audio feature extraction ──────────────────────────────────────────────────

def extract_audio_features(audio_path):
    """
    Extract all audio features for a single audio file.
    Returns (feat_audio, frame_times, beat_length, duration_ms, positions, beat_times)
    or None on failure.
    feat_audio : (T_grid, FEAT_DIM_AUDIO=114) float32
    """
    try:
        import librosa
    except ImportError:
        print("ERROR: pip install librosa"); sys.exit(1)

    try:
        y, sr = librosa.load(audio_path, sr=SR, mono=True)
    except Exception:
        return None

    duration_ms = len(y) / SR * 1000

    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=SR)
        beat_times = librosa.frames_to_time(beat_frames, sr=SR) * 1000
        bpm = float(np.atleast_1d(tempo)[0])
    except Exception:
        return None

    if len(beat_times) == 0 or bpm < 60 or bpm > 400:
        return None

    beat_length = 60000.0 / bpm
    step_ms     = beat_length / SUBDIV

    # ── harmonic / percussive separation ──────────────────────────────────────
    try:
        y_harm, y_perc = librosa.effects.hpss(y)
    except Exception:
        y_harm = y
        y_perc = y

    # ── mel spectrogram ───────────────────────────────────────────────────────
    try:
        mel    = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MEL, hop_length=HOP)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
    except Exception:
        return None

    n_frames    = mel_db.shape[1]
    frame_times = librosa.frames_to_time(np.arange(n_frames), sr=SR, hop_length=HOP) * 1000

    # ── onset strength (full) ─────────────────────────────────────────────────
    try:
        onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
        onset = (onset - onset.mean()) / (onset.std() + 1e-9)
    except Exception:
        onset = np.zeros(n_frames, dtype=np.float32)

    # ── onset_harm ────────────────────────────────────────────────────────────
    try:
        onset_harm = librosa.onset.onset_strength(y=y_harm, sr=SR, hop_length=HOP)
        onset_harm = (onset_harm - onset_harm.mean()) / (onset_harm.std() + 1e-9)
    except Exception:
        onset_harm = np.zeros(n_frames, dtype=np.float32)

    # ── onset_perc ────────────────────────────────────────────────────────────
    try:
        onset_perc = librosa.onset.onset_strength(y=y_perc, sr=SR, hop_length=HOP)
        onset_perc = (onset_perc - onset_perc.mean()) / (onset_perc.std() + 1e-9)
    except Exception:
        onset_perc = np.zeros(n_frames, dtype=np.float32)

    # ── spectral contrast ─────────────────────────────────────────────────────
    try:
        sc = librosa.feature.spectral_contrast(y=y, sr=SR, hop_length=HOP)  # (7, T)
        sc = (sc - sc.mean(axis=1, keepdims=True)) / (sc.std(axis=1, keepdims=True) + 1e-9)
    except Exception:
        sc = np.zeros((N_SC, n_frames), dtype=np.float32)

    # ── chroma CQT ────────────────────────────────────────────────────────────
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP)  # (12, T)
        chroma = (chroma - chroma.mean(axis=1, keepdims=True)) / (chroma.std(axis=1, keepdims=True) + 1e-9)
    except Exception:
        chroma = np.zeros((N_CHROMA, n_frames), dtype=np.float32)

    # ── RMS energy ────────────────────────────────────────────────────────────
    try:
        rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
        rms = (rms - rms.mean()) / (rms.std() + 1e-9)
    except Exception:
        rms = np.zeros(n_frames, dtype=np.float32)

    # ── spectral flatness ─────────────────────────────────────────────────────
    try:
        flat = librosa.feature.spectral_flatness(y=y, hop_length=HOP)[0]
        flat = (flat - flat.mean()) / (flat.std() + 1e-9)
    except Exception:
        flat = np.zeros(n_frames, dtype=np.float32)

    # ── spectral centroid ─────────────────────────────────────────────────────
    try:
        centroid = librosa.feature.spectral_centroid(y=y, sr=SR, hop_length=HOP)[0]
        centroid = centroid / (SR / 2.0)   # normalize to [0, 1]
    except Exception:
        centroid = np.zeros(n_frames, dtype=np.float32)

    # ── zero crossing rate ────────────────────────────────────────────────────
    try:
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=HOP)[0]
        zcr = (zcr - zcr.mean()) / (zcr.std() + 1e-9)
    except Exception:
        zcr = np.zeros(n_frames, dtype=np.float32)

    # ── uniform beat grid ─────────────────────────────────────────────────────
    beat0_ms  = float(beat_times[0])
    positions = []
    t = beat0_ms
    while t <= duration_ms:
        positions.append(t)
        t += step_ms

    if len(positions) < 64:
        return None

    # ── build feature matrix at each grid position ───────────────────────────
    def feat_at(t_ms):
        idx = min(int(np.searchsorted(frame_times, t_ms)), n_frames - 1)
        f   = np.zeros(FEAT_DIM_AUDIO, dtype=np.float32)
        # mel: 0..79
        f[:N_MEL]                                   = mel_db[:, idx]
        # onset: 80
        f[N_MEL]                                    = onset[idx]
        # onset_harm: 81
        f[N_MEL + 1]                                = onset_harm[idx]
        # onset_perc: 82
        f[N_MEL + 2]                                = onset_perc[idx]
        # sc: 83..89
        base_sc = N_MEL + 3
        f[base_sc : base_sc + N_SC]                 = sc[:, idx]
        # chroma: 90..101
        base_ch = base_sc + N_SC
        f[base_ch : base_ch + N_CHROMA]             = chroma[:, idx]
        # rms: 102
        base_rms = base_ch + N_CHROMA
        f[base_rms]                                 = rms[idx]
        # flatness: 103
        f[base_rms + 1]                             = flat[idx]
        # centroid: 104
        f[base_rms + 2]                             = centroid[idx]
        # zcr: 105
        f[base_rms + 3]                             = zcr[idx]
        # phases @ 4 temporal scales: 106..113
        base_ph = base_rms + 4
        for k, scale in enumerate([1, 2, 4, 8]):
            period = beat_length * scale
            phase  = (t_ms % period) / (period + 1e-9)
            f[base_ph + k * 2]     = float(np.sin(2 * np.pi * phase))
            f[base_ph + k * 2 + 1] = float(np.cos(2 * np.pi * phase))
        return f

    feat_audio = np.stack([feat_at(t) for t in positions])   # (T, 114)
    return feat_audio, frame_times, beat_length, duration_ms, positions, beat_times


def extract_labels(groups, positions, beat_length):
    """
    Map hit groups to label arrays.
    Returns:
        hit_labels:  (T, 4) float32 — 1 if any note (hit or LN start) in column
        ln_labels:   (T, 4) float32 — 1 if note is a long note
        ln_dur_lbls: (T, 4) float32 — LN duration in beats (0 for non-LN)
    """
    step_ms = beat_length / SUBDIV
    TOL_MS  = max(step_ms * 0.6, 20.0)
    T       = len(positions)

    hit_labels  = np.zeros((T, 4), dtype=np.float32)
    ln_labels   = np.zeros((T, 4), dtype=np.float32)
    ln_dur_lbls = np.zeros((T, 4), dtype=np.float32)

    pos_arr = np.array(positions)

    for t_note, note_group in groups:
        dists   = np.abs(pos_arr - t_note)
        closest = int(np.argmin(dists))
        if dists[closest] > TOL_MS:
            continue
        for col, is_ln, ln_end_ms in note_group:
            hit_labels[closest, col] = 1.0
            if is_ln:
                ln_labels[closest, col]   = 1.0
                dur_ms                    = max(0.0, ln_end_ms - t_note)
                dur_beats                 = dur_ms / (beat_length + 1e-9)
                ln_dur_lbls[closest, col] = min(float(dur_beats), MAX_LN_BEATS)

    return hit_labels, ln_labels, ln_dur_lbls


def derive_pattern_types(hit_labels, ln_labels):
    """
    Derive pattern type for each timestep.
    hit_labels: (T, 4), ln_labels: (T, 4)
    Returns int64 array (T,).
    """
    T = hit_labels.shape[0]
    pattern_types = np.zeros(T, dtype=np.int64)

    for t in range(T):
        active_cols = np.where(hit_labels[t] > 0.5)[0]
        n_active    = len(active_cols)

        if n_active == 0:
            pattern_types[t] = PATTERN_REST
            continue

        # LN start takes precedence
        if ln_labels[t, active_cols].any():
            pattern_types[t] = PATTERN_LN
            continue

        if n_active >= 2:
            pattern_types[t] = PATTERN_CHORD
            continue

        col = int(active_cols[0])

        # Look back up to 4 steps
        lookback = min(t, 4)
        prev_cols = []
        for k in range(1, lookback + 1):
            prev_active = np.where(hit_labels[t - k] > 0.5)[0]
            if len(prev_active) > 0:
                prev_cols.append(prev_active.tolist())

        # Jack: same column appears in any of the last 4 steps
        is_jack = False
        for pcols in prev_cols:
            if col in pcols:
                is_jack = True
                break
        if is_jack:
            pattern_types[t] = PATTERN_JACK
            continue

        # Collect single-note columns from last 4 steps
        single_col_history = []
        for k in range(1, lookback + 1):
            prev_active = np.where(hit_labels[t - k] > 0.5)[0]
            if len(prev_active) == 1:
                single_col_history.append(int(prev_active[0]))
        single_col_history = single_col_history[:3]  # up to 3 previous single notes

        # Trill: alternating between exactly 2 columns over last 4 steps
        all_single_cols = [col] + single_col_history
        if len(all_single_cols) >= 4:
            unique_cols = set(all_single_cols[:4])
            if len(unique_cols) == 2:
                cols_seq = all_single_cols[:4]
                is_trill = all(
                    cols_seq[i] != cols_seq[i + 1] for i in range(len(cols_seq) - 1)
                )
                if is_trill:
                    pattern_types[t] = PATTERN_TRILL
                    continue

        # Stair up/down: strictly ascending or descending cols over last 4 steps
        if len(all_single_cols) >= 4:
            cols_seq = all_single_cols[:4]
            if all(cols_seq[i] < cols_seq[i + 1] for i in range(len(cols_seq) - 1)):
                pattern_types[t] = PATTERN_STAIR_U
                continue
            if all(cols_seq[i] > cols_seq[i + 1] for i in range(len(cols_seq) - 1)):
                pattern_types[t] = PATTERN_STAIR_D
                continue

        pattern_types[t] = PATTERN_STREAM

    return pattern_types


def extract_features_and_labels(audio_path, groups, version_str):
    """
    Returns (feat_audio, hit_labels, ln_labels, ln_dur_lbls, pattern_types, diff_level, audio_path)
    or (None,)*7.
    feat_audio: (T, FEAT_DIM_AUDIO) — cross-diff context added later.
    """
    result = extract_audio_features(audio_path)
    if result is None:
        return (None,) * 7

    feat_audio, _, beat_length, _, positions, _ = result
    hit_labels, ln_labels, ln_dur_lbls = extract_labels(groups, positions, beat_length)

    note_ratio = hit_labels.sum() / (hit_labels.size + 1e-9)
    if note_ratio < 0.02 or note_ratio > 0.80:
        return (None,) * 7

    diff_level = version_to_diff(version_str)
    if diff_level < 0:
        if   note_ratio < 0.08: diff_level = 0
        elif note_ratio < 0.18: diff_level = 1
        elif note_ratio < 0.28: diff_level = 2
        else:                   diff_level = 3

    pattern_types = derive_pattern_types(hit_labels, ln_labels)

    return feat_audio, hit_labels, ln_labels, ln_dur_lbls, pattern_types, diff_level, audio_path


# ── Cross-difficulty context ──────────────────────────────────────────────────

def build_cross_diff_context(sequences):
    """
    Group sequences by audio_path. For each sequence, append 16 context features
    representing sibling difficulty hit_labels at each position:
        [Easy_C1..Easy_C4 | Normal_C1..C4 | Hard_C1..C4 | Insane_C1..C4]
    Where a difficulty is absent, the 4 values are zero.
    Returns list of (feat_full, hit_labels, ln_labels, ln_dur_lbls, pattern_types, diff_level).
    """
    from collections import defaultdict

    # Map audio_path -> {diff_level: hit_labels}
    audio_map = defaultdict(dict)
    for feat_audio, hit_lbl, ln_lbl, ln_dur, pat_types, diff_level, audio_path in sequences:
        audio_map[audio_path][diff_level] = hit_lbl

    result = []
    for feat_audio, hit_lbl, ln_lbl, ln_dur, pat_types, diff_level, audio_path in sequences:
        T   = len(feat_audio)
        ctx = np.zeros((T, FEAT_DIM_CTX), dtype=np.float32)
        for d in range(DIFF_LEVELS):
            if d == diff_level:
                continue
            if d in audio_map[audio_path]:
                sibling_lbl = audio_map[audio_path][d]
                n = min(T, len(sibling_lbl))
                ctx[:n, d * 4:(d + 1) * 4] = sibling_lbl[:n]
        feat_full = np.concatenate([feat_audio, ctx], axis=1)   # (T, 130)
        result.append((feat_full, hit_lbl, ln_lbl, ln_dur, pat_types, diff_level))

    n_cross = sum(1 for ap, dmap in audio_map.items() if len(dmap) > 1)
    print(f"  Cross-diff pairs: {n_cross} audio files with 2+ difficulties")
    return result


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(feat_dim=FEAT_DIM, diff_levels=DIFF_LEVELS, diff_emb=DIFF_EMB,
                d_model=D_MODEL, nhead=NHEAD, num_layers=N_LAYERS,
                dim_ff=DIM_FF, dropout=DROPOUT):
    try:
        import torch
        import torch.nn as nn
        import math
    except ImportError:
        print("ERROR: pip install torch"); sys.exit(1)

    class PositionalEncoding(nn.Module):
        def __init__(self, d, max_len=2048, drop=0.1):
            super().__init__()
            self.drop = nn.Dropout(drop)
            pe   = torch.zeros(max_len, d)
            pos  = torch.arange(0, max_len).unsqueeze(1).float()
            div  = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return self.drop(x + self.pe[:, :x.size(1)])

    class PatternConvBlock(nn.Module):
        """
        Multi-scale 1D depthwise convolution.
        k=3  -> note pairs (jacks, 2-note bursts)
        k=5  -> trills and short streams
        k=9  -> longer streams (8-note runs)
        k=13 -> LN span detection
        """
        def __init__(self, d, drop=0.1):
            super().__init__()
            self.convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(d, d, k, padding=k // 2, groups=d),  # depthwise
                    nn.Conv1d(d, d, 1),                              # pointwise
                    nn.GELU(),
                )
                for k in [3, 5, 9, 13]
            ])
            self.proj = nn.Linear(4 * d, d)
            self.norm = nn.LayerNorm(d)
            self.drop = nn.Dropout(drop)

        def forward(self, x):
            h = x.transpose(1, 2)                                       # (B, D, T)
            outs = [conv(h).transpose(1, 2) for conv in self.convs]     # 4×(B,T,D)
            h = torch.cat(outs, dim=-1)                                  # (B, T, 4D)
            return self.norm(x + self.drop(self.proj(h)))               # residual

    class DiffHead(nn.Module):
        def __init__(self, d_in, drop=0.1):
            super().__init__()
            self.shared    = nn.Sequential(
                nn.Linear(d_in, d_in // 2),
                nn.GELU(),
                nn.Dropout(drop),
            )
            self.hit_out    = nn.Linear(d_in // 2, 4)
            self.ln_out     = nn.Linear(d_in // 2, 4)
            self.ln_dur_out = nn.Linear(d_in // 2, 4)

        def forward(self, h):
            import torch
            h2 = self.shared(h)
            return (
                self.hit_out(h2),
                self.ln_out(h2),
                torch.sigmoid(self.ln_dur_out(h2)) * MAX_LN_BEATS,
            )

    class ManiaTransformerV4(nn.Module):
        """
        v4: PatternConvBlock(k=3,5,9,13) + Transformer + 4 DiffHeads + pattern head.

        Input : x    (B, T, feat_dim)
                diff (B,)  — difficulty index 0-3
        Output: (list of 4 DiffHead outputs, pattern_logits)
                DiffHead output: (hit_logits, ln_logits, ln_dur)  each (B, T, 4)
                pattern_logits: (B, T, NUM_PATTERNS)
        """
        def __init__(self):
            super().__init__()
            self.diff_emb    = nn.Embedding(diff_levels, diff_emb)
            self.input_proj  = nn.Linear(feat_dim + diff_emb, d_model)
            self.conv_block  = PatternConvBlock(d_model, drop=dropout)
            self.pos_enc     = PositionalEncoding(d_model, drop=dropout)
            enc_layer        = nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward=dim_ff,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
            self.norm        = nn.LayerNorm(d_model)
            self.pattern_head = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, NUM_PATTERNS),
            )
            self.heads = nn.ModuleList([
                DiffHead(d_model, drop=dropout)
                for _ in range(diff_levels)
            ])

        def forward(self, x, diff):
            B, T, _ = x.shape
            d = self.diff_emb(diff).unsqueeze(1).expand(B, T, -1)
            h = self.input_proj(torch.cat([x, d], dim=-1))
            h = self.conv_block(h)
            h = self.pos_enc(h)
            h = self.transformer(h)
            h = self.norm(h)
            pattern_logits = self.pattern_head(h)         # (B, T, NUM_PATTERNS)
            head_outputs   = [head(h) for head in self.heads]   # list of 4 tuples
            return head_outputs, pattern_logits

    # Need torch imported here for the model to reference it
    try:
        import torch
    except ImportError:
        print("ERROR: pip install torch"); sys.exit(1)

    return ManiaTransformerV4()


# ── Dataset ───────────────────────────────────────────────────────────────────

def make_dataset(sequences):
    try:
        import torch
        from torch.utils.data import Dataset
    except ImportError:
        print("ERROR: pip install torch"); sys.exit(1)

    class ManiaDS(Dataset):
        def __init__(self, seqs, seq_len=SEQ_LEN):
            self.samples = []
            stride = seq_len // 2
            for feat_full, hit_lbl, ln_lbl, ln_dur, pat_types, diff_level in seqs:
                T = len(feat_full)
                for start in range(0, T - seq_len, stride):
                    end = start + seq_len
                    self.samples.append((
                        torch.tensor(feat_full[start:end],  dtype=torch.float32),
                        torch.tensor(hit_lbl[start:end],    dtype=torch.float32),
                        torch.tensor(ln_lbl[start:end],     dtype=torch.float32),
                        torch.tensor(ln_dur[start:end],     dtype=torch.float32),
                        torch.tensor(pat_types[start:end],  dtype=torch.int64),
                        torch.tensor(diff_level,             dtype=torch.long),
                    ))

        def __len__(self):  return len(self.samples)
        def __getitem__(self, i): return self.samples[i]

    return ManiaDS(sequences)


# ── Scanner ───────────────────────────────────────────────────────────────────

def scan_all_maps(maps_dir, max_maps):
    found = []
    for root, _, files in os.walk(maps_dir):
        for f in files:
            if f.endswith(".osu"):
                found.append(os.path.join(root, f))
    if max_maps:
        found = found[:max_maps]
    print(f"  Found {len(found)} .osu files")
    return found


# ── Training ──────────────────────────────────────────────────────────────────

def train(maps_dir, out_path, max_maps, epochs):
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
    except ImportError:
        print("ERROR: pip install torch"); sys.exit(1)

    try:
        from tqdm import tqdm
        _tqdm = tqdm
    except ImportError:
        _tqdm = lambda x, **kw: x

    print(f"\nScanning: {maps_dir}")
    candidates = scan_all_maps(maps_dir, max_maps)
    if not candidates:
        print("ERROR: No .osu files found."); sys.exit(1)
    print(f"  Total candidates: {len(candidates)}\n")

    raw_sequences = []
    maps_tried    = 0
    diff_counts   = [0, 0, 0, 0]
    total_ln      = 0
    total_reg     = 0

    for osu_path in _tqdm(candidates, desc="Extracting features"):
        maps_tried += 1
        result = parse_osu(osu_path)
        if result is None:
            continue

        audio_filename, groups, version_str = result
        osu_dir    = os.path.dirname(osu_path)
        audio_path = os.path.join(osu_dir, audio_filename)

        if not os.path.isfile(audio_path):
            base = os.path.splitext(audio_filename)[0]
            for ext in (".mp3", ".ogg", ".wav", ".flac"):
                alt = os.path.join(osu_dir, base + ext)
                if os.path.isfile(alt):
                    audio_path = alt
                    break
            else:
                continue

        out = extract_features_and_labels(audio_path, groups, version_str)
        feat_audio, hit_lbl, ln_lbl, ln_dur, pat_types, diff_level, ap = out
        if feat_audio is None:
            continue

        # Collect LN stats
        ln_notes  = int(ln_lbl.sum())
        reg_notes = int(hit_lbl.sum()) - ln_notes
        total_ln  += ln_notes
        total_reg += max(0, reg_notes)

        raw_sequences.append((feat_audio, hit_lbl, ln_lbl, ln_dur, pat_types, diff_level, ap))
        diff_counts[diff_level] += 1

    print(f"\nUsable maps : {len(raw_sequences)} / {maps_tried}")
    print(f"  Easy:{diff_counts[0]}  Normal:{diff_counts[1]}  "
          f"Hard:{diff_counts[2]}  Insane:{diff_counts[3]}")
    print(f"  LN notes: {total_ln}  |  Regular notes: {total_reg}  "
          f"|  LN ratio: {total_ln / max(total_ln + total_reg, 1):.3f}")

    if len(raw_sequences) < 5:
        print("ERROR: Not enough usable maps."); sys.exit(1)

    # Build cross-difficulty context features
    print("\nBuilding cross-difficulty context...")
    sequences = build_cross_diff_context(raw_sequences)

    dataset = make_dataset(sequences)
    loader  = DataLoader(dataset, batch_size=16, shuffle=True,
                         num_workers=0, drop_last=True)
    print(f"Dataset     : {len(dataset)} sequences x {SEQ_LEN} steps\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model()
    model.to(device)
    params = sum(p.numel() for p in model.parameters())

    print(f"Model       : ManiaTransformerV4  |  device: {device}  |  params: {params:,}")
    print(f"Features    :")
    print(f"  mel({N_MEL}) + onset(1) + onset_harm(1) + onset_perc(1)")
    print(f"  + SC({N_SC}) + chroma({N_CHROMA}) + rms(1) + flatness(1)")
    print(f"  + centroid(1) + zcr(1) + phases_4scales(8)")
    print(f"  = {FEAT_DIM_AUDIO} audio  +  {FEAT_DIM_CTX} ctx  =  {FEAT_DIM} total")
    print(f"Grid        : SUBDIV={SUBDIV}  |  4 difficulty heads  |  {NUM_PATTERNS} pattern classes")
    print(f"Hyperparams : d_model={D_MODEL}  nhead={NHEAD}  layers={N_LAYERS}  "
          f"ff={DIM_FF}  dropout={DROPOUT}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=2e-4)

    def lr_lambda(ep):
        if ep < 3:
            return (ep + 1) / 3.0
        return 0.5 * (1.0 + np.cos(np.pi * (ep - 3) / max(epochs - 3, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    SIBLING_WEIGHT = 0.2

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in loader:
            X, y_hit, y_ln, y_ln_dur, y_pattern, diff = batch
            X        = X.to(device)
            y_hit    = y_hit.to(device)
            y_ln     = y_ln.to(device)
            y_ln_dur = y_ln_dur.to(device)
            y_pattern = y_pattern.to(device)
            diff     = diff.to(device)

            optimizer.zero_grad()
            all_heads, pattern_logits = model(X, diff)

            loss = torch.tensor(0.0, device=device)

            # ── Per-difficulty losses ──────────────────────────────────────────
            for d in range(DIFF_LEVELS):
                mask = (diff == d)
                if not mask.any():
                    continue

                hit_logits, ln_logits, ln_dur = all_heads[d]

                # 1. Hit placement loss with dynamic pos_weight
                note_mean = y_hit[mask].mean().item()
                pw_val    = min((1.0 - note_mean) / (note_mean + 1e-6), 15.0)
                pw_val    = max(pw_val, 2.0)
                pos_weight = torch.tensor([pw_val] * 4, device=device)

                loss = loss + F.binary_cross_entropy_with_logits(
                    hit_logits[mask], y_hit[mask],
                    pos_weight=pos_weight,
                )

                # 2. LN classification loss on actual hit positions
                hit_pos = y_hit[mask] > 0.5
                if hit_pos.any():
                    loss = loss + 0.4 * F.binary_cross_entropy_with_logits(
                        ln_logits[mask][hit_pos],
                        y_ln[mask][hit_pos],
                    )

                # 3. LN duration regression on actual LN positions
                ln_pos = y_ln[mask] > 0.5
                if ln_pos.any():
                    loss = loss + 0.15 * F.mse_loss(
                        ln_dur[mask][ln_pos],
                        y_ln_dur[mask][ln_pos],
                    )

            # 4. Pattern type auxiliary loss (all samples)
            loss = loss + 0.15 * F.cross_entropy(
                pattern_logits.reshape(-1, NUM_PATTERNS),
                y_pattern.reshape(-1),
            )

            # 5. Sibling diff auxiliary (hit loss only, weight=0.2)
            if SIBLING_WEIGHT > 0:
                for d in range(DIFF_LEVELS):
                    mask = (diff == d)
                    if not mask.any():
                        continue
                    note_mean  = y_hit[mask].mean().item()
                    pw_val     = min((1.0 - note_mean) / (note_mean + 1e-6), 15.0)
                    pw_val     = max(pw_val, 2.0)
                    pos_weight = torch.tensor([pw_val] * 4, device=device)
                    for d2 in range(DIFF_LEVELS):
                        if d2 == d:
                            continue
                        hit_logits2, _, _ = all_heads[d2]
                        loss = loss + SIBLING_WEIGHT * F.binary_cross_entropy_with_logits(
                            hit_logits2[mask], y_hit[mask],
                            pos_weight=pos_weight,
                        )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch:3d}/{epochs}  loss: {avg_loss:.4f}  "
              f"lr: {optimizer.param_groups[0]['lr']:.2e}")

    torch.save({
        "model_type":     "ManiaTransformerV4",
        "feat_dim":       FEAT_DIM,
        "feat_dim_audio": FEAT_DIM_AUDIO,
        "feat_dim_ctx":   FEAT_DIM_CTX,
        "n_mel":          N_MEL,
        "n_sc":           N_SC,
        "n_chroma":       N_CHROMA,
        "subdiv":         SUBDIV,
        "d_model":        D_MODEL,
        "nhead":          NHEAD,
        "num_layers":     N_LAYERS,
        "dim_ff":         DIM_FF,
        "diff_levels":    DIFF_LEVELS,
        "diff_emb":       DIFF_EMB,
        "dropout":        DROPOUT,
        "max_ln_beats":   MAX_LN_BEATS,
        "model_state":    model.state_dict(),
        "maps_trained":   len(sequences),
    }, out_path)

    print(f"\nModel saved -> {out_path}")
    print(f"  Maps : {len(sequences)}   Sequences : {len(dataset)}")
    print(f'\nNext: python ManiaMapper.py --ui')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train osu!mania 4K Transformer v4")
    parser.add_argument("--maps-dir", default=DEFAULT_MAPS_DIR)
    parser.add_argument("--out",      default=DEFAULT_OUT)
    parser.add_argument("--max-maps", type=int, default=3000)
    parser.add_argument("--epochs",   type=int, default=50)
    args = parser.parse_args()

    if not os.path.isdir(args.maps_dir):
        print(f"ERROR: Directory not found: {args.maps_dir}"); sys.exit(1)

    print(f"Maps dir : {args.maps_dir}")
    print(f"Output   : {args.out}")
    print(f"Max maps : {args.max_maps}  |  Epochs : {args.epochs}")
    train(args.maps_dir, args.out, args.max_maps, args.epochs)


if __name__ == "__main__":
    main()
