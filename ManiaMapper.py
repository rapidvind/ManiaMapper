"""
ManiaMapper.py
Generates a 4K osu!mania beatmap from an audio file.

Two-stage generation:
  Stage 1 — Segment Classifier: splits the audio into sections and uses the
             model's style_head to decide what pattern type each section
             sounds like. The model determines how many sections and where
             boundaries fall, based on its training.
  Stage 2 — Style-Conditioned Generation: runs the LSTM per segment with the
             classified style_id embedded, then applies style-specific
             parameter modulation (chord density, LN rate, spacing) per position.
             Style params are blended smoothly at boundaries for seamless feel.

Usage:
    python ManiaMapper.py song.mp3
    python ManiaMapper.py song.mp3 --output my_map.osu
    python ManiaMapper.py song.mp3 --nn mania_style_model.pt
"""

import os, sys, random, argparse, zipfile, json
import numpy as np

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

KEYS  = 4
COL_X = [64, 192, 320, 448]   # x-positions for columns 0-3 in 4K

DIFFICULTY_PRESETS = {
    "Easy":   {"hp": 6, "od": 6,  "subdiv": 2, "keep": 0.40, "chord": 0.05, "ln": 0.20, "chord_scale": 0.25, "breathe": 2.0, "triplets": False},
    "Normal": {"hp": 7, "od": 7,  "subdiv": 4, "keep": 0.55, "chord": 0.08, "ln": 0.15, "chord_scale": 0.40, "breathe": 1.5, "triplets": False},
    "Hard":   {"hp": 8, "od": 8,  "subdiv": 8, "keep": 0.70, "chord": 0.12, "ln": 0.10, "chord_scale": 0.55, "breathe": 1.2, "triplets": True},
    "Insane": {"hp": 9, "od": 9,  "subdiv": 8, "keep": 1.0,  "chord": 0.30, "ln": 0.05, "chord_scale": 1.00, "breathe": 2.0, "triplets": True},
}

# ─── STYLE DEFINITIONS ────────────────────────────────────────────────────────

STYLE_CLASSES = [
    "tech",
    "streams",
    "complex_streams",
    "jump_streams",
    "hand_streams",
    "ln",
    "complex_ln",
    "ranked_allround",
    "chord_jacks",
    "chord_jacks_doubles",
]

# How each style modulates base generation parameters per segment.
#   chord_mult  : multiplier on chord probability
#   ln_mult     : multiplier on LN probability
#   breathe_mult: multiplier on column cooldown (higher = more spacing)
#   keep_mult   : multiplier on note density
STYLE_PATTERN_PARAMS = {
    "tech":                {"chord_mult": 1.2,  "ln_mult": 0.7,   "breathe_mult": 1.0,  "keep_mult": 0.9},
    "streams":             {"chord_mult": 0.2,  "ln_mult": 0.05,  "breathe_mult": 0.7,  "keep_mult": 1.3},
    "complex_streams":     {"chord_mult": 0.4,  "ln_mult": 0.05,  "breathe_mult": 0.7,  "keep_mult": 1.2},
    "jump_streams":        {"chord_mult": 0.7,  "ln_mult": 0.05,  "breathe_mult": 0.8,  "keep_mult": 1.2},
    "hand_streams":        {"chord_mult": 1.0,  "ln_mult": 0.05,  "breathe_mult": 0.8,  "keep_mult": 1.1},
    "ln":                  {"chord_mult": 0.3,  "ln_mult": 3.5,   "breathe_mult": 1.6,  "keep_mult": 0.7},
    "complex_ln":          {"chord_mult": 0.5,  "ln_mult": 2.5,   "breathe_mult": 1.4,  "keep_mult": 0.8},
    "ranked_allround":     {"chord_mult": 1.0,  "ln_mult": 1.0,   "breathe_mult": 1.0,  "keep_mult": 1.0},
    "chord_jacks":         {"chord_mult": 2.5,  "ln_mult": 0.05,  "breathe_mult": 0.55, "keep_mult": 1.0},
    "chord_jacks_doubles": {"chord_mult": 3.5,  "ln_mult": 0.05,  "breathe_mult": 0.45, "keep_mult": 1.0},
}
_DEFAULT_STYLE_PARAMS = {"chord_mult": 1.0, "ln_mult": 1.0, "breathe_mult": 1.0, "keep_mult": 1.0}

# Segment classifier settings — the model decides how many segments
_CLASSIFY_WIN_BEATS  = 4    # beats per classification window (rolling)
_CLASSIFY_STRIDE     = 2    # stride between windows in beats
_MIN_SEG_BEATS       = 8    # short segments below this are merged into neighbours
_TRANSITION_FACTOR   = 2    # style params blend over this many beats at boundaries

# Hand layout: left = cols 0,1 / right = cols 2,3
_LEFT  = frozenset({0, 1})
_RIGHT = frozenset({2, 3})

def _same_hand(c1: int, c2: int) -> bool:
    return (c1 in _LEFT and c2 in _LEFT) or (c1 in _RIGHT and c2 in _RIGHT)


# ─── INTERACTIVE PROMPTS ──────────────────────────────────────────────────────

def ask(prompt, choices=None, default=None):
    while True:
        if choices:
            print(f"\n{prompt}")
            for i, c in enumerate(choices, 1):
                marker = " (recommended)" if i == 1 else ""
                print(f"  [{i}] {c}{marker}")
            val = input("> ").strip()
        else:
            val = input(f"{prompt}: ").strip()

        if not val and default is not None:
            return default

        if choices:
            if val.isdigit() and 1 <= int(val) <= len(choices):
                return choices[int(val) - 1]
            matches = [c for c in choices if c.lower().startswith(val.lower())]
            if len(matches) == 1:
                return matches[0]
            print(f"  Enter a number 1-{len(choices)}")
        else:
            if val:
                return val
            if default is not None:
                return default


def get_user_settings(audio_path):
    filename = os.path.splitext(os.path.basename(audio_path))[0]

    print("\n========================================")
    print("       osu!mania Map Generator 4K")
    print("========================================\n")

    title  = ask(f"Song title  (default: {filename})", default=filename)
    artist = ask("Artist name (default: Unknown)",     default="Unknown")

    diff = ask(
        "Difficulty?",
        choices=["Easy", "Normal", "Hard", "Insane"],
    )

    sv_raw = ask("Speed Variation / SV? (y/n, default n)", default="n")
    sv     = sv_raw.lower() in ("y", "yes")

    print(f"\n  Diff: {diff}  |  SV: {'Yes' if sv else 'No'}\n")
    return {"title": title, "artist": artist, "difficulty": diff, "sv": sv}


# ─── AUDIO ANALYSIS ──────────────────────────────────────────────────────────

def analyze_audio(audio_path):
    try:
        import librosa
    except ImportError:
        print("ERROR: librosa not installed. Run: pip install librosa")
        sys.exit(1)

    print("[1/4] Analysing audio...")
    y, sr = librosa.load(audio_path, sr=None)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr) * 1000

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units="frames",
                                              hop_length=512, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr) * 1000

    hop = 512
    rms       = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop) * 1000

    S     = np.abs(librosa.stft(y, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr)
    def _norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-9)

    bass_norm = _norm(S[freqs < 300, :].mean(axis=0))
    mid_norm  = _norm(S[(freqs >= 300) & (freqs < 3000), :].mean(axis=0))
    high_norm = _norm(S[freqs >= 3000, :].mean(axis=0))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop)
    mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-9)

    bpm_orig    = float(np.atleast_1d(tempo)[0])
    bpm         = bpm_orig * 2
    beat_length = 60000.0 / bpm
    duration_ms = len(y) / sr * 1000

    print(f"   BPM: {bpm_orig:.1f} (×2={bpm:.1f})  |  Beats: {len(beat_times)}  |  Onsets: {len(onset_times)}")
    return {
        "bpm": bpm, "bpm_orig": bpm_orig,
        "beat_length": beat_length, "beat_length_orig": 60000.0 / bpm_orig,
        "beat_times": beat_times, "onset_times": onset_times,
        "rms": rms, "rms_times": rms_times,
        "bass_norm": bass_norm, "mid_norm": mid_norm, "high_norm": high_norm,
        "mfcc": mfcc,
        "duration_ms": duration_ms,
    }


# ─── BEAT GRID ───────────────────────────────────────────────────────────────

def build_beat_grid(audio_data, subdiv, triplets=False):
    beat_times  = audio_data["beat_times"]
    beat_length = audio_data["beat_length"]
    onset_times = audio_data["onset_times"]
    duration    = audio_data["duration_ms"]

    step = beat_length / subdiv
    seen = set()
    grid = []

    def add_slot(t, strength):
        t_r = round(t)
        if t_r in seen or t_r > duration:
            return
        seen.add(t_r)
        if any(abs(t_r - o) < step * 0.55 for o in onset_times):
            strength = min(1.0, strength * 1.6)
        grid.append((t_r, strength))

    for i, t_beat in enumerate(beat_times):
        t_next      = beat_times[i + 1] if i + 1 < len(beat_times) else t_beat + beat_length
        actual_step = (t_next - t_beat) / subdiv

        for s in range(subdiv):
            t = t_beat + s * actual_step
            if s == 0:
                strength = 1.0
            elif s == subdiv // 2:
                strength = 0.6
            elif s % 2 == 0:
                strength = 0.35
            else:
                strength = 0.20
            add_slot(t, strength)

        if triplets:
            beat_interval = t_next - t_beat
            add_slot(t_beat + beat_interval / 3.0, 0.18)
            add_slot(t_beat + beat_interval * 2.0 / 3.0, 0.18)

    grid.sort()
    return grid


# ─── ML MODEL HELPERS ─────────────────────────────────────────────────────────

def load_model(model_path: str) -> dict:
    with open(model_path, "r", encoding="utf-8") as f:
        return json.load(f)


def sample_weighted(counts: dict, allowed: list = None):
    if not counts:
        return None
    filtered = {}
    for k, v in counts.items():
        ki = int(k) if isinstance(k, str) else k
        if allowed is None or ki in allowed:
            filtered[ki] = filtered.get(ki, 0) + v
    if not filtered:
        return None
    keys  = list(filtered.keys())
    total = sum(filtered.values())
    r     = random.random() * total
    cumul = 0
    for k in keys:
        cumul += filtered[k]
        if r <= cumul:
            return k
    return keys[-1]


def sample_chord_combo(chord_combos: dict, free_cols: list):
    filtered = {}
    for combo_str, count in chord_combos.items():
        c1, c2 = map(int, combo_str.split(","))
        if c1 in free_cols and c2 in free_cols:
            filtered[combo_str] = count
    if not filtered:
        return None
    total = sum(filtered.values())
    r     = random.random() * total
    cumul = 0
    for k, v in filtered.items():
        cumul += v
        if r <= cumul:
            c1, c2 = map(int, k.split(","))
            return [c1, c2]
    k = list(filtered.keys())[-1]
    c1, c2 = map(int, k.split(","))
    return [c1, c2]


# ─── LSTM MODEL LOADING ───────────────────────────────────────────────────────

_NN_N_MFCC   = 20
_NN_FEAT_DIM = _NN_N_MFCC + 5
_NN_SUBDIV   = 4


def load_nn_model(model_path: str):
    """
    Load the full ManiaStyleLSTM trained by ManiaNNTrainer.py.
    Both note_head (placement) and style_head (segment classification) are loaded.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None

    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

        num_styles = len(ckpt.get("style_classes", STYLE_CLASSES))
        feat_dim   = ckpt.get("feat_dim",   _NN_FEAT_DIM)
        hidden_dim = ckpt.get("hidden_dim", 256)
        num_layers = ckpt.get("num_layers", 3)
        style_dim  = ckpt.get("style_dim",  16)

        class ManiaStyleLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.style_embed = nn.Embedding(num_styles, style_dim)
                self.lstm        = nn.LSTM(feat_dim + style_dim, hidden_dim, num_layers,
                                           batch_first=True, dropout=0.3)
                self.norm        = nn.LayerNorm(hidden_dim)
                self.note_head   = nn.Linear(hidden_dim, 4)
                self.style_head  = nn.Linear(hidden_dim, num_styles)

            def forward(self, x, style_ids):
                """
                x          : (B, T, feat_dim)
                style_ids  : (B,) — style index; use 0 for pure audio-driven classification
                Returns note_logits (B,T,4) and style_logits (B, num_styles)
                """
                emb    = self.style_embed(style_ids)
                emb_t  = emb.unsqueeze(1).expand(-1, x.size(1), -1)
                inp    = torch.cat([x, emb_t], dim=-1)
                out, _ = self.lstm(inp)
                out    = self.norm(out)
                return self.note_head(out), self.style_head(out.mean(dim=1))

        model = ManiaStyleLSTM()
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return {"model": model, "meta": ckpt}
    except Exception as e:
        print(f"   [warn] Could not load NN model: {e}")
        return None


# ─── STAGE 1: MODEL-DRIVEN SEGMENT CLASSIFICATION ────────────────────────────

def segment_and_classify(audio_data, nn_model_data):
    """
    Stage 1 — the model decides how many segments and what style each one is.

    Algorithm:
      1. Run the model's style_head on overlapping _CLASSIFY_WIN_BEATS windows
         (stride = _CLASSIFY_STRIDE beats) with zero style embedding so the
         classification is driven purely by audio.
      2. Accumulate per-beat style probability votes across all windows that
         cover each beat.
      3. Smooth per-beat style assignments with a majority-vote filter (±2 beats)
         to remove single-beat noise.
      4. Detect boundaries where the smoothed style changes.
      5. Merge segments shorter than _MIN_SEG_BEATS into their neighbours so
         the structure stays coherent.

    Returns list of (start_ms, end_ms, style_id, style_name).
    The number of segments is fully determined by the model's output.
    """
    import torch

    model         = nn_model_data["model"]
    meta          = nn_model_data["meta"]
    style_classes = meta.get("style_classes", STYLE_CLASSES)
    n_styles      = len(style_classes)

    beat_times    = audio_data["beat_times"]
    beat_len_orig = audio_data["beat_length_orig"]
    beat_len      = audio_data["beat_length"]
    mfcc          = audio_data["mfcc"]
    bass_norm     = audio_data["bass_norm"]
    mid_norm      = audio_data["mid_norm"]
    high_norm     = audio_data["high_norm"]
    rms_times     = audio_data["rms_times"]
    duration      = audio_data["duration_ms"]
    n_frames      = mfcc.shape[1]
    step_ms       = beat_len / _NN_SUBDIV

    def _feat_at(t_ms):
        idx = min(int(np.searchsorted(rms_times, t_ms)), n_frames - 1)
        f   = np.zeros(_NN_FEAT_DIM, dtype=np.float32)
        f[:_NN_N_MFCC]    = mfcc[:, idx]
        f[_NN_N_MFCC]     = bass_norm[idx]
        f[_NN_N_MFCC + 1] = mid_norm[idx]
        f[_NN_N_MFCC + 2] = high_norm[idx]
        phase              = (t_ms % (beat_len_orig * 4)) / (beat_len_orig * 4 + 1e-9)
        f[_NN_N_MFCC + 3] = float(np.sin(2 * np.pi * phase))
        f[_NN_N_MFCC + 4] = float(np.cos(2 * np.pi * phase))
        return f

    device    = next(model.parameters()).device
    num_beats = len(beat_times)

    # Fallback for very short audio
    if num_beats < _CLASSIFY_WIN_BEATS:
        default_id   = style_classes.index("ranked_allround") if "ranked_allround" in style_classes else 0
        default_name = style_classes[default_id]
        return [(0.0, duration, default_id, default_name)]

    # ── Step 1 & 2: sliding window voting ─────────────────────────────────────
    beat_votes = np.zeros((num_beats, n_styles), dtype=np.float32)

    for i in range(0, num_beats, _CLASSIFY_STRIDE):
        win_end_i = min(i + _CLASSIFY_WIN_BEATS, num_beats)
        win_beats = beat_times[i:win_end_i]
        seg_end_ms = beat_times[win_end_i] if win_end_i < num_beats else duration

        positions = []
        for t_beat in win_beats:
            for s in range(_NN_SUBDIV):
                t = t_beat + s * step_ms
                if t >= seg_end_ms:
                    break
                positions.append(t)

        if len(positions) < 4:
            continue

        feats = np.stack([_feat_at(t) for t in positions])
        X     = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
        dummy = torch.zeros(1, dtype=torch.long, device=device)

        with torch.no_grad():
            _, style_logits = model(X, dummy)
            probs = torch.softmax(style_logits[0], dim=0).cpu().numpy()

        for bi in range(i, win_end_i):
            beat_votes[bi] += probs

    # Per-beat: argmax of accumulated votes
    beat_style_ids = []
    for bi in range(num_beats):
        if beat_votes[bi].sum() > 0:
            beat_style_ids.append(int(np.argmax(beat_votes[bi])))
        else:
            beat_style_ids.append(beat_style_ids[-1] if beat_style_ids else 0)

    # ── Step 3: smooth with majority vote over ±2 beats ───────────────────────
    smoothed = []
    for i in range(num_beats):
        window = beat_style_ids[max(0, i - 2): i + 3]
        counts = {}
        for s in window:
            counts[s] = counts.get(s, 0) + 1
        smoothed.append(max(counts, key=counts.get))

    # ── Step 4: detect boundaries ─────────────────────────────────────────────
    raw_segs = []
    cur_style = smoothed[0]
    cur_start = 0
    for i in range(1, num_beats):
        if smoothed[i] != cur_style:
            raw_segs.append((cur_start, i, cur_style))
            cur_style = smoothed[i]
            cur_start = i
    raw_segs.append((cur_start, num_beats, cur_style))

    # ── Step 5: merge short segments into neighbours ──────────────────────────
    merged = []
    for bi_start, bi_end, style_id in raw_segs:
        length = bi_end - bi_start
        if merged and length < _MIN_SEG_BEATS:
            prev = merged[-1]
            merged[-1] = (prev[0], bi_end, prev[2])   # extend previous, keep its style
        else:
            merged.append((bi_start, bi_end, style_id))

    # Convert beat indices → ms and build final result
    result = []
    for bi_start, bi_end, style_id in merged:
        start_ms   = float(beat_times[bi_start])
        end_ms     = float(beat_times[bi_end]) if bi_end < num_beats else duration
        style_name = style_classes[style_id] if style_id < n_styles else "ranked_allround"
        result.append((start_ms, end_ms, style_id, style_name))

    return result


def _style_params_at(t_ms, segment_map, beat_length_orig):
    """
    Return style generation parameters for time t_ms.

    Within _TRANSITION_FACTOR beats of a segment boundary, parameters are
    linearly interpolated between the two adjacent segments so the map
    transitions smoothly rather than snapping abruptly.
    """
    if not segment_map:
        return _DEFAULT_STYLE_PARAMS

    transition_ms = beat_length_orig * _TRANSITION_FACTOR

    for k, (start_ms, end_ms, _, style_name) in enumerate(segment_map):
        if start_ms <= t_ms < end_ms:
            cur_p = STYLE_PATTERN_PARAMS.get(style_name, _DEFAULT_STYLE_PARAMS)

            # Blend with previous segment near the start boundary
            dist_start = t_ms - start_ms
            if k > 0 and dist_start < transition_ms:
                alpha   = dist_start / transition_ms          # 0 → 1
                prev_p  = STYLE_PATTERN_PARAMS.get(segment_map[k - 1][3], _DEFAULT_STYLE_PARAMS)
                return {key: prev_p[key] * (1.0 - alpha) + cur_p[key] * alpha
                        for key in cur_p}

            # Blend with next segment near the end boundary
            dist_end = end_ms - t_ms
            if k < len(segment_map) - 1 and dist_end < transition_ms:
                alpha   = dist_end / transition_ms            # 1 → 0
                next_p  = STYLE_PATTERN_PARAMS.get(segment_map[k + 1][3], _DEFAULT_STYLE_PARAMS)
                return {key: cur_p[key] * alpha + next_p[key] * (1.0 - alpha)
                        for key in cur_p}

            return cur_p

    # Past the last segment end
    return STYLE_PATTERN_PARAMS.get(segment_map[-1][3], _DEFAULT_STYLE_PARAMS)


# ─── STAGE 2: STYLE-CONDITIONED NOTE GENERATION ──────────────────────────────

def assign_columns_nn(audio_data, nn_model_data, keep, ln_chance, chord_chance=0.15,
                      breathe=1.8, segment_map=None):
    """
    Stage 2 — style-conditioned note generation.

    For each segment from Stage 1:
      - Runs the LSTM forward pass with that segment's style_id embedded so
        the note_head generates probabilities appropriate for that pattern type.
      - Per-position style params (chord density, LN rate, breathe, density)
        are fetched from _style_params_at, which interpolates at boundaries
        for seamless transitions.
    """
    try:
        import torch
    except ImportError:
        return None

    model      = nn_model_data["model"]
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    beat_times    = audio_data["beat_times"]
    beat_len_orig = audio_data["beat_length_orig"]
    beat_len      = audio_data["beat_length"]
    mfcc          = audio_data["mfcc"]
    bass_norm     = audio_data["bass_norm"]
    mid_norm      = audio_data["mid_norm"]
    high_norm     = audio_data["high_norm"]
    rms           = audio_data["rms"]
    rms_times     = audio_data["rms_times"]
    rms_norm      = (rms - rms.min()) / (rms.max() - rms.min() + 1e-6)
    duration      = audio_data["duration_ms"]
    n_frames      = mfcc.shape[1]
    step_ms       = beat_len / _NN_SUBDIV
    min_gap       = 80.0

    def _feat_at(t_ms: float) -> np.ndarray:
        idx = min(int(np.searchsorted(rms_times, t_ms)), n_frames - 1)
        f   = np.zeros(_NN_FEAT_DIM, dtype=np.float32)
        f[:_NN_N_MFCC]    = mfcc[:, idx]
        f[_NN_N_MFCC]     = bass_norm[idx]
        f[_NN_N_MFCC + 1] = mid_norm[idx]
        f[_NN_N_MFCC + 2] = high_norm[idx]
        phase              = (t_ms % (beat_len_orig * 4)) / (beat_len_orig * 4 + 1e-9)
        f[_NN_N_MFCC + 3] = float(np.sin(2 * np.pi * phase))
        f[_NN_N_MFCC + 4] = float(np.cos(2 * np.pi * phase))
        return f

    # Build full position list
    positions = []
    for t_beat in beat_times:
        for s in range(_NN_SUBDIV * 2):
            t = t_beat + s * step_ms
            if t > duration:
                break
            positions.append(t)

    if not positions:
        return []

    features = np.stack([_feat_at(t) for t in positions])   # (T, F)

    # ── Style-conditioned LSTM forward pass per segment ───────────────────────
    # Each segment's positions are processed with its own style_id embedded.
    # This conditions the note_head on the correct pattern type for that section.
    all_probs = np.zeros((len(positions), 4), dtype=np.float32)

    if segment_map:
        for seg_start_ms, seg_end_ms, seg_style_id, _ in segment_map:
            seg_idx = [i for i, t in enumerate(positions)
                       if seg_start_ms <= t < seg_end_ms]
            if not seg_idx:
                continue
            seg_feats = np.stack([features[i] for i in seg_idx])
            X_seg = torch.tensor(seg_feats, dtype=torch.float32).unsqueeze(0).to(device)
            sid   = torch.tensor([seg_style_id], dtype=torch.long, device=device)
            with torch.no_grad():
                note_logits, _ = model(X_seg, sid)
                seg_probs = torch.sigmoid(note_logits[0]).cpu().numpy()
            for local_i, global_i in enumerate(seg_idx):
                all_probs[global_i] = seg_probs[local_i]

        # Fallback for any uncovered positions (e.g. past last segment end)
        uncovered = [i for i, t in enumerate(positions)
                     if not any(s <= t < e for s, e, _, _ in segment_map)]
        if uncovered:
            unc_feats = np.stack([features[i] for i in uncovered])
            X_unc = torch.tensor(unc_feats, dtype=torch.float32).unsqueeze(0).to(device)
            dummy = torch.zeros(1, dtype=torch.long, device=device)
            with torch.no_grad():
                note_logits, _ = model(X_unc, dummy)
                unc_probs = torch.sigmoid(note_logits[0]).cpu().numpy()
            for local_i, global_i in enumerate(uncovered):
                all_probs[global_i] = unc_probs[local_i]
    else:
        X     = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        dummy = torch.zeros(1, dtype=torch.long, device=device)
        with torch.no_grad():
            note_logits, _ = model(X, dummy)
            all_probs = torch.sigmoid(note_logits[0]).cpu().numpy()

    # Smooth rms over ~300ms window
    win = max(1, int(300.0 / ((rms_times[1] - rms_times[0]) if len(rms_times) > 1 else 1.0)))
    kernel       = np.ones(win) / win
    rms_smooth   = np.convolve(rms, kernel, mode='same')
    rsm_s_min, rsm_s_max = rms_smooth.min(), rms_smooth.max()
    rms_smooth_n = (rms_smooth - rsm_s_min) / (rsm_s_max - rsm_s_min + 1e-9)

    rms_norm_arr     = rms_norm
    trill_energy_thr = float(np.percentile(rms_norm_arr, 70))

    bass_hit_thr    = float(np.percentile(bass_norm, 70))
    bass_peak_times = []
    for k in range(1, len(bass_norm) - 1):
        if (bass_norm[k] > bass_hit_thr
                and bass_norm[k] >= bass_norm[k - 1]
                and bass_norm[k] >= bass_norm[k + 1]):
            bass_peak_times.append(float(rms_times[k]))
    bass_peak_arr = np.array(bass_peak_times) if bass_peak_times else np.array([])

    TRILL_PATTERNS = [
        ([0,1], [2,3]),
        ([2,3], [0,1]),
        ([0,3], [1,2]),
        ([0], [1], [2], [3]),
        ([3], [2], [1], [0]),
        ([0], [1], [2], [3], [2], [1]),
        ([0], [2], [0], [2]),
        ([1], [3], [1], [3]),
        ([0], [1], [0], [1]),
        ([2], [3], [2], [3]),
        ([0], [2], [1], [3]),
    ]

    notes           = []
    col_busy        = {}
    last_col        = -1
    prev_col        = -1
    trill_active    = False
    trill_type      = 0
    trill_stair_pos = 0
    trill_remaining = 0
    last_kick_t     = -9999.0
    kick_streak     = 0
    jack_cols       = None

    for i, t in enumerate(positions):
        idx         = min(int(np.searchsorted(rms_times, t)), len(rms_norm_arr) - 1)
        energy      = float(rms_norm_arr[idx])
        energy_s    = float(rms_smooth_n[idx])
        bass_energy = float(bass_norm[idx])

        # Fetch blended style params for this exact position
        sp = _style_params_at(t, segment_map, beat_len_orig)

        song_progress = t / max(duration, 1.0)
        if song_progress < 0.12:
            intro_scale = song_progress / 0.12
        elif song_progress > 0.88:
            intro_scale = (1.0 - song_progress) / 0.12
        else:
            intro_scale = 1.0

        on_bass_hit = (len(bass_peak_arr) > 0
                       and float(np.min(np.abs(bass_peak_arr - t))) < step_ms * 0.6)

        local_keep = keep * sp["keep_mult"] * (0.35 + 0.65 * energy_s) * intro_scale
        if not (on_bass_hit and energy_s < 0.70):
            if random.random() > local_keep:
                continue

        # ── Trill zone ────────────────────────────────────────────────────
        if not trill_active and energy > trill_energy_thr:
            future_high = sum(
                1 for j in range(i, min(i + 12, len(positions)))
                if float(rms_norm_arr[min(int(np.searchsorted(rms_times, positions[j])),
                                         len(rms_norm_arr) - 1)]) > trill_energy_thr
            )
            if future_high >= 8:
                trill_active    = True
                trill_remaining = future_high
                trill_type      = random.randint(0, len(TRILL_PATTERNS) - 1)
                trill_stair_pos = 0

        if trill_active:
            trill_remaining -= 1
            if trill_remaining <= 0:
                trill_active = False

            pattern = TRILL_PATTERNS[trill_type]
            step    = list(pattern[trill_stair_pos % len(pattern)])
            trill_stair_pos += 1

            free = [c for c in step if col_busy.get(c, 0) <= t]
            if len(free) == len(step):
                for col in step:
                    col_busy[col] = t + min_gap
                    notes.append((round(t), col, False, 0))
                last_col = step[0] if len(step) == 1 else -1
                continue

        # ── Normal note placement ──────────────────────────────────────────
        col_probs = all_probs[i]

        def col_score(c):
            if c == last_col:
                return -999.0
            score = float(col_probs[c])
            if last_col >= 0:
                dist = abs(c - last_col)
                if dist == 1:   score += 0.40
                elif dist == 2: score += 0.10
            if c == prev_col:
                score -= 0.28
            if on_bass_hit and c in (0, 3):
                score += 0.35
            return score

        ranked = sorted(range(KEYS), key=col_score, reverse=True)
        best   = next((c for c in ranked if col_busy.get(c, 0) <= t), None)
        if best is None:
            continue

        chosen = [best]

        # ── Chord — style-modulated ────────────────────────────────────────
        effective_chord = min(0.95, chord_chance * sp["chord_mult"] + bass_energy * 0.35)
        if random.random() < effective_chord:
            free_cols = [c for c in range(KEYS) if c != best and col_busy.get(c, 0) <= t]
            if free_cols:
                second = max(free_cols, key=lambda c: col_probs[c])
                chosen.append(second)
                triple_prob = chord_chance * sp["chord_mult"] * (0.4 + 0.6 * energy)
                if random.random() < triple_prob:
                    free_cols2 = [c for c in range(KEYS)
                                  if c not in chosen and col_busy.get(c, 0) <= t]
                    if free_cols2:
                        chosen.append(max(free_cols2, key=lambda c: col_probs[c]))

        # ── Chord jack on consecutive kicks ───────────────────────────────
        in_chord_jack = False
        if on_bass_hit:
            if t - last_kick_t < beat_len_orig * 1.8:
                kick_streak += 1
            else:
                kick_streak = 1
            last_kick_t = t

            if kick_streak >= 2:
                in_chord_jack = True
                free_all = [c for c in range(KEYS) if col_busy.get(c, 0) <= t]
                if jack_cols and all(c in free_all for c in jack_cols):
                    chosen = list(jack_cols)
                elif len(free_all) >= 2:
                    chosen = sorted(random.sample(free_all, 2))
                    jack_cols = tuple(chosen)
                else:
                    in_chord_jack = False
        else:
            if t - last_kick_t > beat_len_orig * 3.0:
                kick_streak = 0
                jack_cols   = None

        if not in_chord_jack:
            chosen = sorted(chosen)

        prev_col = last_col
        last_col = chosen[0] if len(chosen) == 1 else -1

        local_breathe = breathe * sp["breathe_mult"]

        # ── LN — style-modulated ──────────────────────────────────────────
        if in_chord_jack:
            for col in chosen:
                col_busy[col] = t + min_gap * local_breathe
                notes.append((round(t), col, False, 0))
            continue

        local_ln = ln_chance * sp["ln_mult"]

        if energy < 0.35:
            r = random.random()
            if r < 0.45:
                ln_prob  = min(0.90, local_ln * 8.0)
                ln_beats = random.uniform(1.5, 3.5)
            elif r < 0.75:
                ln_prob  = min(0.75, local_ln * 5.0)
                ln_beats = random.uniform(0.3, 0.9)
            else:
                ln_prob  = 0.0
                ln_beats = 0.0
        elif energy < 0.65:
            ln_prob  = min(0.80, local_ln * 2.5)
            ln_beats = random.uniform(0.6, 1.5)
        else:
            if random.random() < 0.10:
                ln_prob  = min(0.35, local_ln * 1.5)
                ln_beats = random.uniform(1.0, 2.0)
            else:
                ln_prob  = min(0.60, local_ln * 2.5)
                ln_beats = random.uniform(0.15, 0.45)

        double_ln = (len(chosen) == 2 and energy < 0.65 and random.random() < 0.08)

        for col_idx, col in enumerate(chosen):
            is_ln = random.random() < ln_prob if (double_ln and col_idx < 2) \
                    else ((len(chosen) == 1) and random.random() < ln_prob)
            if is_ln:
                ln_end = round(t + beat_len_orig * ln_beats)
                col_busy[col] = ln_end + min_gap
                notes.append((round(t), col, True, ln_end))
            else:
                col_busy[col] = t + min_gap * local_breathe
                notes.append((round(t), col, False, 0))

    return notes


# ─── ML COLUMN ASSIGNMENT ─────────────────────────────────────────────────────

def assign_columns_ml(audio_data, model, subdiv, keep, ln_chance, chord_chance, breathe=1.5):
    beat_length = audio_data["beat_length"]
    beat_times  = audio_data["beat_times"]
    onset_arr   = sorted(audio_data["onset_times"])
    rms         = audio_data["rms"]
    rms_times   = audio_data["rms_times"]
    rms_norm    = (rms - rms.min()) / (rms.max() - rms.min() + 1e-6)
    bass_norm   = audio_data.get("bass_norm", rms_norm)
    high_norm   = audio_data.get("high_norm", rms_norm)
    duration    = audio_data["duration_ms"]
    min_gap     = max(beat_length / subdiv * 0.85, 80.0)

    def _at(arr, t):
        idx = min(int(np.searchsorted(rms_times, t)), len(arr) - 1)
        return float(arr[idx])

    candidates = set()
    for t in onset_arr:
        if t > duration:
            break
        e = _at(rms_norm, t)
        if random.random() < keep * (0.35 + e * 0.85):
            candidates.add(round(t))

    onset_set = set(round(t) for t in onset_arr)
    for t_beat in beat_times:
        e = _at(rms_norm, t_beat)
        if e > 0.45 and random.random() < keep * e:
            t_r = round(t_beat)
            if not any(abs(t_r - o) < 50 for o in onset_set):
                candidates.add(t_r)

    raw_times    = sorted(candidates)
    active_times = []
    last_t = -9999
    for t in raw_times:
        if t - last_t >= 80:
            active_times.append(t)
            last_t = t

    s2s        = model.get("single_to_single", {})
    c2s        = model.get("chord_to_single", {})
    combo_pool = model.get("chord_combos", {})

    notes       = []
    col_busy    = {}
    last_cols   = []
    last_single = -1
    prev_single = -1

    for t in active_times:
        free_cols = [c for c in range(KEYS) if col_busy.get(c, 0) <= t]
        if not free_cols:
            continue

        energy = _at(rms_norm, t)
        bass   = _at(bass_norm, t)
        high   = _at(high_norm, t)

        bass_bonus      = bass * chord_chance * 0.6
        effective_chord = min(0.85, chord_chance * (0.2 + energy * 0.8) + bass_bonus)

        if random.random() < effective_chord and len(free_cols) >= 2:
            if bass > 0.55 and len([c for c in free_cols if c in (0, 3)]) >= 1:
                outer = [c for c in free_cols if c in (0, 3)]
                inner = [c for c in free_cols if c in (1, 2)]
                if outer and inner:
                    chosen = sorted([random.choice(outer), random.choice(inner)])
                else:
                    pair = sample_chord_combo(combo_pool, free_cols)
                    chosen = pair if pair else sorted(random.sample(free_cols, 2))
            else:
                pair = sample_chord_combo(combo_pool, free_cols)
                chosen = pair if pair else sorted(random.sample(free_cols, 2))
            last_single = -1
            prev_single = -1
        else:
            if last_cols:
                if len(last_cols) == 1:
                    transitions = s2s.get(str(last_cols[0]), {})
                elif len(last_cols) == 2:
                    transitions = c2s.get(f"{last_cols[0]},{last_cols[1]}", {})
                else:
                    transitions = {}
            else:
                transitions = {}

            no_double = [c for c in free_cols if c != last_single] if last_single >= 0 else list(free_cols)
            if not no_double:
                no_double = list(free_cols)

            if last_single >= 0 and prev_single >= 0 and _same_hand(last_single, prev_single):
                no_aba = [c for c in no_double if c != prev_single]
                candidates_col = no_aba if no_aba else no_double
            else:
                candidates_col = no_double

            if high > bass * 1.4 and high > 0.45:
                inner = [c for c in candidates_col if c in (1, 2)]
                if inner:
                    candidates_col = inner

            col = sample_weighted(transitions, allowed=candidates_col)
            if col is None:
                col = sample_weighted(model.get("col_balance", {}), allowed=candidates_col)
            if col is None:
                col = random.choice(candidates_col)
            chosen = [col]

            prev_single = last_single
            last_single = col

        local_ln = min(0.90, ln_chance * (2.0 - energy * 1.5))

        for col in chosen:
            is_ln = (len(chosen) == 1) and random.random() < local_ln
            if is_ln:
                ln_end = round(t + beat_length * 0.85)
                col_busy[col] = ln_end + min_gap
                notes.append((t, col, True, ln_end))
            else:
                col_busy[col] = t + min_gap * breathe
                notes.append((t, col, False, 0))

        if len(chosen) > 1:
            rest_until = t + min_gap * 1.5
            for c in range(KEYS):
                if c not in chosen and col_busy.get(c, 0) < rest_until:
                    col_busy[c] = rest_until

        last_cols = sorted(chosen)

    return notes


# ─── RULE-BASED COLUMN ASSIGNMENT (fallback) ─────────────────────────────────

def assign_columns(audio_data, subdiv, keep, chord_chance, ln_chance, breathe=1.5, triplets=False):
    beat_length = audio_data["beat_length"]
    grid        = build_beat_grid(audio_data, subdiv, triplets=triplets)
    min_gap     = max(beat_length / subdiv * 0.85, 90.0)
    rms         = audio_data["rms"]
    rms_times   = audio_data["rms_times"]
    rms_norm    = (rms - rms.min()) / (rms.max() - rms.min() + 1e-6)

    raw_times = []
    for t, strength in grid:
        idx       = min(int(np.searchsorted(rms_times, t)), len(rms_norm) - 1)
        energy    = float(rms_norm[idx])
        slot_keep = min(1.0, strength * keep * (0.5 + energy * 0.8))
        if random.random() < slot_keep:
            raw_times.append(t)
    raw_times.sort()

    active_times  = []
    last_accepted = -9999
    for t in raw_times:
        if t - last_accepted >= 80:
            active_times.append(t)
            last_accepted = t

    notes       = []
    col_busy    = {}
    col_cycle   = 0
    last_single = -1
    prev_single = -1

    for t in active_times:
        free_cols = [c for c in range(KEYS) if col_busy.get(c, 0) <= t]
        if not free_cols:
            continue

        idx    = min(int(np.searchsorted(rms_times, t)), len(rms_norm) - 1)
        energy = float(rms_norm[idx])

        effective_chord = chord_chance * (0.25 + energy * 0.75)

        if random.random() < effective_chord and len(free_cols) >= 2:
            chosen = sorted(random.sample(free_cols, 2))
            last_single = -1
            prev_single = -1
        else:
            no_double = [c for c in free_cols if c != last_single] if last_single >= 0 else list(free_cols)
            if not no_double:
                no_double = list(free_cols)

            if last_single >= 0 and prev_single >= 0 and _same_hand(last_single, prev_single):
                no_aba = [c for c in no_double if c != prev_single]
                candidates = no_aba if no_aba else no_double
            else:
                candidates = no_double

            col = candidates[col_cycle % len(candidates)]
            col_cycle += 1
            chosen = [col]

            prev_single = last_single
            last_single = col

        local_ln = min(0.90, ln_chance * (2.0 - energy * 1.5))

        for col in chosen:
            is_ln = (len(chosen) == 1) and random.random() < local_ln
            if is_ln:
                ln_end = round(t + beat_length * 0.85)
                col_busy[col] = ln_end + min_gap
                notes.append((t, col, True, ln_end))
            else:
                col_busy[col] = t + min_gap * breathe
                notes.append((t, col, False, 0))

        if len(chosen) > 1:
            rest_until = t + min_gap * 1.5
            for c in range(KEYS):
                if c not in chosen and col_busy.get(c, 0) < rest_until:
                    col_busy[c] = rest_until

    return notes


# ─── SV GENERATION ───────────────────────────────────────────────────────────

def generate_sv_points(audio_data):
    rms         = audio_data["rms"]
    rms_times   = audio_data["rms_times"]
    beat_length = audio_data["beat_length"]
    start       = audio_data["beat_times"][0] if len(audio_data["beat_times"]) else 0
    duration    = audio_data["duration_ms"]

    rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-6)
    sv_points, last_sv, t = [], None, start

    while t < duration:
        idx     = min(int(np.searchsorted(rms_times, t)), len(rms_norm) - 1)
        energy  = rms_norm[idx]
        sv_mult = round((0.70 + energy * 0.60) * 4) / 4
        bpm_val = round(-100.0 / sv_mult, 2)
        if bpm_val != last_sv:
            sv_points.append((round(t), bpm_val))
            last_sv = bpm_val
        t += beat_length * 2

    return sv_points


# ─── OSU FILE WRITER ─────────────────────────────────────────────────────────

def write_osu(settings, audio_data, notes, sv_points, output_path):
    d           = DIFFICULTY_PRESETS[settings["difficulty"]]
    beat_length = audio_data["beat_length"]
    offset      = 0
    audio_file  = os.path.basename(settings["audio_path"])
    diff        = settings["difficulty"]

    lines = [
        "osu file format v14", "",
        "[General]",
        f"AudioFilename: {audio_file}",
        "AudioLeadIn: 0",
        "PreviewTime: -1",
        "Countdown: 0",
        "SampleSet: Soft",
        "Mode: 3",
        "LetterboxInBreaks: 0",
        "", "[Editor]",
        "DistanceSpacing: 1", "BeatDivisor: 4", "GridSize: 32", "TimelineZoom: 1",
        "", "[Metadata]",
        f"Title:{settings['title']}",
        f"TitleUnicode:{settings['title']}",
        f"Artist:{settings['artist']}",
        f"ArtistUnicode:{settings['artist']}",
        "Creator:ManiaMapper AI",
        f"Version:{diff} [Mix]",
        "Source:", "Tags:ai generated mania",
        "BeatmapID:0", "BeatmapSetID:-1",
        "", "[Difficulty]",
        f"HPDrainRate:{d['hp']}",
        f"CircleSize:{KEYS}",
        f"OverallDifficulty:{d['od']}",
        "ApproachRate:5",
        "SliderMultiplier:1.4",
        "SliderTickRate:1",
        "", "[Events]",
        "//Background and Video events",
        "//Break Periods",
        "", "[TimingPoints]",
        f"{offset},{beat_length:.6f},4,2,1,100,1,0",
    ]

    for sv_time, sv_val in sv_points:
        lines.append(f"{sv_time},{sv_val:.2f},4,2,1,100,0,0")

    lines += ["", "[HitObjects]"]
    for (t, col, is_ln, ln_end) in sorted(notes, key=lambda n: n[0]):
        x = COL_X[col]
        if is_ln:
            lines.append(f"{x},192,{t},128,0,{ln_end}:0:0:0:0:")
        else:
            lines.append(f"{x},192,{t},1,0,0:0:0:0:")

    osu_content = "\n".join(lines)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(osu_content)

    return len(notes), osu_content


# ─── OSZ PACKAGER ────────────────────────────────────────────────────────────

def build_osz(settings, audio_data, notes, sv_points, output_path):
    diff        = settings["difficulty"]
    safe_title  = "".join(c for c in settings["title"]  if c.isalnum() or c in " -_")
    safe_artist = "".join(c for c in settings["artist"] if c.isalnum() or c in " -_")
    audio_src   = settings["audio_path"]
    audio_name  = os.path.basename(audio_src)
    osu_name    = f"{safe_artist} - {safe_title} [{diff}].osu"

    count, osu_content = write_osu(settings, audio_data, notes, sv_points, None)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(osu_name, osu_content.encode("utf-8"))
        zf.write(audio_src, audio_name)

    return count


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="4K osu!mania map generator")
    parser.add_argument("audio",    help="Audio file (mp3, ogg, wav, flac)")
    parser.add_argument("--output", "-o", help="Output .osz path")
    parser.add_argument("--nn",           help="Path to mania_style_model.pt (LSTM)")
    args = parser.parse_args()

    if not os.path.isfile(args.audio):
        print(f"ERROR: File not found: {args.audio}")
        sys.exit(1)

    scripts_dir   = os.path.dirname(os.path.abspath(__file__))
    nn_model_path = args.nn or os.path.join(scripts_dir, "mania_style_model.pt")

    if not os.path.isfile(nn_model_path):
        print(f"ERROR: LSTM model not found at {nn_model_path}")
        print("Run ManiaNNTrainer.py first to train the model.")
        sys.exit(1)

    print(f"Loading LSTM model: {nn_model_path}")
    nn_model = load_nn_model(nn_model_path)
    if nn_model is None:
        print("ERROR: Failed to load LSTM model.")
        sys.exit(1)
    m = nn_model["meta"]
    print(f"  Trained on {m.get('maps_trained','?')} maps  |  Styles: {len(m.get('style_classes', STYLE_CLASSES))}")

    settings = get_user_settings(args.audio)
    settings["audio_path"] = args.audio

    audio_data = analyze_audio(args.audio)

    # ── Stage 1: model decides segment boundaries and pattern types ───────
    print("[2/4] Classifying song structure...")
    segment_map = segment_and_classify(audio_data, nn_model)

    print(f"   {len(segment_map)} section(s) detected:")
    for start_ms, end_ms, style_id, style_name in segment_map:
        duration_s = (end_ms - start_ms) / 1000
        print(f"   {int(start_ms/1000):4d}s – {int(end_ms/1000):4d}s  ({duration_s:.0f}s)  →  {style_name}")

    # ── Stage 2: style-conditioned note generation ────────────────────────
    print("[3/4] Generating notes (style-conditioned per section)...")
    d      = DIFFICULTY_PRESETS[settings["difficulty"]]
    chord  = d["chord"] * d["chord_scale"]
    notes  = assign_columns_nn(
        audio_data    = audio_data,
        nn_model_data = nn_model,
        keep          = d["keep"],
        ln_chance     = d["ln"],
        chord_chance  = chord,
        breathe       = d["breathe"],
        segment_map   = segment_map,
    )
    print(f"   base keep: {d['keep']}  chord: {chord:.2f}  breathe: {d['breathe']}×")
    print(f"   (all parameters modulated per section with {_TRANSITION_FACTOR}-beat blended transitions)")

    sv_points = []
    if settings["sv"]:
        print("   Generating SV points...")
        sv_points = generate_sv_points(audio_data)

    safe_title  = "".join(c for c in settings["title"]  if c.isalnum() or c in " -_")
    safe_artist = "".join(c for c in settings["artist"] if c.isalnum() or c in " -_")

    if args.output:
        output_path = args.output
    else:
        audio_dir   = os.path.dirname(os.path.abspath(args.audio))
        output_path = os.path.join(
            audio_dir,
            f"{safe_artist} - {safe_title} [{settings['difficulty']}].osz"
        )

    print("[4/4] Building .osz...")
    count = build_osz(settings, audio_data, notes, sv_points, output_path)

    print(f"\n  Done!")
    print(f"  Notes    : {count}")
    print(f"  BPM      : {audio_data['bpm']:.1f}")
    print(f"  Sections : {len(segment_map)}")
    print(f"  Output   : {output_path}")
    print(f"\n  Double-click the .osz to import directly into osu!")


if __name__ == "__main__":
    main()
