"""
ManiaMapper.py
Generates a 4K osu!mania beatmap from an audio file.

Usage:
    python ManiaMapper.py song.mp3
    python ManiaMapper.py song.mp3 --output my_map.osu
    python ManiaMapper.py song.mp3 --model mania_model.json
"""

import os, sys, random, argparse, zipfile, json
import numpy as np

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

KEYS  = 4
COL_X = [64, 192, 320, 448]   # x-positions for columns 0-3 in 4K

# breathe=1.5: blocks same-column AA (1 step) but allows cross-hand ABAB trills (2 steps).
# Same-hand ABA (0→1→0) is caught by hand-aware filtering in note selection.
DIFFICULTY_PRESETS = {
    #  keep         = fraction of onset-aligned slots that get a note (0=nothing, 1=all beats)
    #  chord_scale  = multiplier on Markov chord rate — keep low for flowing single notes
    #  breathe      = per-column cooldown multiplier — higher = more gap between same-col hits
    "Easy":   {"hp": 6, "od": 6,  "subdiv": 2, "keep": 0.40, "chord": 0.05, "ln": 0.20, "chord_scale": 0.25, "breathe": 2.0, "triplets": False},
    "Normal": {"hp": 7, "od": 7,  "subdiv": 4, "keep": 0.55, "chord": 0.08, "ln": 0.15, "chord_scale": 0.40, "breathe": 1.5, "triplets": False},
    "Hard":   {"hp": 8, "od": 8,  "subdiv": 8, "keep": 0.70, "chord": 0.12, "ln": 0.10, "chord_scale": 0.55, "breathe": 1.2, "triplets": True},
    "Insane": {"hp": 9, "od": 9,  "subdiv": 8, "keep": 1.0, "chord": 0.30, "ln": 0.05, "chord_scale": 1.00, "breathe": 2.0, "triplets": True},
}

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

    print("[1/3] Analysing audio...")
    y, sr = librosa.load(audio_path, sr=None)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr) * 1000

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units="frames",
                                              hop_length=512, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr) * 1000

    # Full RMS (overall energy)
    hop = 512
    rms       = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop) * 1000

    # Spectral bands + MFCC — used for both Markov band-aware mapping and LSTM inference
    S     = np.abs(librosa.stft(y, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr)
    def _norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-9)

    bass_norm = _norm(S[freqs < 300, :].mean(axis=0))
    mid_norm  = _norm(S[(freqs >= 300) & (freqs < 3000), :].mean(axis=0))
    high_norm = _norm(S[freqs >= 3000, :].mean(axis=0))

    # MFCC — needed by the LSTM model
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop)
    mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-9)

    bpm_orig    = float(np.atleast_1d(tempo)[0])
    bpm         = bpm_orig * 2          # doubled for finer Markov/rule-based grid
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
    """
    Generate timing grid snapped to beat subdivisions.
    subdiv=2 -> 8th notes, 4 -> 16th, 8 -> 32nd.
    triplets=True also inserts 1/3-beat and 2/3-beat slots for off-beat triplet patterns.
    Returns list of (time_ms, strength), deduped and sorted.
    """
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

        # Standard subdivision slots
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

        # Triplet slots: 1/3 and 2/3 of the beat interval
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
    """
    Sample a key from counts dict weighted by count values.
    Restricts to keys whose int value is in allowed if provided.
    """
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
    """
    Sample a Markov-weighted 2-col chord restricted to free_cols.
    Returns a sorted 2-list or None if nothing fits.
    """
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


# ─── LSTM INFERENCE (mania_nn_model.pt) ──────────────────────────────────────

_NN_N_MFCC   = 20
_NN_FEAT_DIM = _NN_N_MFCC + 5
_NN_SUBDIV   = 4   # training subdivision


def load_nn_model(model_path: str):
    """Load the trained LSTM from ManiaNNTrainer.py."""
    try:
        import torch
    except ImportError:
        return None

    try:
        import torch.nn as nn
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

        class ManiaLSTM(nn.Module):
            def __init__(self, feat_dim, hidden_dim, num_layers):
                super().__init__()
                self.lstm = nn.LSTM(feat_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
                self.norm = nn.LayerNorm(hidden_dim)
                self.fc   = nn.Linear(hidden_dim, 4)
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(self.norm(out))

        model = ManiaLSTM(
            feat_dim   = ckpt.get("feat_dim",   _NN_FEAT_DIM),
            hidden_dim = ckpt.get("hidden_dim", 256),
            num_layers = ckpt.get("num_layers", 3),
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return {"model": model, "meta": ckpt}
    except Exception as e:
        print(f"   [warn] Could not load NN model: {e}")
        return None


def assign_columns_nn(audio_data, nn_model_data, keep, ln_chance, chord_chance=0.15, breathe=1.8):
    """
    LSTM-based note generation.

    The trained model sees beat-aligned MFCC + spectral features and outputs
    per-column probabilities at each 16th-note position.
    Threshold + anti-pattern filters (breathe, ABA) are applied post-hoc.
    """
    try:
        import torch
    except ImportError:
        return None

    model      = nn_model_data["model"]
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    beat_times   = audio_data["beat_times"]
    beat_len_orig= audio_data["beat_length_orig"]   # original BPM beat length
    beat_len     = audio_data["beat_length"]         # doubled BPM beat length (finer grid)
    mfcc         = audio_data["mfcc"]               # (20, T_frames)
    bass_norm    = audio_data["bass_norm"]
    mid_norm     = audio_data["mid_norm"]
    high_norm    = audio_data["high_norm"]
    rms          = audio_data["rms"]
    rms_times    = audio_data["rms_times"]
    rms_norm     = (rms - rms.min()) / (rms.max() - rms.min() + 1e-6)
    duration     = audio_data["duration_ms"]
    n_frames     = mfcc.shape[1]
    # Use doubled-BPM step for 2× denser grid (32nd notes at original BPM)
    step_ms      = beat_len / _NN_SUBDIV
    min_gap      = 80.0  # hard physical minimum between same-column notes

    def _feat_at(t_ms: float) -> np.ndarray:
        idx = min(int(np.searchsorted(rms_times, t_ms)), n_frames - 1)
        f   = np.zeros(_NN_FEAT_DIM, dtype=np.float32)
        f[:_NN_N_MFCC]        = mfcc[:, idx]
        f[_NN_N_MFCC]         = bass_norm[idx]
        f[_NN_N_MFCC + 1]     = mid_norm[idx]
        f[_NN_N_MFCC + 2]     = high_norm[idx]
        phase                  = (t_ms % (beat_len_orig * 4)) / (beat_len_orig * 4 + 1e-9)
        f[_NN_N_MFCC + 3]     = float(np.sin(2 * np.pi * phase))
        f[_NN_N_MFCC + 4]     = float(np.cos(2 * np.pi * phase))
        return f

    # Build beat-aligned positions using doubled BPM → 2× more grid slots
    positions = []
    for t_beat in beat_times:
        for s in range(_NN_SUBDIV * 2):   # *2 to cover doubled BPM subdivisions
            t = t_beat + s * step_ms
            if t > duration:
                break
            positions.append(t)

    if not positions:
        return []

    features = np.stack([_feat_at(t) for t in positions])          # (T, F)
    X        = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, F)

    with torch.no_grad():
        probs = torch.sigmoid(model(X))[0].cpu().numpy()           # (T, 4)

    # Smooth rms_norm over ~300ms window to avoid burst-gap-burst in slow sections
    win = max(1, int(300.0 / ((rms_times[1] - rms_times[0]) if len(rms_times) > 1 else 1.0)))
    kernel       = np.ones(win) / win
    rms_smooth   = np.convolve(rms, kernel, mode='same')
    rsm_s_min, rsm_s_max = rms_smooth.min(), rms_smooth.max()
    rms_smooth_n = (rms_smooth - rsm_s_min) / (rsm_s_max - rsm_s_min + 1e-9)

    rms_norm_arr     = rms_norm                                   # instant energy (for LN/chord)
    trill_energy_thr = float(np.percentile(rms_norm_arr, 70))    # top 30% triggers trill

    # Precompute bass peak times — local maxima above 70th percentile in bass_norm
    # These are kick/bass drum hits that must always trigger a note
    bass_hit_thr   = float(np.percentile(bass_norm, 70))
    bass_peak_times = []
    for k in range(1, len(bass_norm) - 1):
        if (bass_norm[k] > bass_hit_thr
                and bass_norm[k] >= bass_norm[k - 1]
                and bass_norm[k] >= bass_norm[k + 1]):
            bass_peak_times.append(float(rms_times[k]))
    bass_peak_arr = np.array(bass_peak_times) if bass_peak_times else np.array([])

    # All repeating trill patterns — each entry is a cycle of column sets
    TRILL_PATTERNS = [
        ([0,1], [2,3]),          # 0: chord trill L→R→L→R
        ([2,3], [0,1]),          # 1: chord trill R→L→R→L (same but opposite phase start)
        ([0,3], [1,2]),          # 2: outer/inner chord trill
        ([0], [1], [2], [3]),    # 3: staircase up  0→1→2→3
        ([3], [2], [1], [0]),    # 4: staircase down 3→2→1→0
        ([0], [1], [2], [3], [2], [1]),  # 5: staircase up-down pingpong
        ([0], [2], [0], [2]),    # 6: cross-hand ping-pong (col 0 & 2)
        ([1], [3], [1], [3]),    # 7: cross-hand ping-pong (col 1 & 3)
        ([0], [1], [0], [1]),    # 8: fast left-hand trill
        ([2], [3], [2], [3]),    # 9: fast right-hand trill
        ([0], [2], [1], [3]),    # 10: zigzag across columns
    ]

    notes           = []
    col_busy        = {}
    last_col        = -1
    prev_col        = -1
    trill_active    = False
    trill_type      = 0
    trill_stair_pos = 0
    trill_remaining = 0
    last_kick_t     = -9999.0   # time of last detected kick
    kick_streak     = 0         # consecutive kick count
    jack_cols       = None      # chord columns used in active chord jack

    for i, t in enumerate(positions):
        idx         = min(int(np.searchsorted(rms_times, t)), len(rms_norm_arr) - 1)
        energy      = float(rms_norm_arr[idx])          # instant — for LN / chord / trill
        energy_s    = float(rms_smooth_n[idx])          # smoothed — for density decisions
        bass_energy = float(bass_norm[idx])             # bass hit strength

        # ── Song-position ramp ────────────────────────────────────────────
        song_progress = t / max(duration, 1.0)
        if song_progress < 0.12:
            intro_scale = song_progress / 0.12
        elif song_progress > 0.88:
            intro_scale = (1.0 - song_progress) / 0.12
        else:
            intro_scale = 1.0

        # Check if this position falls on a bass hit (within half a step)
        on_bass_hit = (len(bass_peak_arr) > 0
                       and float(np.min(np.abs(bass_peak_arr - t))) < step_ms * 0.6)

        # Bass hits in slow/medium sections always fire; others use smoothed energy gate
        if not (on_bass_hit and energy_s < 0.70):
            local_keep = keep * (0.35 + 0.65 * energy_s) * intro_scale
            if random.random() > local_keep:
                continue

        # ── Trill zone detection ───────────────────────────────────────────
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
            if len(free) == len(step):   # all columns in this step are free
                for col in step:
                    col_busy[col] = t + min_gap
                    notes.append((round(t), col, False, 0))
                last_col = step[0] if len(step) == 1 else -1
                continue

        # ── Normal note placement ──────────────────────────────────────────
        col_probs = probs[i]

        def col_score(c):
            if c == last_col:
                return -999.0
            score = float(col_probs[c])
            if last_col >= 0:
                dist = abs(c - last_col)
                if dist == 1:   score += 0.40
                elif dist == 2: score += 0.10
            # Soft penalty for returning to prev_col — discourages A→B→A back-and-forths
            # but doesn't hard-block them (still possible when other cols are busy)
            if c == prev_col:
                score -= 0.28
            # Bass hits feel natural on outer columns (0, 3) like a kick drum
            if on_bass_hit and c in (0, 3):
                score += 0.35
            return score

        ranked = sorted(range(KEYS), key=col_score, reverse=True)

        best = None
        for c in ranked:
            if col_busy.get(c, 0) <= t:
                best = c
                break
        if best is None:
            continue

        chosen = [best]

        # ── Chord — boosted on hard bass hits ─────────────────────────────
        # bass_energy > 0.6 = identifiable kick → higher chord probability
        effective_chord = min(0.95, chord_chance + bass_energy * 0.35)
        if random.random() < effective_chord:
            free_cols = [c for c in range(KEYS) if c != best and col_busy.get(c, 0) <= t]
            if free_cols:
                second = max(free_cols, key=lambda c: col_probs[c])
                chosen.append(second)

                # Triple chord — scales with energy
                triple_prob = chord_chance * (0.4 + 0.6 * energy)
                if random.random() < triple_prob:
                    free_cols2 = [c for c in range(KEYS)
                                  if c not in chosen and col_busy.get(c, 0) <= t]
                    if free_cols2:
                        chosen.append(max(free_cols2, key=lambda c: col_probs[c]))

        # ── Chord jack on consecutive hard kicks ──────────────────────────
        in_chord_jack = False
        if on_bass_hit:
            if t - last_kick_t < beat_len_orig * 1.8:
                kick_streak += 1
            else:
                kick_streak = 1
            last_kick_t = t

            if kick_streak >= 2:
                in_chord_jack = True
                # Force at least a 2-col chord; reuse jack_cols if still valid
                free_all = [c for c in range(KEYS) if col_busy.get(c, 0) <= t]
                if jack_cols and all(c in free_all for c in jack_cols):
                    chosen = list(jack_cols)   # repeat same chord = jack feel
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

        # ── LN logic ──────────────────────────────────────────────────────
        # No LNs during chord jacks — player needs clean hits on every kick
        if in_chord_jack:
            for col in chosen:
                col_busy[col] = t + min_gap * breathe
                notes.append((round(t), col, False, 0))
            continue

        if energy < 0.35:
            # Slow: mix of long (45%), short (30%), plain tap (25%)
            r = random.random()
            if r < 0.45:
                ln_prob  = min(0.90, ln_chance * 8.0)
                ln_beats = random.uniform(1.5, 3.5)
            elif r < 0.75:
                ln_prob  = min(0.75, ln_chance * 5.0)
                ln_beats = random.uniform(0.3, 0.9)
            else:
                ln_prob  = 0.0          # plain single tap / chord
                ln_beats = 0.0
        elif energy < 0.65:
            ln_prob  = min(0.80, ln_chance * 2.5)
            ln_beats = random.uniform(0.6, 1.5)
        else:
            # Fast: short LNs common, long LNs rare (10%)
            if random.random() < 0.10:
                ln_prob  = min(0.35, ln_chance * 1.5)
                ln_beats = random.uniform(1.0, 2.0)
            else:
                ln_prob  = min(0.60, ln_chance * 2.5)
                ln_beats = random.uniform(0.15, 0.45)

        # Two simultaneous LNs — rare (8% of chord placements in slow/mid)
        double_ln = (len(chosen) == 2 and energy < 0.65 and random.random() < 0.08)

        for col_idx, col in enumerate(chosen):
            is_ln = random.random() < ln_prob if (double_ln and col_idx < 2) \
                    else ((len(chosen) == 1) and random.random() < ln_prob)
            if is_ln:
                ln_end = round(t + beat_len_orig * ln_beats)
                col_busy[col] = ln_end + min_gap
                notes.append((round(t), col, True, ln_end))
            else:
                col_busy[col] = t + min_gap * breathe
                notes.append((round(t), col, False, 0))

    return notes


# ─── ML COLUMN ASSIGNMENT ─────────────────────────────────────────────────────

def assign_columns_ml(audio_data, model, subdiv, keep, ln_chance, chord_chance, breathe=1.5):
    """
    Onset-driven, band-aware column assignment using the Markov model.

    Timing:
      - Primary candidates = detected audio onsets (actual musical events)
      - Secondary fill     = beat downbeats in high-energy sections with no nearby onset
      - Global 80ms dedup prevents visual stacking

    Band awareness:
      - Bass-heavy onset (kick) → prefer chord or outer columns (0, 3)
      - High-frequency onset (hi-hat) → prefer inner columns (1, 2)
      - Mid/balanced onset → Markov-weighted column selection

    Anti-pattern rules:
      - breathe cooldown per column blocks same-column repeats
      - Hand-aware ABA filter: same-hand ABA blocked, cross-hand trill allowed
    """
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

    # ── Step 1: collect candidate note times ──────────────────────────────
    candidates = set()

    # A) Onset-driven: every detected onset, gated by energy
    for t in onset_arr:
        if t > duration:
            break
        e = _at(rms_norm, t)
        if random.random() < keep * (0.35 + e * 0.85):
            candidates.add(round(t))

    # B) Beat fill: strong downbeats not already covered by a nearby onset
    onset_set = set(round(t) for t in onset_arr)
    for t_beat in beat_times:
        e = _at(rms_norm, t_beat)
        if e > 0.45 and random.random() < keep * e:
            t_r = round(t_beat)
            if not any(abs(t_r - o) < 50 for o in onset_set):
                candidates.add(t_r)

    # Sort and enforce global 80ms minimum gap
    raw_times = sorted(candidates)
    active_times = []
    last_t = -9999
    for t in raw_times:
        if t - last_t >= 80:
            active_times.append(t)
            last_t = t

    # ── Step 2: assign columns ─────────────────────────────────────────────
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

        # Bass-heavy hit → bonus chord probability (accent the kick drum)
        bass_bonus      = bass * chord_chance * 0.6
        effective_chord = min(0.85, chord_chance * (0.2 + energy * 0.8) + bass_bonus)

        # ── Double chord (2 notes, bass-aware column preference) ───────────
        if random.random() < effective_chord and len(free_cols) >= 2:
            if bass > 0.55 and len([c for c in free_cols if c in (0, 3)]) >= 1:
                # Kick: pair one outer + one inner column for a strong feel
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

        # ── Single note (Markov + hand-aware + band preference) ────────────
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

            # Hi-hat / high-freq hit → prefer inner columns (1, 2)
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

        # LN: quiet passages → more long notes
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


# ─── RULE-BASED COLUMN ASSIGNMENT (fallback when no model) ───────────────────

def assign_columns(audio_data, subdiv, keep, chord_chance, ln_chance, breathe=1.5, triplets=False):
    """
    Rule-based fallback. Same anti-pattern rules as assign_columns_ml:
    breathe cooldown + hand-aware same-hand-ABA filtering.
    """
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

    active_times = []
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

        # ── Double chord ────────────────────────────────────────────────────
        if random.random() < effective_chord and len(free_cols) >= 2:
            chosen = sorted(random.sample(free_cols, 2))
            last_single = -1
            prev_single = -1

        # ── Single note (round-robin + hand-aware filtering) ────────────────
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
    bpm         = audio_data["bpm"]
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
    parser.add_argument("--nn",           help="Path to mania_nn_model.pt (LSTM)")
    args = parser.parse_args()

    if not os.path.isfile(args.audio):
        print(f"ERROR: File not found: {args.audio}")
        sys.exit(1)

    scripts_dir   = os.path.dirname(os.path.abspath(__file__))
    nn_model_path = args.nn or os.path.join(scripts_dir, "mania_nn_model.pt")

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
    print(f"  Trained on {m.get('maps_trained','?')} maps  |  LSTM ready")

    settings = get_user_settings(args.audio)
    settings["audio_path"] = args.audio

    audio_data = analyze_audio(args.audio)

    print("[2/3] Generating notes...")
    d      = DIFFICULTY_PRESETS[settings["difficulty"]]
    chord  = d["chord"] * d["chord_scale"]
    notes  = assign_columns_nn(
        audio_data   = audio_data,
        nn_model_data= nn_model,
        keep         = d["keep"],
        ln_chance    = d["ln"],
        chord_chance = chord,
        breathe      = d["breathe"],
    )
    print(f"   [LSTM]  keep: {d['keep']}  |  chord: {chord:.2f}  |  breathe: {d['breathe']}×")

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

    print("[3/3] Building .osz...")
    count = build_osz(settings, audio_data, notes, sv_points, output_path)

    print(f"\n  Done!")
    print(f"  Notes  : {count}")
    print(f"  BPM    : {audio_data['bpm']:.1f}")
    print(f"  Mode   : LSTM")
    print(f"  Output : {output_path}")
    print(f"\n  Double-click the .osz to import directly into osu!")


if __name__ == "__main__":
    main()
