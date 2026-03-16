"""
ManiaMapper.py
Generates a 4K osu!mania beatmap from an audio file.

The model trained by ManiaNNTrainer.py has learned — from many real maps —
what note patterns tend to appear for a given audio context. This script feeds
the new song's audio features through that model and samples from the resulting
probabilities to place notes.

No hardcoded patterns. No style labels. No rule-based overrides.
The model decides everything about pattern type, density, chords, and feel.
Difficulty controls only how many of the model's suggestions survive (threshold).
Variation between runs comes naturally from probabilistic sampling.

The only non-model constraint is the 80ms physical minimum gap per column
(a human finger cannot re-hit the same key faster than that).

Usage:
    python ManiaMapper.py song.mp3
    python ManiaMapper.py song.mp3 --output my_map.osz
    python ManiaMapper.py song.mp3 --nn mania_model.pt
"""

import os, sys, argparse, zipfile
import numpy as np

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

KEYS   = 4
COL_X  = [64, 192, 320, 448]

# fill : target fraction of (time × column) slots that get notes.
#        The model ranks every candidate by confidence; difficulty decides
#        how far down that ranked list we go — not which ones randomly survive.
DIFFICULTY_PRESETS = {
    "Easy":   {"hp": 6, "od": 6,  "fill": 0.08},   # top  8% of model predictions
    "Normal": {"hp": 7, "od": 7,  "fill": 0.15},   # top 15%
    "Hard":   {"hp": 8, "od": 8,  "fill": 0.25},   # top 25%
    "Insane": {"hp": 9, "od": 9,  "fill": 0.38},   # top 38%
}

MIN_GAP_MS = 80.0   # same column cannot fire faster than this (physical limit)

_NN_N_MFCC   = 20
_NN_FEAT_DIM = _NN_N_MFCC + 5
_NN_SUBDIV   = 4


# ─── INTERACTIVE PROMPTS ──────────────────────────────────────────────────────

def ask(prompt, choices=None, default=None):
    while True:
        if choices:
            print(f"\n{prompt}")
            for i, c in enumerate(choices, 1):
                print(f"  [{i}] {c}")
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
    diff   = ask("Difficulty?", choices=["Easy", "Normal", "Hard", "Insane"])
    sv_raw = ask("Speed Variation / SV? (y/n, default n)", default="n")
    sv     = sv_raw.lower() in ("y", "yes")
    print(f"\n  Diff: {diff}  |  SV: {'Yes' if sv else 'No'}\n")
    return {"title": title, "artist": artist, "difficulty": diff, "sv": sv}


# ─── AUDIO ANALYSIS ──────────────────────────────────────────────────────────

def analyze_audio(audio_path):
    try:
        import librosa
    except ImportError:
        print("ERROR: pip install librosa")
        sys.exit(1)

    print("[1/3] Analysing audio...")
    y, sr = librosa.load(audio_path, sr=None)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr) * 1000

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
    beat_length = 60000.0 / bpm_orig
    duration_ms = len(y) / sr * 1000

    print(f"   BPM: {bpm_orig:.1f}  |  Beats: {len(beat_times)}")
    return {
        "bpm_orig":     bpm_orig,
        "beat_length":  beat_length,
        "beat_times":   beat_times,
        "rms":          rms,
        "rms_times":    rms_times,
        "bass_norm":    bass_norm,
        "mid_norm":     mid_norm,
        "high_norm":    high_norm,
        "mfcc":         mfcc,
        "duration_ms":  duration_ms,
    }


# ─── MODEL LOADING ────────────────────────────────────────────────────────────

def load_nn_model(model_path):
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None

    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

        feat_dim   = ckpt.get("feat_dim",   _NN_FEAT_DIM)
        hidden_dim = ckpt.get("hidden_dim", 256)
        num_layers = ckpt.get("num_layers", 3)

        class ManiaMapperLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(feat_dim, hidden_dim, num_layers,
                                    batch_first=True, dropout=0.3)
                self.norm = nn.LayerNorm(hidden_dim)
                self.fc   = nn.Linear(hidden_dim, 4)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(self.norm(out))

        model = ManiaMapperLSTM()
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return {"model": model, "meta": ckpt}
    except Exception as e:
        print(f"   [warn] Could not load model: {e}")
        return None


# ─── NOTE GENERATION ─────────────────────────────────────────────────────────

def generate_notes(audio_data, nn_model_data, fill):
    """
    Feed audio features through the trained LSTM and place notes based on the
    model's confidence, cutting the least certain ones for lower difficulties.

    The model outputs a probability for each of the 4 columns at every 16th-note
    position. These probabilities encode everything — not just timing but which
    columns fire together (chords), which repeat (jacks), which alternate
    (streams). Column identity IS the pattern, so it is used exactly as the
    model outputs it.

    Difficulty works by thresholding: a higher threshold keeps only the notes
    the model is most confident about. The main structural notes (high
    probability) always survive. The marginal ones are cut first.
    No randomness — same song + same difficulty always produces the same map.

    The only non-model constraint is MIN_GAP_MS: same column cannot fire twice
    within 80ms (physical human limit).
    """
    try:
        import torch
    except ImportError:
        return []

    model  = nn_model_data["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    beat_times    = audio_data["beat_times"]
    beat_len_orig = audio_data["beat_length"]
    mfcc          = audio_data["mfcc"]
    bass_norm     = audio_data["bass_norm"]
    mid_norm      = audio_data["mid_norm"]
    high_norm     = audio_data["high_norm"]
    rms_times     = audio_data["rms_times"]
    duration      = audio_data["duration_ms"]
    n_frames      = mfcc.shape[1]
    step_ms       = beat_len_orig / _NN_SUBDIV

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

    # Build 16th-note position grid
    positions = []
    for t_beat in beat_times:
        for s in range(_NN_SUBDIV):
            t = t_beat + s * step_ms
            if t > duration:
                break
            positions.append(t)

    if not positions:
        return []

    # Single forward pass — the LSTM's recurrent state carries context across
    # the whole song, so it naturally understands section structure.
    features = np.stack([_feat_at(t) for t in positions])
    X        = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(X))[0].cpu().numpy()   # (T, 4)

    # Percentile threshold: keep the top `fill` fraction of all model predictions.
    # e.g. fill=0.38 (Insane) → threshold = 62nd percentile of all probs.
    # The model's confidence ranking decides which notes are most important;
    # difficulty just controls how far down that ranked list we go.
    threshold = float(np.percentile(probs, (1.0 - fill) * 100))
    threshold = max(0.20, threshold)   # never accept very uncertain predictions

    # Collect every (prob, time, col) above the threshold
    candidates = [
        (float(probs[i, col]), t, col)
        for i, t in enumerate(positions)
        for col in range(KEYS)
        if probs[i, col] >= threshold
    ]

    # Sort highest confidence first so that when two notes on the same column
    # conflict with the physical gap, the more important one is kept.
    candidates.sort(key=lambda c: c[0], reverse=True)

    col_busy = {}   # col → time after which the column is free again
    placed   = []

    for _, t, col in candidates:
        if col_busy.get(col, -9999.0) > t:
            continue   # physical gap — skip, less confident note loses
        placed.append((round(t), col, False, 0))
        col_busy[col] = t + MIN_GAP_MS

    # Return in time order
    return sorted(placed, key=lambda n: n[0])


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
    audio_file  = os.path.basename(settings["audio_path"])
    diff        = settings["difficulty"]

    lines = [
        "osu file format v14", "",
        "[General]",
        f"AudioFilename: {audio_file}",
        "AudioLeadIn: 0", "PreviewTime: -1", "Countdown: 0",
        "SampleSet: Soft", "Mode: 3", "LetterboxInBreaks: 0",
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
        "ApproachRate:5", "SliderMultiplier:1.4", "SliderTickRate:1",
        "", "[Events]",
        "//Background and Video events", "//Break Periods",
        "", "[TimingPoints]",
        f"0,{beat_length:.6f},4,2,1,100,1,0",
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
    safe_title  = "".join(c for c in settings["title"]  if c.isalnum() or c in " -_")
    safe_artist = "".join(c for c in settings["artist"] if c.isalnum() or c in " -_")
    diff        = settings["difficulty"]
    audio_src   = settings["audio_path"]
    osu_name    = f"{safe_artist} - {safe_title} [{diff}].osu"

    count, osu_content = write_osu(settings, audio_data, notes, sv_points, None)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(osu_name, osu_content.encode("utf-8"))
        zf.write(audio_src, os.path.basename(audio_src))

    return count


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="4K osu!mania map generator")
    parser.add_argument("audio",          help="Audio file (mp3, ogg, wav, flac)")
    parser.add_argument("--output", "-o", help="Output .osz path")
    parser.add_argument("--nn",           help="Path to mania_model.pt")
    args = parser.parse_args()

    if not os.path.isfile(args.audio):
        print(f"ERROR: File not found: {args.audio}")
        sys.exit(1)

    scripts_dir   = os.path.dirname(os.path.abspath(__file__))
    nn_model_path = args.nn or os.path.join(scripts_dir, "mania_model.pt")

    if not os.path.isfile(nn_model_path):
        print(f"ERROR: Model not found at {nn_model_path}")
        print("Run ManiaNNTrainer.py first.")
        sys.exit(1)

    print(f"Loading model: {nn_model_path}")
    nn_model = load_nn_model(nn_model_path)
    if nn_model is None:
        print("ERROR: Failed to load model.")
        sys.exit(1)
    print(f"  Trained on {nn_model['meta'].get('maps_trained', '?')} maps")

    settings = get_user_settings(args.audio)
    settings["audio_path"] = args.audio

    audio_data = analyze_audio(args.audio)

    print("[2/3] Generating notes...")
    d     = DIFFICULTY_PRESETS[settings["difficulty"]]
    notes = generate_notes(audio_data, nn_model, fill=d["fill"])
    print(f"   {len(notes)} notes  |  fill: top {int(d['fill']*100)}% of model predictions")

    sv_points = []
    if settings["sv"]:
        sv_points = generate_sv_points(audio_data)

    safe_title  = "".join(c for c in settings["title"]  if c.isalnum() or c in " -_")
    safe_artist = "".join(c for c in settings["artist"] if c.isalnum() or c in " -_")

    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.audio)),
        f"{safe_artist} - {safe_title} [{settings['difficulty']}].osz"
    )

    print("[3/3] Building .osz...")
    count = build_osz(settings, audio_data, notes, sv_points, output_path)

    print(f"\n  Done!")
    print(f"  Notes  : {count}")
    print(f"  BPM    : {audio_data['bpm_orig']:.1f}")
    print(f"  Output : {output_path}")
    print(f"\n  Double-click the .osz to import into osu!")


if __name__ == "__main__":
    main()
