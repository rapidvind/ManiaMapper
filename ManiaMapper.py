"""
ManiaMapper.py
Generates a 4K osu!mania beatmap from an audio file.

Usage:
    python ManiaMapper.py song.mp3              # CLI mode
    python ManiaMapper.py --ui                  # GUI mode (file picker inside)
    python ManiaMapper.py song.mp3 --ui         # GUI mode, audio pre-loaded
"""

import os, sys, argparse, zipfile
import numpy as np

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

KEYS   = 4
COL_X  = [64, 192, 320, 448]

DIFFICULTY_PRESETS = {
    "Easy":   {"hp": 6, "od": 6,  "fill": 0.04, "max_chord": 1},
    "Normal": {"hp": 7, "od": 7,  "fill": 0.07, "max_chord": 2},
    "Hard":   {"hp": 8, "od": 8,  "fill": 0.12, "max_chord": 3},
    "Insane": {"hp": 9, "od": 9,  "fill": 0.16, "max_chord": 3},
}

MIN_GAP_MS        = 40.0
_NN_SR            = 22050
_NN_HOP           = 512
_NN_SUBDIV        = 4          # 16th-note inference grid
_NN_N_MEL         = 80
_NN_N_SC          = 7
_NN_N_CHROMA      = 12
_NN_FEAT_DIM_AUDIO = _NN_N_MEL + 1 + _NN_N_SC + _NN_N_CHROMA + 1 + 1 + 6  # 108
_NN_FEAT_DIM_CTX  = 16         # zeros at inference (no sibling maps)
_NN_FEAT_DIM      = _NN_FEAT_DIM_AUDIO + _NN_FEAT_DIM_CTX               # 124

DIFF_IDX = {"Easy": 0, "Normal": 1, "Hard": 2, "Insane": 3}


# ─── CLI PROMPTS ──────────────────────────────────────────────────────────────

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
        print("ERROR: pip install librosa"); sys.exit(1)

    print("[1/3] Analysing audio...")
    y, sr = librosa.load(audio_path, sr=_NN_SR, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times  = librosa.frames_to_time(beat_frames, sr=sr) * 1000
    bpm_orig    = float(np.atleast_1d(tempo)[0])
    beat_length = 60000.0 / bpm_orig
    duration_ms = len(y) / sr * 1000
    hop         = _NN_HOP

    # ── mel spectrogram ────────────────────────────────────────────────────────
    mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=_NN_N_MEL, hop_length=hop)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)

    # ── onset strength ─────────────────────────────────────────────────────────
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    onset = (onset - onset.mean()) / (onset.std() + 1e-9)

    # ── spectral contrast ──────────────────────────────────────────────────────
    try:
        sc = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop)
        sc = (sc - sc.mean(axis=1, keepdims=True)) / (sc.std(axis=1, keepdims=True) + 1e-9)
    except Exception:
        sc = np.zeros((_NN_N_SC, mel_db.shape[1]), dtype=np.float32)

    # ── chroma CQT ─────────────────────────────────────────────────────────────
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
        chroma = (chroma - chroma.mean(axis=1, keepdims=True)) / (chroma.std(axis=1, keepdims=True) + 1e-9)
    except Exception:
        chroma = np.zeros((_NN_N_CHROMA, mel_db.shape[1]), dtype=np.float32)

    # ── RMS (also kept for SV) ─────────────────────────────────────────────────
    rms_raw   = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms_raw)), sr=sr, hop_length=hop) * 1000
    rms_feat  = (rms_raw - rms_raw.mean()) / (rms_raw.std() + 1e-9)

    # ── spectral flatness ──────────────────────────────────────────────────────
    try:
        flat = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0]
        flat = (flat - flat.mean()) / (flat.std() + 1e-9)
    except Exception:
        flat = np.zeros(mel_db.shape[1], dtype=np.float32)

    n_frames    = mel_db.shape[1]
    frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop) * 1000

    # ── uniform beat grid (inference at 16th-note resolution) ─────────────────
    beat0_ms  = float(beat_times[0]) if len(beat_times) else 0.0
    step_ms   = beat_length / _NN_SUBDIV
    positions = []
    t = beat0_ms
    while t <= duration_ms:
        positions.append(t)
        t += step_ms

    # ── feature matrix at each grid position ──────────────────────────────────
    def feat_at(t_ms):
        idx = min(int(np.searchsorted(frame_times, t_ms)), n_frames - 1)
        f   = np.zeros(_NN_FEAT_DIM_AUDIO, dtype=np.float32)
        f[:_NN_N_MEL]                                         = mel_db[:, idx]
        f[_NN_N_MEL]                                          = onset[idx]
        f[_NN_N_MEL+1 : _NN_N_MEL+1+_NN_N_SC]               = sc[:, idx]
        f[_NN_N_MEL+1+_NN_N_SC : _NN_N_MEL+1+_NN_N_SC+_NN_N_CHROMA] = chroma[:, idx]
        f[_NN_N_MEL+1+_NN_N_SC+_NN_N_CHROMA]                 = rms_feat[idx]
        f[_NN_N_MEL+1+_NN_N_SC+_NN_N_CHROMA+1]               = flat[idx]
        for k, scale in enumerate([1, 2, 4]):
            period = beat_length * scale
            phase  = (t_ms % period) / (period + 1e-9)
            base   = _NN_N_MEL + 1 + _NN_N_SC + _NN_N_CHROMA + 2 + k * 2
            f[base]   = float(np.sin(2 * np.pi * phase))
            f[base+1] = float(np.cos(2 * np.pi * phase))
        return f

    feat_audio = np.stack([feat_at(t) for t in positions])   # (T, 108)
    # Pad with zeros for cross-diff context (no sibling maps at inference)
    feat_full  = np.concatenate(
        [feat_audio, np.zeros((len(positions), _NN_FEAT_DIM_CTX), dtype=np.float32)],
        axis=1)                                                # (T, 124)

    print(f"   BPM: {bpm_orig:.1f}  |  Grid steps: {len(positions)}")
    return dict(bpm_orig=bpm_orig, beat_length=beat_length, beat_times=beat_times,
                feat_full=feat_full, onset=onset, frame_times=frame_times,
                rms=rms_raw, rms_times=rms_times, n_frames=n_frames,
                positions=positions, step_ms=step_ms, duration_ms=duration_ms,
                # raw signals kept for visualization
                mel_db=mel_db, sc=sc, chroma=chroma)


# ─── MODEL ───────────────────────────────────────────────────────────────────

def _build_transformer(feat_dim, diff_levels, diff_emb, d_model, nhead,
                        num_layers, dim_ff, dropout):
    """Build ManiaTransformerV3 matching the trainer architecture."""
    import torch
    import torch.nn as nn
    import math

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
        def __init__(self, d, drop=0.1):
            super().__init__()
            self.convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(d, d, k, padding=k // 2, groups=d),
                    nn.Conv1d(d, d, 1),
                    nn.GELU(),
                )
                for k in [3, 5, 9]
            ])
            self.proj = nn.Linear(3 * d, d)
            self.norm = nn.LayerNorm(d)
            self.drop = nn.Dropout(drop)

        def forward(self, x):
            h = x.transpose(1, 2)
            outs = [conv(h).transpose(1, 2) for conv in self.convs]
            h = torch.cat(outs, dim=-1)
            return self.norm(x + self.drop(self.proj(h)))

    class ManiaTransformerV3(nn.Module):
        def __init__(self):
            super().__init__()
            self.diff_emb   = nn.Embedding(diff_levels, diff_emb)
            self.input_proj = nn.Linear(feat_dim + diff_emb, d_model)
            self.conv_block = PatternConvBlock(d_model, drop=dropout)
            self.pos_enc    = PositionalEncoding(d_model, drop=dropout)
            enc_layer       = nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward=dim_ff,
                dropout=dropout, batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
            self.norm        = nn.LayerNorm(d_model)
            self.heads       = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, 4),
                )
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
            return [head(h) for head in self.heads]   # list of (B, T, 4)

    return ManiaTransformerV3()


def load_nn_model(model_path):
    try:
        import torch
    except ImportError:
        return None
    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

        # Read hyperparams saved by trainer (with sensible fallbacks)
        feat_dim    = ckpt.get("feat_dim",    _NN_FEAT_DIM)
        diff_levels = ckpt.get("diff_levels", 4)
        diff_emb    = ckpt.get("diff_emb",    8)
        d_model     = ckpt.get("d_model",     256)
        nhead       = ckpt.get("nhead",       8)
        num_layers  = ckpt.get("num_layers",  4)
        dim_ff      = ckpt.get("dim_ff",      1024)
        dropout     = ckpt.get("dropout",     0.1)

        model_type  = ckpt.get("model_type", "ManiaTransformerV3")
        if "V3" not in model_type:
            print(f"   [warn] Checkpoint type '{model_type}' — retrain with ManiaNNTrainer.py for best results")

        m = _build_transformer(feat_dim, diff_levels, diff_emb,
                               d_model, nhead, num_layers, dim_ff, dropout)
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        return {"model": m, "meta": ckpt}
    except Exception as e:
        print(f"   [warn] Could not load model: {e}"); return None


# ─── NOTE GENERATION ─────────────────────────────────────────────────────────

def generate_notes(audio_data, nn_model_data, fill, difficulty="Hard", max_chord=2):
    try:
        import torch
    except ImportError:
        return []

    model  = nn_model_data["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    positions = audio_data["positions"]
    feat_full = audio_data["feat_full"]       # (T, 124) — pre-computed in analyze_audio
    diff_idx  = DIFF_IDX.get(difficulty, 2)

    if not positions:
        return [], {"all_prob": np.zeros((0, 4)), "positions": [], "threshold": 0.5}

    T        = len(positions)
    all_prob = np.zeros((T, KEYS), dtype=np.float32)
    diff_t   = torch.tensor([diff_idx], dtype=torch.long).to(device)
    CHUNK, OVERLAP = 512, 64

    with torch.no_grad():
        i = 0
        while i < T:
            j = min(i + CHUNK, T)
            x = torch.tensor(feat_full[i:j], dtype=torch.float32).unsqueeze(0).to(device)
            d = diff_t.expand(x.size(0))
            all_heads = model(x, d)            # list of 4 tensors (1, chunk, 4)
            all_prob[i:j] = torch.sigmoid(all_heads[diff_idx])[0].cpu().numpy()
            i = j - OVERLAP if j < T else T

    # threshold from fill% — no max(0.20,...) floor
    threshold = float(np.percentile(all_prob, (1.0 - fill) * 100))

    # time-ordered placement with per-step chord cap + column balance weighting
    col_last   = {c: -9999.0 for c in range(KEYS)}
    col_counts = [0] * KEYS
    placed     = []

    for i, t in enumerate(positions):
        total_placed = sum(col_counts) + 1
        avg_per_col  = total_placed / KEYS
        eligible = []
        for col in range(KEYS):
            p = float(all_prob[i, col])
            if p >= threshold and t - col_last[col] >= MIN_GAP_MS:
                # boost underrepresented columns, penalise overrepresented ones
                bal = avg_per_col / max(col_counts[col], 1)
                bal = min(max(bal, 0.25), 4.0)   # clamp to avoid extremes
                eligible.append((p * bal, col))
        eligible.sort(reverse=True)
        for _, col in eligible[:max_chord]:
            placed.append((round(t), col, False, 0))
            col_last[col]   = t
            col_counts[col] += 1

    return placed, {"all_prob": all_prob, "positions": positions, "threshold": threshold}


# ─── ANALYSIS / VISUALIZATION ────────────────────────────────────────────────

def save_analysis(notes, analysis, audio_data, settings, output_dir):
    """Save full analysis PNG (song vs map side-by-side) + stats JSON."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import json as _json
    except ImportError:
        print("   [info] pip install matplotlib for map analysis charts"); return

    all_prob  = analysis["all_prob"]
    positions = analysis["positions"]
    threshold = analysis["threshold"]
    duration  = audio_data["duration_ms"] / 1000
    bpm       = audio_data["bpm_orig"]
    pos_sec   = [t / 1000 for t in positions]

    COL_COLORS = ["#4caf82", "#4aa8d4", "#f0a030", "#ff66ab"]
    COL_NAMES  = ["← C1", "← C2", "← C3", "← C4"]
    BG, SURF   = "#0e0b1a", "#130e22"
    frame_sec  = audio_data["frame_times"] / 1000

    # ── figure layout ─────────────────────────────────────────────────────────
    # Left column: song signals  |  Right column: map signals
    fig = plt.figure(figsize=(26, 14), facecolor=BG)
    # 7 rows: onset/RMS | chroma | 4 col probs | note roll | NPS density
    outer = gridspec.GridSpec(1, 2, figure=fig, wspace=0.06,
                              left=0.04, right=0.98, top=0.93, bottom=0.06)
    gs_l  = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=outer[0], hspace=0.05,
                                             height_ratios=[1, 1.5, 0.6, 0.6])
    gs_r  = gridspec.GridSpecFromSubplotSpec(6, 1, subplot_spec=outer[1], hspace=0.05,
                                             height_ratios=[1, 1, 1, 1, 1.2, 1])

    def _ax(spec, hide_x=True):
        ax = fig.add_subplot(spec)
        ax.set_facecolor(SURF)
        ax.tick_params(colors="#9d8cbb", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#3a2d62")
        if hide_x:
            ax.set_xticklabels([])
        ax.set_xlim(0, duration)
        return ax

    # ════════════════════════ LEFT: SONG SIGNALS ═══════════════════════════════

    # Row 0 — Onset strength + RMS energy
    ax_ons = _ax(gs_l[0])
    onset_raw = audio_data["onset"]
    ax_ons.fill_between(frame_sec[:len(onset_raw)], onset_raw,
                        alpha=0.55, color="#ff66ab")
    ax_ons.plot(frame_sec[:len(onset_raw)], onset_raw,
                color="#ff66ab", linewidth=0.5)
    rms_raw = audio_data["rms"]
    rms_norm = (rms_raw - rms_raw.min()) / (rms_raw.max() - rms_raw.min() + 1e-9)
    rms_t = audio_data["rms_times"] / 1000
    ax_ons.plot(rms_t[:len(rms_norm)], rms_norm * onset_raw.max(),
                color="#c084fc", linewidth=0.8, alpha=0.7, label="RMS (scaled)")
    ax_ons.set_ylabel("Onset / RMS", color="#9d8cbb", fontsize=8)
    ax_ons.set_title("SONG SIGNALS", color="#9d8cbb", fontsize=9, loc="left", pad=4)

    # Row 1 — Chroma heatmap (harmonic content)
    ax_chr = _ax(gs_l[1])
    chroma = audio_data["chroma"]           # (12, T_frames)
    note_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    im = ax_chr.imshow(chroma, aspect="auto", origin="lower",
                       extent=[0, duration, -0.5, 11.5],
                       cmap="magma", vmin=-2, vmax=2, interpolation="nearest")
    ax_chr.set_yticks(range(12))
    ax_chr.set_yticklabels(note_names, color="#9d8cbb", fontsize=6)
    ax_chr.set_ylabel("Chroma", color="#9d8cbb", fontsize=8)

    # Row 2 — Spectral contrast (texture)
    ax_sc = _ax(gs_l[2])
    sc = audio_data["sc"]                   # (7, T_frames)
    sc_mean = sc.mean(axis=0)
    ax_sc.fill_between(frame_sec[:len(sc_mean)], sc_mean,
                       alpha=0.5, color="#4aa8d4")
    ax_sc.plot(frame_sec[:len(sc_mean)], sc_mean, color="#4aa8d4", linewidth=0.5)
    ax_sc.set_ylabel("Spec. Contrast", color="#9d8cbb", fontsize=8)

    # Row 3 — Beat grid markers
    ax_bt = _ax(gs_l[3], hide_x=False)
    for bt in audio_data["beat_times"] / 1000:
        if 0 <= bt <= duration:
            ax_bt.axvline(bt, color="#6450a4", linewidth=0.4, alpha=0.7)
    ax_bt.set_ylabel("Beats", color="#9d8cbb", fontsize=8)
    ax_bt.set_xlabel("Time (s)", color="#9d8cbb", fontsize=9)

    # ════════════════════════ RIGHT: MAP SIGNALS ═══════════════════════════════

    # Rows 0-3 — Per-column probability traces
    for col in range(4):
        ax = _ax(gs_r[col])
        p  = all_prob[:, col] if len(all_prob) else []
        if len(p):
            ax.fill_between(pos_sec, p, alpha=0.45, color=COL_COLORS[col])
            ax.plot(pos_sec, p, color=COL_COLORS[col], linewidth=0.6, alpha=0.9)
            ax.axhline(threshold, color="white", linewidth=0.7,
                       linestyle="--", alpha=0.45)
        ax.set_ylim(0, 1)
        ax.set_ylabel(COL_NAMES[col], color=COL_COLORS[col],
                      fontsize=8, rotation=0, labelpad=28, va="center")
        if col == 0:
            ax.set_title("MAP SIGNALS", color="#9d8cbb", fontsize=9, loc="left", pad=4)

    # Row 4 — Note piano-roll
    ax_n = _ax(gs_r[4])
    if notes:
        nt = [n[0] / 1000 for n in notes]
        nc = [n[1]         for n in notes]
        ax_n.scatter(nt, nc, c=[COL_COLORS[c] for c in nc],
                     s=4, alpha=0.9, marker="|", linewidths=1.5)
    ax_n.set_ylim(-0.5, 3.5)
    ax_n.set_yticks([0, 1, 2, 3])
    ax_n.set_yticklabels(["C1", "C2", "C3", "C4"], color="#9d8cbb", fontsize=7)
    ax_n.set_ylabel("Notes", color="#9d8cbb", fontsize=8)

    # Row 5 — NPS density over time
    ax_d = _ax(gs_r[5], hide_x=False)
    if notes and duration > 0:
        bin_w   = 1.0
        bins    = int(duration / bin_w) + 1
        density = [0.0] * bins
        col_den = [[0.0] * bins for _ in range(4)]
        for t_ms, c, *_ in notes:
            b = int(t_ms / 1000 / bin_w)
            if b < bins:
                density[b] += 1
                col_den[c][b] += 1
        bc = [b * bin_w + bin_w / 2 for b in range(bins)]
        ax_d.fill_between(bc, density, alpha=0.55, color="#c084fc")
        ax_d.plot(bc, density, color="#c084fc", linewidth=0.7)
        for col in range(4):
            ax_d.fill_between(bc, col_den[col], alpha=0.22, color=COL_COLORS[col])
        avg_nps = len(notes) / duration
        ax_d.axhline(avg_nps, color="white", linewidth=0.7, linestyle="--", alpha=0.5)
        ax_d.text(duration * 0.01, avg_nps * 1.06, f"avg {avg_nps:.1f} NPS",
                  color="white", fontsize=7, alpha=0.7)
    ax_d.set_ylim(bottom=0)
    ax_d.set_ylabel("NPS", color="#9d8cbb", fontsize=8)
    ax_d.set_xlabel("Time (s)", color="#9d8cbb", fontsize=9)

    # ── stats ─────────────────────────────────────────────────────────────────
    chord_dist = {}
    step_notes = {}
    for t, c, _, _ in notes:
        step_notes.setdefault(round(t), []).append(c)
    for cols in step_notes.values():
        n = len(cols)
        chord_dist[n] = chord_dist.get(n, 0) + 1

    col_counts = [0, 0, 0, 0]
    for _, c, _, _ in notes:
        col_counts[c] += 1

    nps     = len(notes) / duration if duration > 0 else 0
    cd_str  = "  ".join(f"{k}-note:{v}" for k, v in sorted(chord_dist.items()))
    bal_str = "  ".join(f"C{i+1}:{col_counts[i]}" for i in range(4))

    title = (f"{settings['title']}  ·  {settings['difficulty']}  ·  "
             f"{len(notes)} notes  ·  {nps:.1f} NPS  ·  {bpm:.1f} BPM\n"
             f"chords: {cd_str}    columns: {bal_str}")
    fig.suptitle(title, color="#ff66ab", fontsize=9, y=0.98)

    safe = "".join(c for c in settings["title"] if c.isalnum() or c in " -_")
    png_path  = os.path.join(output_dir, f"{safe}_{settings['difficulty']}_analysis.png")
    json_path = os.path.join(output_dir, f"{safe}_{settings['difficulty']}_analysis.json")

    plt.savefig(png_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()

    total_notes  = len(notes)
    col_pct      = {f"col{i+1}_pct": round(col_counts[i] / max(total_notes, 1) * 100, 1) for i in range(4)}
    stats = {
        "title": settings["title"], "artist": settings["artist"],
        "difficulty": settings["difficulty"], "bpm": round(bpm, 1),
        "duration_s": round(duration, 1), "total_notes": total_notes,
        "notes_per_second": round(nps, 2), "threshold_used": round(float(threshold), 4),
        "chord_distribution": chord_dist,
        "column_balance": {f"col{i+1}": col_counts[i] for i in range(4)},
        "column_balance_pct": col_pct,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        _json.dump(stats, f, indent=2)

    print(f"   Analysis PNG  → {png_path}")
    print(f"   Analysis JSON → {json_path}")


# ─── SV ──────────────────────────────────────────────────────────────────────

def generate_sv_points(audio_data):
    rms       = audio_data["rms"]
    rms_times = audio_data["rms_times"]
    beat_len  = audio_data["beat_length"]
    start     = audio_data["beat_times"][0] if len(audio_data["beat_times"]) else 0
    duration  = audio_data["duration_ms"]
    rms_norm  = (rms - rms.min()) / (rms.max() - rms.min() + 1e-6)
    sv_pts, last, t = [], None, start
    while t < duration:
        idx     = min(int(np.searchsorted(rms_times, t)), len(rms_norm) - 1)
        sv_mult = round((0.70 + rms_norm[idx] * 0.60) * 4) / 4
        bpm_val = round(-100.0 / sv_mult, 2)
        if bpm_val != last:
            sv_pts.append((round(t), bpm_val)); last = bpm_val
        t += beat_len * 2
    return sv_pts


# ─── .OSU / .OSZ ─────────────────────────────────────────────────────────────

def write_osu(settings, audio_data, notes, sv_points, output_path):
    d = DIFFICULTY_PRESETS[settings["difficulty"]]
    lines = [
        "osu file format v14", "",
        "[General]",
        f"AudioFilename: {os.path.basename(settings['audio_path'])}",
        "AudioLeadIn: 0", "PreviewTime: -1", "Countdown: 0",
        "SampleSet: Soft", "Mode: 3", "LetterboxInBreaks: 0",
        "", "[Editor]", "DistanceSpacing: 1", "BeatDivisor: 4", "GridSize: 32", "TimelineZoom: 1",
        "", "[Metadata]",
        f"Title:{settings['title']}", f"TitleUnicode:{settings['title']}",
        f"Artist:{settings['artist']}", f"ArtistUnicode:{settings['artist']}",
        "Creator:ManiaMapper AI", f"Version:{settings['difficulty']} [Mix]",
        "Source:", "Tags:ai generated mania", "BeatmapID:0", "BeatmapSetID:-1",
        "", "[Difficulty]",
        f"HPDrainRate:{d['hp']}", f"CircleSize:{KEYS}",
        f"OverallDifficulty:{d['od']}", "ApproachRate:5",
        "SliderMultiplier:1.4", "SliderTickRate:1",
        "", "[Events]", "//Background and Video events", "//Break Periods",
        "", "[TimingPoints]",
        f"0,{audio_data['beat_length']:.6f},4,2,1,100,1,0",
    ]
    for sv_time, sv_val in sv_points:
        lines.append(f"{sv_time},{sv_val:.2f},4,2,1,100,0,0")
    lines += ["", "[HitObjects]"]
    for (t, col, is_ln, ln_end) in sorted(notes, key=lambda n: n[0]):
        x = COL_X[col]
        lines.append(f"{x},192,{t},128,0,{ln_end}:0:0:0:0:" if is_ln
                     else f"{x},192,{t},1,0,0:0:0:0:")
    osu_content = "\n".join(lines)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(osu_content)
    return len(notes), osu_content


def build_osz(settings, audio_data, notes, sv_points, output_path):
    safe_t = "".join(c for c in settings["title"]  if c.isalnum() or c in " -_")
    safe_a = "".join(c for c in settings["artist"] if c.isalnum() or c in " -_")
    count, content = write_osu(settings, audio_data, notes, sv_points, None)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{safe_a} - {safe_t} [{settings['difficulty']}].osu",
                    content.encode("utf-8"))
        zf.write(settings["audio_path"], os.path.basename(settings["audio_path"]))
    return count


# ─── GUI HELPERS ─────────────────────────────────────────────────────────────

def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _get_font(size, bold=True):
    """Return a PIL ImageFont, falling back gracefully."""
    from PIL import ImageFont
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    for path in [f"C:/Windows/Fonts/{name}", f"/usr/share/fonts/truetype/msttcorefonts/{name}"]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()

def _pill_photo(w, h, text, fill_hex, parent_bg_hex,
                fg_hex="#ffffff", font_size=11, bold=True, scale=3):
    """
    Render a smooth anti-aliased pill button as ImageTk.PhotoImage.
    Rendered at `scale`× resolution then downsampled with LANCZOS.
    """
    from PIL import Image, ImageDraw, ImageTk
    resample = getattr(getattr(Image, "Resampling", None), "LANCZOS", None) or Image.LANCZOS

    W, H   = w * scale, h * scale
    r      = H // 2
    bg_rgb = _hex_to_rgb(parent_bg_hex)
    fg_rgb = _hex_to_rgb(fg_hex)
    fl_rgb = _hex_to_rgb(fill_hex)

    img  = Image.new("RGB", (W, H), bg_rgb)
    draw = ImageDraw.Draw(img)

    try:
        draw.rounded_rectangle([0, 0, W - 1, H - 1], radius=r, fill=fl_rgb)
    except AttributeError:
        # Pillow < 8.2 — manual arcs
        d = 2 * r
        draw.ellipse([0, 0, d - 1, d - 1],             fill=fl_rgb)
        draw.ellipse([W - d, 0, W - 1, d - 1],         fill=fl_rgb)
        draw.ellipse([0, H - d, d - 1, H - 1],         fill=fl_rgb)
        draw.ellipse([W - d, H - d, W - 1, H - 1],     fill=fl_rgb)
        draw.rectangle([r, 0, W - r - 1, H - 1],       fill=fl_rgb)
        draw.rectangle([0, r, W - 1, H - r - 1],       fill=fl_rgb)

    font = _get_font(font_size * scale, bold=bold)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    cx, cy = W // 2, H // 2
    draw.text(
        (cx - (right + left) // 2, cy - (bottom + top) // 2),
        text, fill=fg_rgb, font=font
    )

    img = img.resize((w, h), resample)
    return ImageTk.PhotoImage(img)


# ─── GUI CLASS ────────────────────────────────────────────────────────────────

class ManiaMapperGUI:
    # palette
    BG      = "#0e0b1a"
    SURF    = "#1b1530"
    SURF2   = "#251e40"
    BD      = "#3a2d62"
    BD_LT   = "#6450a4"
    PINK    = "#ff66ab"
    PINK_H  = "#ff3d94"
    PINK_DK = "#b02860"
    PINK_LT = "#ff99cc"
    TEXT    = "#ede8f5"
    DIM     = "#9d8cbb"
    MUTE    = "#6a578a"
    INP     = "#130e22"
    OK      = "#56d4a0"

    DIFFS = {
        "Easy":   ("#4caf82", "#357a5a"),
        "Normal": ("#4aa8d4", "#327898"),
        "Hard":   ("#f0a030", "#b07820"),
        "Insane": ("#ff66ab", "#cc3377"),
    }

    # layout constants (derived from W=660)
    W, H       = 660, 780
    CARD_MX    = 18          # card outer horizontal margin
    CARD_BD    = 1           # border thickness
    CARD_PX    = 32          # card inner horizontal padding
    CONTENT_W  = 660 - 2 * (18 + 1 + 32)   # = 558
    DIFF_H     = 38
    GEN_H      = 56

    def __init__(self, root, audio_path=None):
        import queue as _q
        self.root       = root
        self.audio_path = audio_path
        self._q         = _q.Queue()

        self._selected_diff  = "Hard"
        self._open_btn       = None
        self._gen_enabled    = True
        self._gen_text       = "✨   Generate Map!"

        # PIL image stores (keep references so GC doesn't collect them)
        self._diff_imgs  = {}   # diff → {fill_hex: PhotoImage}
        self._gen_img    = {}   # "normal" | "disabled"

        self._setup_window()
        self._build()

        if audio_path:
            self._audio_var.set(os.path.basename(audio_path))
            stem = os.path.splitext(os.path.basename(audio_path))[0]
            self._title_var.set(stem)

        self._poll()

    # ── window ────────────────────────────────────────────────────────────────

    def _setup_window(self):
        self.root.title("ManiaMapper")
        self.root.geometry(f"{self.W}x{self.H}")
        self.root.resizable(False, False)
        self.root.configure(bg=self.BG)
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{self.W}x{self.H}+{(sw-self.W)//2}+{(sh-self.H)//2}")

    # ── layout ────────────────────────────────────────────────────────────────

    def _build(self):
        self._build_header()
        self._build_card()

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=self.BG)
        hdr.pack(fill="x", pady=(26, 0))

        dot_row = tk.Frame(hdr, bg=self.BG)
        dot_row.pack()
        for _ in range(5):
            tk.Label(dot_row, text="⬥", font=("Segoe UI", 8),
                     fg=self.PINK, bg=self.BG).pack(side="left", padx=3)

        tk.Label(hdr, text="ManiaMapper",
                 font=("Segoe UI", 32, "bold"),
                 fg=self.PINK, bg=self.BG).pack(pady=(8, 3))
        tk.Label(hdr, text="osu!mania AI Map Generator  ·  4K",
                 font=("Segoe UI", 11),
                 fg=self.MUTE, bg=self.BG).pack()

    def _build_card(self):
        # Rounded card via Canvas + PIL
        card_cv = tk.Canvas(self.root, bg=self.BG, bd=0, highlightthickness=0)
        card_cv.pack(fill="both", expand=True, padx=self.CARD_MX, pady=(16, 18))

        inner = tk.Frame(card_cv, bg=self.SURF, padx=self.CARD_PX, pady=24)
        inner_win = card_cv.create_window(1, 1, anchor="nw", window=inner)

        def _redraw_card(evt):
            from PIL import Image, ImageDraw, ImageTk
            w, h = evt.width, evt.height
            if w < 4 or h < 4:
                return
            scale, r = 2, 18
            img = Image.new("RGB", (w * scale, h * scale), _hex_to_rgb(self.BG))
            drw = ImageDraw.Draw(img)
            drw.rounded_rectangle([0, 0, w*scale-1, h*scale-1],
                                  radius=r*scale, fill=_hex_to_rgb(self.BD_LT))
            drw.rounded_rectangle([scale, scale, w*scale-scale-1, h*scale-scale-1],
                                  radius=(r-1)*scale, fill=_hex_to_rgb(self.SURF))
            resample = getattr(getattr(Image, "Resampling", None), "LANCZOS", None) or Image.LANCZOS
            img = img.resize((w, h), resample)
            card_cv._bg = ImageTk.PhotoImage(img)
            card_cv.delete("card_bg")
            card_cv.create_image(0, 0, anchor="nw", image=card_cv._bg, tags="card_bg")
            card_cv.tag_lower("card_bg")
            card_cv.itemconfig(inner_win, width=w - 2, height=h - 2)

        card_cv.bind("<Configure>", _redraw_card)

        # glass shimmer at top
        tk.Frame(inner, bg="#2e2550", height=1).pack(fill="x", pady=(0, 20))

        self._audio_var  = tk.StringVar(value="No file selected")
        self._title_var  = tk.StringVar()
        self._artist_var = tk.StringVar(value="Unknown")
        self._sv_var     = tk.BooleanVar(value=False)

        self._make_audio_row(inner)
        tk.Frame(inner, bg=self.BD, height=1).pack(fill="x", pady=(2, 16))
        self._make_entry_row(inner, "Song Title", self._title_var)
        self._make_entry_row(inner, "Artist",     self._artist_var)
        self._make_diff_row(inner)
        self._make_sv_row(inner)
        tk.Frame(inner, bg=self.SURF, height=8).pack()
        self._make_gen_btn(inner)
        self._make_progress(inner)

    # ── rounded widget helpers ─────────────────────────────────────────────────

    def _rounded_entry(self, parent, var, radius=10, height=38):
        """Canvas with PIL rounded background containing an Entry widget."""
        from PIL import Image, ImageDraw, ImageTk
        c = tk.Canvas(parent, bg=self.SURF, bd=0, highlightthickness=0, height=height)
        e = tk.Entry(c, textvariable=var,
                     font=("Segoe UI", 11), fg=self.TEXT, bg=self.INP,
                     relief="flat", bd=0,
                     insertbackground=self.PINK,
                     selectbackground=self.BD_LT,
                     selectforeground=self.TEXT)
        win_id = [None]
        focused = [False]

        def _draw(f=False):
            w, h = c.winfo_width(), c.winfo_height()
            if w < 4:
                return
            scale = 2
            img = Image.new("RGB", (w * scale, h * scale), _hex_to_rgb(self.SURF))
            drw = ImageDraw.Draw(img)
            bd_col = _hex_to_rgb(self.PINK if f else self.BD)
            drw.rounded_rectangle([0, 0, w*scale-1, h*scale-1],
                                  radius=radius*scale,
                                  fill=_hex_to_rgb(self.INP),
                                  outline=bd_col, width=scale)
            resample = getattr(getattr(Image, "Resampling", None), "LANCZOS", None) or Image.LANCZOS
            img = img.resize((w, h), resample)
            c._bg = ImageTk.PhotoImage(img)
            c.delete("bg")
            c.create_image(0, 0, anchor="nw", image=c._bg, tags="bg")
            c.tag_lower("bg")
            ey = max(0, (h - 24) // 2)
            if win_id[0] is None:
                win_id[0] = c.create_window(12, ey, anchor="nw", window=e, width=w - 24, height=24)
            else:
                c.coords(win_id[0], 12, ey)
                c.itemconfig(win_id[0], width=w - 24, height=24)

        e.bind("<FocusIn>",   lambda _: (focused.__setitem__(0, True),  _draw(True)))
        e.bind("<FocusOut>",  lambda _: (focused.__setitem__(0, False), _draw(False)))
        c.bind("<Configure>", lambda _: _draw(focused[0]))
        return c

    def _rounded_label_canvas(self, parent, var, radius=10, height=38):
        """Canvas with PIL rounded background containing a read-only Label."""
        from PIL import Image, ImageDraw, ImageTk
        c = tk.Canvas(parent, bg=self.SURF, bd=0, highlightthickness=0, height=height)
        lbl = tk.Label(c, textvariable=var,
                       font=("Segoe UI", 10), fg=self.DIM, bg=self.INP, anchor="w")
        win_id = [None]

        def _draw():
            w, h = c.winfo_width(), c.winfo_height()
            if w < 4:
                return
            scale = 2
            img = Image.new("RGB", (w * scale, h * scale), _hex_to_rgb(self.SURF))
            drw = ImageDraw.Draw(img)
            drw.rounded_rectangle([0, 0, w*scale-1, h*scale-1],
                                  radius=radius*scale,
                                  fill=_hex_to_rgb(self.INP),
                                  outline=_hex_to_rgb(self.BD), width=scale)
            resample = getattr(getattr(Image, "Resampling", None), "LANCZOS", None) or Image.LANCZOS
            img = img.resize((w, h), resample)
            c._bg = ImageTk.PhotoImage(img)
            c.delete("bg")
            c.create_image(0, 0, anchor="nw", image=c._bg, tags="bg")
            c.tag_lower("bg")
            ly = max(0, (h - 20) // 2)
            if win_id[0] is None:
                win_id[0] = c.create_window(12, ly, anchor="nw", window=lbl, width=w - 24)
            else:
                c.coords(win_id[0], 12, ly)
                c.itemconfig(win_id[0], width=w - 24)

        c.bind("<Configure>", lambda _: _draw())
        return c

    # ── row builders ──────────────────────────────────────────────────────────

    def _lbl(self, parent, text):
        tk.Label(parent, text=text,
                 font=("Segoe UI", 10, "bold"),
                 fg=self.DIM, bg=self.SURF,
                 width=13, anchor="w").pack(side="left")

    def _make_audio_row(self, parent):
        row = tk.Frame(parent, bg=self.SURF)
        row.pack(fill="x", pady=(0, 14))

        self._lbl(row, "Audio File")

        inner = tk.Frame(row, bg=self.SURF)
        inner.pack(side="left", fill="x", expand=True)

        name_cv = self._rounded_label_canvas(inner, self._audio_var)
        name_cv.pack(side="left", fill="x", expand=True, padx=(0, 10))

        bw, bh = 82, 36
        br_n = _pill_photo(bw, bh, "Browse", self.SURF2, self.SURF, fg_hex=self.PINK,     font_size=10)
        br_h = _pill_photo(bw, bh, "Browse", self.BD,    self.SURF, fg_hex=self.PINK_LT,  font_size=10)
        br_lbl = tk.Label(inner, image=br_n, bg=self.SURF, cursor="hand2", bd=0)
        br_lbl.pack(side="right")
        br_lbl._imgs = (br_n, br_h)
        br_lbl.bind("<Enter>",    lambda _: br_lbl.config(image=br_h))
        br_lbl.bind("<Leave>",    lambda _: br_lbl.config(image=br_n))
        br_lbl.bind("<Button-1>", lambda _: self._browse())

    def _make_entry_row(self, parent, label, var):
        row = tk.Frame(parent, bg=self.SURF)
        row.pack(fill="x", pady=(0, 14))
        self._lbl(row, label)
        c = self._rounded_entry(row, var)
        c.pack(side="left", fill="x", expand=True)

    def _make_diff_row(self, parent):
        row = tk.Frame(parent, bg=self.SURF)
        row.pack(fill="x", pady=(0, 14))
        self._lbl(row, "Difficulty")

        btns = tk.Frame(row, bg=self.SURF)
        btns.pack(side="left", fill="x", expand=True)

        # compute width: content width minus label column (13 chars × ~8px ≈ 104px)
        # spread 4 buttons across remaining space with 9px gaps
        available = self.CONTENT_W - 104
        bw = (available - 9 * 3) // 4   # ~107px each

        for i, diff in enumerate(("Easy", "Normal", "Hard", "Insane")):
            fill_n, fill_h = self.DIFFS[diff]
            is_active = (diff == "Hard")

            imgs = {
                "active":   _pill_photo(bw, self.DIFF_H, diff, fill_n,   self.SURF, font_size=10),
                "hover":    _pill_photo(bw, self.DIFF_H, diff, fill_h,   self.SURF, font_size=10),
                "inactive": _pill_photo(bw, self.DIFF_H, diff, self.SURF2, self.SURF,
                                        fg_hex=self.MUTE, font_size=10, bold=False),
            }
            self._diff_imgs[diff] = imgs

            lbl = tk.Label(btns, image=imgs["active" if is_active else "inactive"],
                           bg=self.SURF, cursor="hand2", bd=0)
            lbl.pack(side="left", padx=(0, 9 if i < 3 else 0))
            imgs["_lbl"] = lbl   # store ref on the dict

            def _click(d=diff):
                self._selected_diff = d
                for dd, di in self._diff_imgs.items():
                    di["_lbl"].config(image=di["active" if dd == d else "inactive"])

            def _enter(_, d=diff):
                if self._selected_diff != d:
                    self._diff_imgs[d]["_lbl"].config(image=self._diff_imgs[d]["hover"])

            def _leave(_, d=diff):
                di = self._diff_imgs[d]
                di["_lbl"].config(image=di["active" if self._selected_diff == d else "inactive"])

            lbl.bind("<Button-1>", lambda _, d=diff: _click(d))
            lbl.bind("<Enter>",    _enter)
            lbl.bind("<Leave>",    _leave)

    def _make_sv_row(self, parent):
        row = tk.Frame(parent, bg=self.SURF)
        row.pack(fill="x", pady=(0, 4))
        self._lbl(row, "Speed Variation")

        tw, th = 52, 26
        track = tk.Canvas(row, width=tw, height=th,
                          bg=self.SURF, bd=0, highlightthickness=0, cursor="hand2")
        track.pack(side="left")

        def _draw_sv():
            track.delete("all")
            on = self._sv_var.get()
            # track body
            fill = self.PINK if on else self.SURF2
            if True:
                d = th
                track.create_arc(0,    0,    d,    d,    start=90,  extent=90,
                                 style="pieslice", fill=fill, outline="")
                track.create_arc(tw-d, 0,    tw,   d,    start=0,   extent=90,
                                 style="pieslice", fill=fill, outline="")
                track.create_arc(0,    th-d, d,    th,   start=180, extent=90,
                                 style="pieslice", fill=fill, outline="")
                track.create_arc(tw-d, th-d, tw,   th,   start=270, extent=90,
                                 style="pieslice", fill=fill, outline="")
                track.create_rectangle(th//2, 0,  tw-th//2, th, fill=fill, outline="")
                track.create_rectangle(0,     th//2, tw, th-th//2, fill=fill, outline="")
            # knob
            kx = tw - 4 - 18 if on else 4
            track.create_oval(kx, 4, kx + 18, th - 4,
                               fill="white" if on else self.DIM, outline="")

        _draw_sv()
        self._sv_draw = _draw_sv

        def _toggle(_=None):
            self._sv_var.set(not self._sv_var.get())
            _draw_sv()
            _sv_lbl.config(
                text=("Enabled  —  scroll speed follows energy"
                      if self._sv_var.get() else
                      "Disabled  —  constant scroll speed"),
                fg=self.PINK_LT if self._sv_var.get() else self.MUTE,
            )

        track.bind("<Button-1>", _toggle)

        _sv_lbl = tk.Label(row,
                           text="Disabled  —  constant scroll speed",
                           font=("Segoe UI", 10),
                           fg=self.MUTE, bg=self.SURF)
        _sv_lbl.pack(side="left", padx=(14, 0))

    def _make_gen_btn(self, parent):
        gw, gh = self.CONTENT_W, self.GEN_H

        img_normal   = _pill_photo(gw, gh, "✨   Generate Map!", self.PINK,   self.SURF, font_size=13)
        img_hover    = _pill_photo(gw, gh, "✨   Generate Map!", self.PINK_H, self.SURF, font_size=13)
        img_disabled = _pill_photo(gw, gh, "Generating…",       self.PINK_DK, self.SURF,
                                   fg_hex=self.DIM, font_size=13)
        self._gen_imgs = {"normal": img_normal, "hover": img_hover, "disabled": img_disabled}

        self._gen_lbl = tk.Label(parent, image=img_normal,
                                 bg=self.SURF, cursor="hand2", bd=0)
        self._gen_lbl.pack(pady=(6, 0))

        self._gen_lbl.bind("<Enter>",    lambda _: self._gen_lbl.config(
            image=self._gen_imgs["hover"]) if self._gen_enabled else None)
        self._gen_lbl.bind("<Leave>",    lambda _: self._gen_lbl.config(
            image=self._gen_imgs["normal"]) if self._gen_enabled else None)
        self._gen_lbl.bind("<Button-1>", lambda _: self._on_generate() if self._gen_enabled else None)

    def _gen_set_state(self, enabled, text_key="normal"):
        self._gen_enabled = enabled
        self._gen_lbl.config(
            image=self._gen_imgs[text_key],
            cursor="hand2" if enabled else "arrow",
        )

    def _make_progress(self, parent):
        self._prog_outer = tk.Frame(parent, bg=self.SURF)
        self._prog_outer.pack(fill="x", pady=(16, 0))

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Mania.Horizontal.TProgressbar",
                        troughcolor=self.BD, background=self.PINK,
                        borderwidth=0, thickness=10)

        self._prog_var = tk.DoubleVar(value=0)
        self._prog_bar = ttk.Progressbar(self._prog_outer,
                                          variable=self._prog_var,
                                          style="Mania.Horizontal.TProgressbar",
                                          maximum=100)
        self._prog_bar.pack(fill="x", pady=(0, 7))

        stat_row = tk.Frame(self._prog_outer, bg=self.SURF)
        stat_row.pack(fill="x")

        self._status_var = tk.StringVar(value="")
        self._status_lbl = tk.Label(stat_row, textvariable=self._status_var,
                                    font=("Segoe UI", 10), fg=self.DIM,
                                    bg=self.SURF, anchor="w")
        self._status_lbl.pack(side="left", fill="x", expand=True)

        self._pct_var = tk.StringVar(value="")
        tk.Label(stat_row, textvariable=self._pct_var,
                 font=("Segoe UI", 10, "bold"),
                 fg=self.PINK, bg=self.SURF).pack(side="right")

        self._prog_outer.pack_forget()

    # ── logic ─────────────────────────────────────────────────────────────────

    def _browse(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio", "*.mp3 *.ogg *.wav *.flac"), ("All", "*.*")]
        )
        if not path:
            return
        self.audio_path = path
        self._audio_var.set(os.path.basename(path))
        if not self._title_var.get():
            self._title_var.set(os.path.splitext(os.path.basename(path))[0])

    def _on_generate(self):
        from tkinter import messagebox
        if not self.audio_path or not os.path.isfile(self.audio_path):
            messagebox.showerror("No audio file",
                                 "Please select an audio file first.")
            return

        settings = {
            "title":      self._title_var.get().strip() or
                          os.path.splitext(os.path.basename(self.audio_path))[0],
            "artist":     self._artist_var.get().strip() or "Unknown",
            "difficulty": self._selected_diff,
            "sv":         self._sv_var.get(),
            "audio_path": self.audio_path,
        }

        if self._open_btn:
            self._open_btn.destroy()
            self._open_btn = None

        self._gen_set_state(False, "disabled")
        self._prog_var.set(0)
        self._status_var.set("Starting…")
        self._pct_var.set("0%")
        self._status_lbl.config(fg=self.DIM)
        self._prog_outer.pack(fill="x", pady=(16, 0))

        import threading
        threading.Thread(target=self._worker, args=(settings,), daemon=True).start()

    def _worker(self, settings):
        def upd(pct, msg):
            self._q.put(("progress", pct, msg))
        try:
            scripts_dir = os.path.dirname(os.path.abspath(__file__))
            upd(5,  "Loading model…")
            nn = load_nn_model(os.path.join(scripts_dir, "mania_model.pt"))
            if nn is None:
                self._q.put(("error", "Model not found. Run ManiaNNTrainer.py first.")); return

            upd(15, f"Analysing audio  ·  {os.path.basename(settings['audio_path'])}")
            audio = analyze_audio(settings["audio_path"])

            upd(55, "Running Transformer — generating notes…")
            d               = DIFFICULTY_PRESETS[settings["difficulty"]]
            notes, analysis = generate_notes(audio, nn, fill=d["fill"],
                                             difficulty=settings["difficulty"],
                                             max_chord=d["max_chord"])

            upd(78, "Generating SV…" if settings["sv"] else "Finalising…")
            sv = generate_sv_points(audio) if settings["sv"] else []

            upd(90, "Packaging .osz…")
            safe_t   = "".join(c for c in settings["title"]  if c.isalnum() or c in " -_")
            safe_a   = "".join(c for c in settings["artist"] if c.isalnum() or c in " -_")
            out_dir  = os.path.dirname(os.path.abspath(settings["audio_path"]))
            out      = os.path.join(out_dir, f"{safe_a} - {safe_t} [{settings['difficulty']}].osz")
            count    = build_osz(settings, audio, notes, sv, out)
            save_analysis(notes, analysis, audio, settings, out_dir)
            upd(100, f"Done!  {count} notes  ·  {audio['bpm_orig']:.1f} BPM")
            self._q.put(("done", count, out, audio["bpm_orig"]))
        except Exception as exc:
            self._q.put(("error", str(exc)))

    def _poll(self):
        try:
            while True:
                msg  = self._q.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    _, pct, text = msg
                    self._prog_var.set(pct)
                    self._status_var.set(text)
                    self._pct_var.set(f"{int(pct)}%")
                elif kind == "done":
                    _, count, out_path, bpm = msg
                    self._prog_var.set(100)
                    self._pct_var.set("100%")
                    self._gen_set_state(True, "normal")
                    self._on_done(count, out_path, bpm)
                elif kind == "error":
                    from tkinter import messagebox
                    _, err = msg
                    self._gen_set_state(True, "normal")
                    self._status_var.set(f"Error: {err}")
                    self._pct_var.set("")
                    messagebox.showerror("ManiaMapper", err)
        except Exception:
            pass
        self.root.after(80, self._poll)

    def _on_done(self, count, out_path, bpm):
        self._status_var.set(
            f"✓  {count} notes  ·  {bpm:.1f} BPM  ·  {os.path.basename(out_path)}")
        self._status_lbl.config(fg=self.OK)

        self._open_btn = tk.Button(
            self._prog_outer,
            text="Open .osz  ↗",
            command=lambda: os.startfile(out_path),
            font=("Segoe UI", 10, "bold"),
            fg=self.PINK, bg=self.SURF,
            relief="flat", cursor="hand2",
            activeforeground=self.PINK_H, activebackground=self.SURF,
        )
        self._open_btn.pack(anchor="e", pady=(8, 0))


# ─── UI LAUNCHER ─────────────────────────────────────────────────────────────

def _enable_dpi():
    """Make tkinter DPI-aware on Windows so it renders crisp on HiDPI screens."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _launch_ui(audio_path=None):
    _enable_dpi()

    try:
        import tkinter as _tk
        import tkinter.ttk as _ttk
        global tk, ttk
        tk, ttk = _tk, _ttk
    except ImportError:
        print("ERROR: tkinter is not available.")
        sys.exit(1)

    root = tk.Tk()

    # Sync tk scaling to actual screen DPI for sharp fonts
    try:
        dpi = root.winfo_fpixels("1i")
        root.tk.call("tk", "scaling", dpi / 72.0)
    except Exception:
        pass

    ManiaMapperGUI(root, audio_path=audio_path)
    root.mainloop()


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="4K osu!mania map generator")
    parser.add_argument("audio", nargs="?",   help="Audio file (mp3, ogg, wav, flac)")
    parser.add_argument("--output", "-o",     help="Output .osz path")
    parser.add_argument("--nn",               help="Path to mania_model.pt")
    parser.add_argument("--ui", action="store_true", help="Open graphical UI")
    args = parser.parse_args()

    if args.ui:
        if args.audio and not os.path.isfile(args.audio):
            print(f"ERROR: File not found: {args.audio}"); sys.exit(1)
        _launch_ui(args.audio)
        return

    if not args.audio:
        parser.error("audio file is required (or use --ui)")
    if not os.path.isfile(args.audio):
        print(f"ERROR: File not found: {args.audio}"); sys.exit(1)

    scripts_dir   = os.path.dirname(os.path.abspath(__file__))
    nn_model_path = args.nn or os.path.join(scripts_dir, "mania_model.pt")
    if not os.path.isfile(nn_model_path):
        print(f"ERROR: Model not found at {nn_model_path}\nRun ManiaNNTrainer.py first.")
        sys.exit(1)

    print(f"Loading model: {nn_model_path}")
    nn_model = load_nn_model(nn_model_path)
    if nn_model is None:
        print("ERROR: Failed to load model."); sys.exit(1)
    print(f"  Trained on {nn_model['meta'].get('maps_trained', '?')} maps")

    settings = get_user_settings(args.audio)
    settings["audio_path"] = args.audio
    audio_data = analyze_audio(args.audio)

    print("[2/3] Generating notes...")
    d               = DIFFICULTY_PRESETS[settings["difficulty"]]
    notes, analysis = generate_notes(audio_data, nn_model, fill=d["fill"],
                                     difficulty=settings["difficulty"],
                                     max_chord=d["max_chord"])
    print(f"   {len(notes)} notes  |  fill: top {int(d['fill']*100)}%")

    sv_points = generate_sv_points(audio_data) if settings["sv"] else []

    safe_title  = "".join(c for c in settings["title"]  if c.isalnum() or c in " -_")
    safe_artist = "".join(c for c in settings["artist"] if c.isalnum() or c in " -_")
    out_dir     = os.path.dirname(os.path.abspath(args.audio))
    output_path = args.output or os.path.join(
        out_dir, f"{safe_artist} - {safe_title} [{settings['difficulty']}].osz"
    )

    print("[3/3] Building .osz...")
    count = build_osz(settings, audio_data, notes, sv_points, output_path)
    save_analysis(notes, analysis, audio_data, settings, out_dir)
    print(f"\n  Done!  {count} notes  |  BPM: {audio_data['bpm_orig']:.1f}")
    print(f"  Output: {output_path}")
    print(f"\n  Double-click the .osz to import into osu!")


if __name__ == "__main__":
    try:
        import tkinter as tk
        import tkinter.ttk as ttk
    except ImportError:
        tk = ttk = None
    main()
