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
    python ManiaMapper.py --ui
    python ManiaMapper.py song.mp3 --ui
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

    positions = []
    for t_beat in beat_times:
        for s in range(_NN_SUBDIV):
            t = t_beat + s * step_ms
            if t > duration:
                break
            positions.append(t)

    if not positions:
        return []

    features = np.stack([_feat_at(t) for t in positions])
    X        = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(X))[0].cpu().numpy()

    threshold = float(np.percentile(probs, (1.0 - fill) * 100))
    threshold = max(0.20, threshold)

    candidates = [
        (float(probs[i, col]), t, col)
        for i, t in enumerate(positions)
        for col in range(KEYS)
        if probs[i, col] >= threshold
    ]

    candidates.sort(key=lambda c: c[0], reverse=True)

    col_busy = {}
    placed   = []

    for _, t, col in candidates:
        if col_busy.get(col, -9999.0) > t:
            continue
        placed.append((round(t), col, False, 0))
        col_busy[col] = t + MIN_GAP_MS

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


# ─── GUI ─────────────────────────────────────────────────────────────────────

class ManiaMapperGUI:
    # osu!-inspired palette
    PINK       = "#ff66ab"
    PINK_HOV   = "#e0508f"
    PINK_DARK  = "#cc3d7a"
    PINK_LT    = "#ffe6f2"
    HEADER_BG  = "#2d1b33"
    HEADER_SUB = "#c49dbf"
    BG         = "#f7f4fa"
    CARD       = "#ffffff"
    DARK       = "#2d2d2d"
    MUTED      = "#888888"
    BORDER     = "#e6ddf0"
    INPUT_BG   = "#faf6fd"
    SUCCESS    = "#4caf82"

    def __init__(self, root, audio_path=None):
        self.root = root
        self.audio_path = audio_path
        self._open_folder_btn = None

        import queue as _queue
        self._q = _queue.Queue()

        self._setup_window()
        self._build_ui()

        if audio_path:
            self._audio_var.set(os.path.basename(audio_path))
            stem = os.path.splitext(os.path.basename(audio_path))[0]
            self._title_var.set(stem)

        self._poll_queue()

    def _setup_window(self):
        self.root.title("ManiaMapper")
        W, H = 480, 580
        self.root.geometry(f"{W}x{H}")
        self.root.resizable(False, False)
        self.root.configure(bg=self.BG)
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{W}x{H}+{(sw-W)//2}+{(sh-H)//2}")

    # ── layout ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()
        self._build_card()

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=self.HEADER_BG, height=110)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        # pink dot accent row
        dot_row = tk.Frame(hdr, bg=self.HEADER_BG)
        dot_row.pack(pady=(14, 0))
        for _ in range(3):
            tk.Label(dot_row, text="●", font=("Segoe UI", 7),
                     fg=self.PINK, bg=self.HEADER_BG).pack(side="left", padx=3)

        tk.Label(hdr, text="ManiaMapper",
                 font=("Segoe UI", 26, "bold"),
                 fg=self.PINK, bg=self.HEADER_BG).pack(pady=(4, 0))
        tk.Label(hdr, text="osu!mania AI Map Generator  ·  4K",
                 font=("Segoe UI", 9),
                 fg=self.HEADER_SUB, bg=self.HEADER_BG).pack()

    def _build_card(self):
        import tkinter.ttk as ttk

        # outer frame with soft shadow effect (border trick)
        outer = tk.Frame(self.root, bg=self.BORDER)
        outer.pack(fill="both", expand=True, padx=18, pady=18)

        card = tk.Frame(outer, bg=self.CARD, padx=28, pady=22)
        card.pack(fill="both", expand=True, padx=1, pady=1)

        # ── audio file row ───────────────────────────────────────────────
        self._audio_var = tk.StringVar(value="No file selected")
        self._make_row(card, "Audio File", widget=self._build_audio_row(card))

        # divider
        tk.Frame(card, bg=self.BORDER, height=1).pack(fill="x", pady=(4, 14))

        # ── fields ──────────────────────────────────────────────────────
        self._title_var  = tk.StringVar()
        self._artist_var = tk.StringVar(value="Unknown")
        self._make_entry_row(card, "Song Title", self._title_var)
        self._make_entry_row(card, "Artist",     self._artist_var)

        # ── difficulty ──────────────────────────────────────────────────
        self._diff_var = tk.StringVar(value="Hard")
        diff_frame = tk.Frame(card, bg=self.CARD)
        self._make_row(card, "Difficulty", widget=self._build_diff_row(card))

        # ── SV toggle ───────────────────────────────────────────────────
        self._sv_var = tk.BooleanVar(value=False)
        self._make_row(card, "Speed Variation", widget=self._build_sv_row(card))

        # spacer
        tk.Frame(card, bg=self.CARD, height=10).pack()

        # ── generate button ─────────────────────────────────────────────
        self._gen_btn = tk.Button(
            card, text="✨   Generate Map",
            command=self._on_generate,
            font=("Segoe UI", 13, "bold"),
            fg="white", bg=self.PINK,
            activebackground=self.PINK_HOV, activeforeground="white",
            relief="flat", cursor="hand2",
            pady=11, padx=20,
        )
        self._gen_btn.pack(fill="x", pady=(0, 14))
        self._bind_hover(self._gen_btn, self.PINK, self.PINK_HOV)

        # ── progress area ───────────────────────────────────────────────
        self._prog_outer = tk.Frame(card, bg=self.CARD)
        self._prog_outer.pack(fill="x")

        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "Mania.Horizontal.TProgressbar",
            troughcolor=self.BORDER,
            background=self.PINK,
            borderwidth=0,
            thickness=10,
        )

        self._prog_var = tk.DoubleVar(value=0)
        self._prog_bar = ttk.Progressbar(
            self._prog_outer,
            variable=self._prog_var,
            style="Mania.Horizontal.TProgressbar",
            maximum=100,
        )
        self._prog_bar.pack(fill="x", pady=(0, 5))

        status_row = tk.Frame(self._prog_outer, bg=self.CARD)
        status_row.pack(fill="x")

        self._status_var = tk.StringVar(value="")
        tk.Label(status_row, textvariable=self._status_var,
                 font=("Segoe UI", 9), fg=self.MUTED,
                 bg=self.CARD, anchor="w").pack(side="left", fill="x", expand=True)

        self._pct_var = tk.StringVar(value="")
        tk.Label(status_row, textvariable=self._pct_var,
                 font=("Segoe UI", 9, "bold"), fg=self.PINK,
                 bg=self.CARD, anchor="e").pack(side="right")

        self._prog_outer.pack_forget()

    # ── sub-widget builders ──────────────────────────────────────────────────

    def _build_audio_row(self, parent):
        f = tk.Frame(parent, bg=self.CARD)
        lbl = tk.Label(f, textvariable=self._audio_var,
                       font=("Segoe UI", 9), fg=self.DARK,
                       bg=self.INPUT_BG, anchor="w", padx=8,
                       relief="flat",
                       highlightthickness=1, highlightbackground=self.BORDER)
        lbl.pack(side="left", fill="x", expand=True, ipady=5)

        btn = tk.Button(f, text="Browse…",
                        command=self._browse,
                        font=("Segoe UI", 9), fg=self.PINK, bg=self.CARD,
                        relief="flat", cursor="hand2", padx=6,
                        activeforeground=self.PINK_DARK, activebackground=self.CARD)
        btn.pack(side="right", padx=(6, 0))
        return f

    def _build_diff_row(self, parent):
        f = tk.Frame(parent, bg=self.CARD)
        for diff in ("Easy", "Normal", "Hard", "Insane"):
            self._make_diff_btn(f, diff)
        return f

    def _make_diff_btn(self, parent, diff):
        colors = {
            "Easy":   "#5cb85c",
            "Normal": "#5bc0de",
            "Hard":   "#f0ad4e",
            "Insane": self.PINK,
        }
        c = colors[diff]

        def select():
            self._diff_var.set(diff)
            for d, b in self._diff_btns.items():
                active = d == diff
                b.config(
                    bg=colors[d] if active else self.CARD,
                    fg="white"   if active else self.MUTED,
                    relief="flat",
                    highlightbackground=colors[d] if active else self.BORDER,
                )

        if not hasattr(self, "_diff_btns"):
            self._diff_btns = {}

        btn = tk.Button(
            parent, text=diff,
            command=select,
            font=("Segoe UI", 9, "bold"),
            fg=self.MUTED if diff != "Hard" else "white",
            bg=colors["Hard"] if diff == "Hard" else self.CARD,
            relief="flat", cursor="hand2", padx=10, pady=5,
            highlightthickness=1,
            highlightbackground=c if diff == "Hard" else self.BORDER,
        )
        btn.pack(side="left", padx=(0, 6))
        self._diff_btns[diff] = btn

    def _build_sv_row(self, parent):
        f = tk.Frame(parent, bg=self.CARD)

        self._sv_toggle_bg = tk.Frame(f, bg=self.BORDER, width=44, height=22,
                                       cursor="hand2")
        self._sv_toggle_bg.pack(side="left")
        self._sv_toggle_bg.pack_propagate(False)

        self._sv_knob = tk.Frame(self._sv_toggle_bg, bg=self.MUTED,
                                  width=18, height=18)
        self._sv_knob.place(x=2, y=2)

        self._sv_lbl = tk.Label(f, text="Disabled — notes use constant scroll speed",
                                 font=("Segoe UI", 9), fg=self.MUTED, bg=self.CARD)
        self._sv_lbl.pack(side="left", padx=(10, 0))

        self._sv_toggle_bg.bind("<Button-1>", lambda _: self._toggle_sv())
        self._sv_knob.bind("<Button-1>",      lambda _: self._toggle_sv())
        return f

    # ── helpers ──────────────────────────────────────────────────────────────

    def _make_row(self, parent, label, widget):
        row = tk.Frame(parent, bg=self.CARD)
        row.pack(fill="x", pady=(0, 10))
        tk.Label(row, text=label,
                 font=("Segoe UI", 9, "bold"), fg=self.MUTED,
                 bg=self.CARD, width=14, anchor="w").pack(side="left")
        widget.pack(side="left", fill="x", expand=True)

    def _make_entry_row(self, parent, label, var):
        row = tk.Frame(parent, bg=self.CARD)
        row.pack(fill="x", pady=(0, 10))
        tk.Label(row, text=label,
                 font=("Segoe UI", 9, "bold"), fg=self.MUTED,
                 bg=self.CARD, width=14, anchor="w").pack(side="left")
        e = tk.Entry(row, textvariable=var,
                     font=("Segoe UI", 10), fg=self.DARK,
                     bg=self.INPUT_BG, relief="flat",
                     highlightthickness=1, highlightbackground=self.BORDER,
                     highlightcolor=self.PINK,
                     insertbackground=self.PINK)
        e.pack(side="left", fill="x", expand=True, ipady=5)

    def _bind_hover(self, widget, normal, hover):
        widget.bind("<Enter>", lambda _: widget.config(bg=hover))
        widget.bind("<Leave>", lambda _: widget.config(bg=normal))

    def _toggle_sv(self):
        new_val = not self._sv_var.get()
        self._sv_var.set(new_val)
        if new_val:
            self._sv_toggle_bg.config(bg=self.PINK)
            self._sv_knob.config(bg="white")
            self._sv_knob.place(x=24, y=2)
            self._sv_lbl.config(
                text="Enabled — scroll speed follows song energy",
                fg=self.PINK)
        else:
            self._sv_toggle_bg.config(bg=self.BORDER)
            self._sv_knob.config(bg=self.MUTED)
            self._sv_knob.place(x=2, y=2)
            self._sv_lbl.config(
                text="Disabled — notes use constant scroll speed",
                fg=self.MUTED)

    def _browse(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio files", "*.mp3 *.ogg *.wav *.flac"), ("All", "*.*")]
        )
        if not path:
            return
        self.audio_path = path
        self._audio_var.set(os.path.basename(path))
        if not self._title_var.get():
            self._title_var.set(os.path.splitext(os.path.basename(path))[0])

    # ── generation ───────────────────────────────────────────────────────────

    def _on_generate(self):
        from tkinter import messagebox
        if not self.audio_path or not os.path.isfile(self.audio_path):
            messagebox.showerror("No audio", "Please select an audio file first.")
            return

        settings = {
            "title":      self._title_var.get().strip() or
                          os.path.splitext(os.path.basename(self.audio_path))[0],
            "artist":     self._artist_var.get().strip() or "Unknown",
            "difficulty": self._diff_var.get(),
            "sv":         self._sv_var.get(),
            "audio_path": self.audio_path,
        }

        # reset state
        if self._open_folder_btn:
            self._open_folder_btn.destroy()
            self._open_folder_btn = None

        self._gen_btn.config(state="disabled", text="Generating…", bg="#cc5590")
        self._prog_var.set(0)
        self._status_var.set("Starting…")
        self._pct_var.set("0%")
        self._prog_outer.pack(fill="x")

        import threading
        threading.Thread(target=self._worker, args=(settings,), daemon=True).start()

    def _worker(self, settings):
        def upd(pct, msg):
            self._q.put(("progress", pct, msg))

        try:
            scripts_dir   = os.path.dirname(os.path.abspath(__file__))
            nn_model_path = os.path.join(scripts_dir, "mania_model.pt")

            upd(5,  "Loading model…")
            nn = load_nn_model(nn_model_path)
            if nn is None:
                self._q.put(("error", "Model not found. Run ManiaNNTrainer.py first."))
                return

            upd(15, f"Analysing audio  ·  {os.path.basename(settings['audio_path'])}")
            audio = analyze_audio(settings["audio_path"])

            upd(55, "Running LSTM — generating notes…")
            d     = DIFFICULTY_PRESETS[settings["difficulty"]]
            notes = generate_notes(audio, nn, fill=d["fill"])

            upd(78, "Generating SV points…" if settings["sv"] else "Skipping SV…")
            sv = generate_sv_points(audio) if settings["sv"] else []

            upd(90, "Packaging .osz…")
            safe_t = "".join(c for c in settings["title"]  if c.isalnum() or c in " -_")
            safe_a = "".join(c for c in settings["artist"] if c.isalnum() or c in " -_")
            out    = os.path.join(
                os.path.dirname(os.path.abspath(settings["audio_path"])),
                f"{safe_a} - {safe_t} [{settings['difficulty']}].osz"
            )
            count = build_osz(settings, audio, notes, sv, out)

            upd(100, f"Done!  {count} notes  ·  {audio['bpm_orig']:.1f} BPM")
            self._q.put(("done", count, out, audio["bpm_orig"]))

        except Exception as exc:
            self._q.put(("error", str(exc)))

    def _poll_queue(self):
        try:
            import queue as _queue
            while True:
                msg = self._q.get_nowait()
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
                    self._gen_btn.config(
                        state="normal", text="✨   Generate Map", bg=self.PINK)
                    self._on_done(count, out_path, bpm)
                elif kind == "error":
                    from tkinter import messagebox
                    _, err = msg
                    self._gen_btn.config(
                        state="normal", text="✨   Generate Map", bg=self.PINK)
                    self._status_var.set(f"Error: {err}")
                    self._pct_var.set("")
                    messagebox.showerror("ManiaMapper error", err)
        except Exception:
            pass
        self.root.after(80, self._poll_queue)

    def _on_done(self, count, out_path, bpm):
        self._status_var.set(
            f"✓  {count} notes  ·  {bpm:.1f} BPM  ·  {os.path.basename(out_path)}")
        self._status_var_lbl_color(self.SUCCESS)

        self._open_folder_btn = tk.Button(
            self._prog_outer,
            text="Open folder  ↗",
            command=lambda: os.startfile(os.path.dirname(out_path)),
            font=("Segoe UI", 9, "bold"),
            fg=self.PINK, bg=self.CARD,
            relief="flat", cursor="hand2",
            activeforeground=self.PINK_DARK, activebackground=self.CARD,
        )
        self._open_folder_btn.pack(anchor="e", pady=(4, 0))

    def _status_var_lbl_color(self, color):
        # walk the prog_outer children to find the status label
        for child in self._prog_outer.winfo_children():
            if isinstance(child, tk.Frame):
                for sub in child.winfo_children():
                    if isinstance(sub, tk.Label) and sub.cget("textvariable") == str(self._status_var):
                        sub.config(fg=color)
                        return


# ─── UI LAUNCHER ─────────────────────────────────────────────────────────────

def _launch_ui(audio_path=None):
    try:
        import tkinter as tk
    except ImportError:
        print("ERROR: tkinter is not available in this Python installation.")
        sys.exit(1)

    root = tk.Tk()
    ManiaMapperGUI(root, audio_path=audio_path)
    root.mainloop()


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="4K osu!mania map generator")
    parser.add_argument("audio", nargs="?",    help="Audio file (mp3, ogg, wav, flac)")
    parser.add_argument("--output", "-o",      help="Output .osz path")
    parser.add_argument("--nn",                help="Path to mania_model.pt")
    parser.add_argument("--ui", action="store_true",
                        help="Open graphical UI instead of CLI prompts")
    args = parser.parse_args()

    # ── GUI mode ──────────────────────────────────────────────────────────────
    if args.ui:
        if args.audio and not os.path.isfile(args.audio):
            print(f"ERROR: File not found: {args.audio}")
            sys.exit(1)
        _launch_ui(args.audio)
        return

    # ── CLI mode ──────────────────────────────────────────────────────────────
    if not args.audio:
        parser.error("audio file is required (or use --ui for the graphical interface)")

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
    # need tk at module level for ManiaMapperGUI class body references
    try:
        import tkinter as tk
        import tkinter.ttk as ttk
    except ImportError:
        tk = None
        ttk = None
    main()
