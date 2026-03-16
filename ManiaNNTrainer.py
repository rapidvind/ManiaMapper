"""
ManiaNNTrainer.py  —  osu!mania 4K trainer  (v2)
=================================================
Key improvements over v1
-------------------------
  - Features: 80-band mel spectrogram + onset strength instead of 20 MFCCs
    (onset strength is the single most important signal for note placement)
  - SUBDIV = 8 (32nd-note grid, double the resolution of v1's 16th notes)
  - Beat grid: uniform from first beat, no drift accumulation
  - Model: Transformer encoder with difficulty embedding
    (sees full sequence context instead of LSTM's short memory)
  - Difficulty conditioning: parsed from Version string + density fallback
  - Dynamic pos_weight per batch (adapts to actual note density)
  - sr = 22050 fixed everywhere (eliminates inference/training mismatch)

Usage
-----
    python ManiaNNTrainer.py
    python ManiaNNTrainer.py --maps-dir C:\\ManiaStyles --epochs 30
    python ManiaNNTrainer.py --maps-dir C:\\ManiaStyles --epochs 30 --max-maps 3000

Requirements: pip install torch librosa numpy tqdm
"""

import os, sys, argparse, warnings
import numpy as np

warnings.filterwarnings("ignore")

# ── Feature / grid config (must match ManiaMapper.py exactly) ─────────────────
N_MEL      = 80           # mel spectrogram bins
SUBDIV     = 8            # subdivisions per beat  (8 = 32nd-note grid)
HOP        = 512          # librosa hop length
SR         = 22050        # fixed sample rate — must match ManiaMapper.py
FEAT_DIM   = N_MEL + 3   # mel(80) + onset(1) + sin_phase(1) + cos_phase(1) = 83
SEQ_LEN    = 256          # training sequence length
DIFF_LEVELS = 4           # Easy / Normal / Hard / Insane
DIFF_EMB   = 8            # embedding dimension for difficulty

# Transformer hyper-parameters
D_MODEL  = 256
NHEAD    = 8
N_LAYERS = 4
DIM_FF   = 1024
DROPOUT  = 0.1

DEFAULT_MAPS_DIR = r"C:\Users\Aravind Dora\Desktop\ManiaMapper\ManiaStyles"
DEFAULT_OUT      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mania_model.pt")


# ── osu! parser ───────────────────────────────────────────────────────────────

def x_to_col(x):
    return min(3, max(0, x * 4 // 512))


def version_to_diff(version_str):
    """Map Version string to 0-3 difficulty level. Returns -1 if unknown."""
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
    return -1   # unknown — will fall back to density-based assignment


def parse_osu(path):
    """
    Parse a 4K .osu file.
    Returns (audio_filename, groups, version_str) or None if not usable.
    groups = list of (time_ms, [cols])
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
                    if len(parts) < 3:
                        continue
                    try:
                        raw_notes.append((int(parts[2]), x_to_col(int(parts[0]))))
                    except ValueError:
                        continue
    except Exception:
        return None

    if mode != 3 or keys != 4.0 or not audio_file or not raw_notes:
        return None

    raw_notes.sort()
    groups = []
    i = 0
    while i < len(raw_notes):
        t0, c0 = raw_notes[i]
        cols = {c0}
        j = i + 1
        while j < len(raw_notes) and raw_notes[j][0] - t0 <= 10:
            cols.add(raw_notes[j][1])
            j += 1
        groups.append((t0, sorted(cols)))
        i = j

    return audio_file, groups, (version or "")


# ── Audio feature extraction ──────────────────────────────────────────────────

def extract_features_and_labels(audio_path, groups, version_str):
    """
    Returns (features, labels, diff_level) or (None, None, None).
      features : (T, FEAT_DIM)  float32 — mel + onset + phase at each 32nd-note step
      labels   : (T, 4)         float32 — which columns have a note at each step
      diff_level: int 0-3
    """
    try:
        import librosa
    except ImportError:
        print("ERROR: pip install librosa"); sys.exit(1)

    try:
        y, sr = librosa.load(audio_path, sr=SR, mono=True)
    except Exception:
        return None, None, None

    duration_ms = len(y) / SR * 1000

    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=SR)
        beat_times = librosa.frames_to_time(beat_frames, sr=SR) * 1000
        bpm = float(np.atleast_1d(tempo)[0])
    except Exception:
        return None, None, None

    if len(beat_times) == 0 or bpm < 60 or bpm > 400:
        return None, None, None

    beat_length = 60000.0 / bpm
    step_ms     = beat_length / SUBDIV

    # ── mel spectrogram (80 bins, normalised) ─────────────────────────────────
    try:
        mel     = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MEL, hop_length=HOP)
        mel_db  = librosa.power_to_db(mel, ref=np.max)
        mel_db  = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
    except Exception:
        return None, None, None

    # ── onset strength (single most important feature for note timing) ─────────
    try:
        onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
        onset = (onset - onset.mean()) / (onset.std() + 1e-9)
    except Exception:
        return None, None, None

    n_frames    = mel_db.shape[1]
    frame_times = librosa.frames_to_time(np.arange(n_frames), sr=SR, hop_length=HOP) * 1000

    # ── uniform beat grid (no drift) ──────────────────────────────────────────
    beat0_ms  = float(beat_times[0])
    positions = []
    t = beat0_ms
    while t <= duration_ms:
        positions.append(t)
        t += step_ms

    if len(positions) < 64:
        return None, None, None

    def feat_at(t_ms):
        idx = min(int(np.searchsorted(frame_times, t_ms)), n_frames - 1)
        f   = np.zeros(FEAT_DIM, dtype=np.float32)
        f[:N_MEL]     = mel_db[:, idx]
        f[N_MEL]      = onset[idx]
        phase          = (t_ms % (beat_length * 4)) / (beat_length * 4 + 1e-9)
        f[N_MEL + 1]  = float(np.sin(2 * np.pi * phase))
        f[N_MEL + 2]  = float(np.cos(2 * np.pi * phase))
        return f

    features = np.stack([feat_at(t) for t in positions])
    labels   = np.zeros((len(positions), 4), dtype=np.float32)
    pos_arr  = np.array(positions)
    TOL_MS   = max(step_ms * 0.6, 20.0)

    for t_note, cols in groups:
        dists   = np.abs(pos_arr - t_note)
        closest = int(np.argmin(dists))
        if dists[closest] <= TOL_MS:
            for c in cols:
                labels[closest, c] = 1.0

    note_ratio = labels.sum() / (labels.size + 1e-9)
    if note_ratio < 0.02 or note_ratio > 0.80:
        return None, None, None

    # ── difficulty level ───────────────────────────────────────────────────────
    diff_level = version_to_diff(version_str)
    if diff_level < 0:
        # Fall back to measured note density
        if   note_ratio < 0.08: diff_level = 0
        elif note_ratio < 0.18: diff_level = 1
        elif note_ratio < 0.28: diff_level = 2
        else:                   diff_level = 3

    return features, labels, diff_level


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(feat_dim=FEAT_DIM, diff_levels=DIFF_LEVELS, diff_emb=DIFF_EMB,
                d_model=D_MODEL, nhead=NHEAD, num_layers=N_LAYERS,
                dim_ff=DIM_FF, dropout=DROPOUT):
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("ERROR: pip install torch"); sys.exit(1)

    class PositionalEncoding(nn.Module):
        def __init__(self, d, max_len=2048, drop=0.1):
            super().__init__()
            self.drop = nn.Dropout(drop)
            pe   = torch.zeros(max_len, d)
            pos  = torch.arange(0, max_len).unsqueeze(1).float()
            div  = torch.exp(torch.arange(0, d, 2).float() * (-9.21034 / d))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return self.drop(x + self.pe[:, :x.size(1)])

    class ManiaTransformer(nn.Module):
        """
        Transformer that maps (audio features + difficulty) → note probabilities.

        Advantages over the previous LSTM:
          - Self-attention sees the entire sequence at once → learns verse/chorus patterns
          - Difficulty embedding conditions note density on selected difficulty
          - Mel spectrogram + onset strength features give direct signal for note placement

        Input : x     (batch, time, feat_dim)  — audio features at each 32nd-note step
                diff  (batch,)                 — difficulty index 0-3
        Output: (batch, time, 4)               — logits for 4 columns
        """
        def __init__(self):
            super().__init__()
            self.diff_emb   = nn.Embedding(diff_levels, diff_emb)
            self.input_proj = nn.Linear(feat_dim + diff_emb, d_model)
            self.pos_enc    = PositionalEncoding(d_model, drop=dropout)
            enc_layer       = nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward=dim_ff,
                dropout=dropout, batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
            self.norm        = nn.LayerNorm(d_model)
            self.fc          = nn.Linear(d_model, 4)

        def forward(self, x, diff):
            B, T, _ = x.shape
            d = self.diff_emb(diff).unsqueeze(1).expand(B, T, -1)
            x = self.input_proj(torch.cat([x, d], dim=-1))
            x = self.pos_enc(x)
            x = self.transformer(x)
            return self.fc(self.norm(x))

    return ManiaTransformer()


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
            for feats, lbls, diff_level in seqs:
                T = len(feats)
                for start in range(0, T - seq_len, stride):
                    end = start + seq_len
                    self.samples.append((
                        torch.tensor(feats[start:end], dtype=torch.float32),
                        torch.tensor(lbls[start:end],  dtype=torch.float32),
                        torch.tensor(diff_level,        dtype=torch.long),
                    ))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            return self.samples[i]

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

    sequences  = []
    maps_tried = 0
    diff_counts = [0, 0, 0, 0]

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

        feats, lbls, diff_level = extract_features_and_labels(audio_path, groups, version_str)
        if feats is None:
            continue

        sequences.append((feats, lbls, diff_level))
        diff_counts[diff_level] += 1

    print(f"\nUsable maps : {len(sequences)} / {maps_tried}")
    print(f"  Easy:{diff_counts[0]}  Normal:{diff_counts[1]}  Hard:{diff_counts[2]}  Insane:{diff_counts[3]}")

    if len(sequences) < 5:
        print("ERROR: Not enough usable maps. Make sure audio files are next to the .osu files.")
        sys.exit(1)

    dataset = make_dataset(sequences)
    loader  = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
    print(f"Dataset     : {len(dataset)} sequences × {SEQ_LEN} steps\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model()
    model.to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model       : ManiaTransformer  |  device: {device}  |  params: {params:,}")
    print(f"Grid        : SUBDIV={SUBDIV} (32nd notes)  |  features: mel({N_MEL}) + onset + phase = {FEAT_DIM}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for X, y_note, diff in loader:
            X, y_note, diff = X.to(device), y_note.to(device), diff.to(device)

            # Dynamic pos_weight: adapts to actual note density in this batch
            note_mean = y_note.mean().item()
            pw_val    = min((1.0 - note_mean) / (note_mean + 1e-6), 15.0)
            pw_val    = max(pw_val, 2.0)
            pos_weight = torch.tensor([pw_val] * 4, device=device)
            criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            optimizer.zero_grad()
            loss = criterion(model(X, diff), y_note)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch:3d}/{epochs}  loss: {avg_loss:.4f}  lr: {scheduler.get_last_lr()[0]:.2e}")

    torch.save({
        "model_state":  model.state_dict(),
        "model_type":   "ManiaTransformer",
        "feat_dim":     FEAT_DIM,
        "n_mel":        N_MEL,
        "subdiv":       SUBDIV,
        "d_model":      D_MODEL,
        "nhead":        NHEAD,
        "num_layers":   N_LAYERS,
        "dim_ff":       DIM_FF,
        "diff_levels":  DIFF_LEVELS,
        "diff_emb":     DIFF_EMB,
        "dropout":      DROPOUT,
        "maps_trained": len(sequences),
    }, out_path)

    print(f"\nModel saved → {out_path}")
    print(f"  Maps : {len(sequences)}   Sequences : {len(dataset)}")
    print(f'\nNext: python ManiaMapper.py "song.mp3" --ui')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train osu!mania 4K Transformer")
    parser.add_argument("--maps-dir", default=DEFAULT_MAPS_DIR)
    parser.add_argument("--out",      default=DEFAULT_OUT)
    parser.add_argument("--max-maps", type=int, default=3000)
    parser.add_argument("--epochs",   type=int, default=30)
    args = parser.parse_args()

    if not os.path.isdir(args.maps_dir):
        print(f"ERROR: Directory not found: {args.maps_dir}"); sys.exit(1)

    print(f"Maps dir : {args.maps_dir}")
    print(f"Output   : {args.out}")
    print(f"Max maps : {args.max_maps}  |  Epochs : {args.epochs}")
    train(args.maps_dir, args.out, args.max_maps, args.epochs)


if __name__ == "__main__":
    main()
