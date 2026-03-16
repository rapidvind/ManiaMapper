"""
ManiaNNTrainer.py  —  osu!mania 4K trainer  (v3)
=================================================
What's new in v3
-----------------
  Features (108 audio + 16 cross-diff context = 124 total):
    mel spectrogram (80)        — tonal content, main signal
    onset strength (1)          — WHEN hits occur
    spectral contrast (7)       — peak/valley texture (percussive vs tonal)
    chroma CQT (12)             — harmonic/melodic content, phrase detection
    RMS energy (1)              — loudness → note density signal
    spectral flatness (1)       — noisy (drums) vs tonal (melody)
    phase @ beat/2beat/4beat    — rhythmic position at 3 resolutions (6)
    cross-diff context (16)     — sibling map labels for same audio
                                  (4 difficulties × 4 columns)
                                  zeros when not available

  Architecture: PatternConvBlock + Transformer + 4 difficulty heads
    PatternConvBlock: multi-scale 1D depthwise convs (k=3,5,9) explicitly
      model streams/jacks/trills before the attention layers
    4 separate output heads (one per difficulty) share the encoder but
      learn distinct note-density / pattern signatures per difficulty

  Training:
    Multi-task loss: target diff (weight=1.0) + available sibling diffs
      (weight=0.25) → cross-difficulty pattern transfer
    Dynamic pos_weight, gradient clipping, CosineAnnealingLR
    sr=22050 fixed, SUBDIV=8 (32nd-note training grid)

Usage
-----
    python ManiaNNTrainer.py
    python ManiaNNTrainer.py --maps-dir C:\\ManiaStyles --epochs 40
    python ManiaNNTrainer.py --epochs 40 --max-maps 3000
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
FEAT_DIM_AUDIO = N_MEL + 1 + N_SC + N_CHROMA + 1 + 1 + 6   # = 108
FEAT_DIM_CTX   = 16       # 4 diffs × 4 cols cross-difficulty context
FEAT_DIM       = FEAT_DIM_AUDIO + FEAT_DIM_CTX               # = 124
SEQ_LEN     = 256
DIFF_LEVELS = 4
DIFF_EMB    = 16

# Transformer hyper-parameters
D_MODEL  = 256
NHEAD    = 8
N_LAYERS = 4
DIM_FF   = 1024
DROPOUT  = 0.15

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
    groups, i = [], 0
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

def extract_audio_features(audio_path):
    """
    Extract all audio features for a single audio file.
    Returns (feat_audio, frame_times, beat_length, duration_ms, positions) or None.
    feat_audio : (T_grid, FEAT_DIM_AUDIO)  float32
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

    # ── mel spectrogram ────────────────────────────────────────────────────────
    try:
        mel    = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MEL, hop_length=HOP)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
    except Exception:
        return None

    # ── onset strength ─────────────────────────────────────────────────────────
    try:
        onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
        onset = (onset - onset.mean()) / (onset.std() + 1e-9)
    except Exception:
        return None

    # ── spectral contrast ──────────────────────────────────────────────────────
    try:
        sc = librosa.feature.spectral_contrast(y=y, sr=SR, hop_length=HOP)  # (7, T)
        sc = (sc - sc.mean(axis=1, keepdims=True)) / (sc.std(axis=1, keepdims=True) + 1e-9)
    except Exception:
        sc = np.zeros((N_SC, mel_db.shape[1]), dtype=np.float32)

    # ── chroma CQT ─────────────────────────────────────────────────────────────
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP)  # (12, T)
        chroma = (chroma - chroma.mean(axis=1, keepdims=True)) / (chroma.std(axis=1, keepdims=True) + 1e-9)
    except Exception:
        chroma = np.zeros((N_CHROMA, mel_db.shape[1]), dtype=np.float32)

    # ── RMS energy ─────────────────────────────────────────────────────────────
    try:
        rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
        rms = (rms - rms.mean()) / (rms.std() + 1e-9)
    except Exception:
        rms = np.zeros(mel_db.shape[1], dtype=np.float32)

    # ── spectral flatness ──────────────────────────────────────────────────────
    try:
        flat = librosa.feature.spectral_flatness(y=y, hop_length=HOP)[0]
        flat = (flat - flat.mean()) / (flat.std() + 1e-9)
    except Exception:
        flat = np.zeros(mel_db.shape[1], dtype=np.float32)

    n_frames    = mel_db.shape[1]
    frame_times = librosa.frames_to_time(np.arange(n_frames), sr=SR, hop_length=HOP) * 1000

    # ── uniform beat grid ──────────────────────────────────────────────────────
    beat0_ms  = float(beat_times[0])
    positions = []
    t = beat0_ms
    while t <= duration_ms:
        positions.append(t)
        t += step_ms

    if len(positions) < 64:
        return None

    # ── build feature matrix at each grid position ────────────────────────────
    def feat_at(t_ms):
        idx = min(int(np.searchsorted(frame_times, t_ms)), n_frames - 1)
        f   = np.zeros(FEAT_DIM_AUDIO, dtype=np.float32)
        f[:N_MEL]                               = mel_db[:, idx]
        f[N_MEL]                                = onset[idx]
        f[N_MEL+1 : N_MEL+1+N_SC]              = sc[:, idx]
        f[N_MEL+1+N_SC : N_MEL+1+N_SC+N_CHROMA] = chroma[:, idx]
        f[N_MEL+1+N_SC+N_CHROMA]               = rms[idx]
        f[N_MEL+1+N_SC+N_CHROMA+1]             = flat[idx]
        # multi-scale phase: beat, 2-beat, 4-beat
        for k, scale in enumerate([1, 2, 4]):
            period = beat_length * scale
            phase  = (t_ms % period) / (period + 1e-9)
            base   = N_MEL + 1 + N_SC + N_CHROMA + 2 + k * 2
            f[base]   = float(np.sin(2 * np.pi * phase))
            f[base+1] = float(np.cos(2 * np.pi * phase))
        return f

    feat_audio = np.stack([feat_at(t) for t in positions])   # (T, 108)
    return feat_audio, frame_times, beat_length, duration_ms, positions, beat_times


def extract_labels(groups, positions, beat_length):
    """Map hit groups → binary label matrix (T, 4)."""
    step_ms = beat_length / SUBDIV
    TOL_MS  = max(step_ms * 0.6, 20.0)
    labels  = np.zeros((len(positions), 4), dtype=np.float32)
    pos_arr = np.array(positions)
    for t_note, cols in groups:
        dists   = np.abs(pos_arr - t_note)
        closest = int(np.argmin(dists))
        if dists[closest] <= TOL_MS:
            for c in cols:
                labels[closest, c] = 1.0
    return labels


def extract_features_and_labels(audio_path, groups, version_str):
    """
    Returns (feat_full, labels, diff_level, audio_path) or (None, None, None, None).
    feat_full : (T, FEAT_DIM_AUDIO) — cross-diff context added later
    """
    result = extract_audio_features(audio_path)
    if result is None:
        return None, None, None, None

    feat_audio, _, beat_length, _, positions, _ = result
    labels = extract_labels(groups, positions, beat_length)

    note_ratio = labels.sum() / (labels.size + 1e-9)
    if note_ratio < 0.02 or note_ratio > 0.80:
        return None, None, None, None

    diff_level = version_to_diff(version_str)
    if diff_level < 0:
        if   note_ratio < 0.08: diff_level = 0
        elif note_ratio < 0.18: diff_level = 1
        elif note_ratio < 0.28: diff_level = 2
        else:                   diff_level = 3

    return feat_audio, labels, diff_level, audio_path


# ── Cross-difficulty context ──────────────────────────────────────────────────

def build_cross_diff_context(sequences):
    """
    Group sequences by audio_path.  For each sequence, append 16 features
    representing sibling difficulty labels at each position:
        [Easy_C1..Easy_C4 | Normal_C1..C4 | Hard_C1..C4 | Insane_C1..C4]
    Where a difficulty is absent, the 4 values are zero.
    Returns list of (feat_full, labels, diff_level).
    """
    from collections import defaultdict
    # Map audio_path → {diff_level: (features_audio, labels)}
    audio_map = defaultdict(dict)
    for feat_audio, lbls, diff_level, audio_path in sequences:
        audio_map[audio_path][diff_level] = (feat_audio, lbls)

    result = []
    for feat_audio, lbls, diff_level, audio_path in sequences:
        T   = len(feat_audio)
        ctx = np.zeros((T, FEAT_DIM_CTX), dtype=np.float32)
        for d in range(DIFF_LEVELS):
            if d == diff_level:
                continue
            if d in audio_map[audio_path]:
                sibling_lbl = audio_map[audio_path][d][1]
                n = min(T, len(sibling_lbl))
                ctx[:n, d*4:(d+1)*4] = sibling_lbl[:n]
        feat_full = np.concatenate([feat_audio, ctx], axis=1)   # (T, 124)
        result.append((feat_full, lbls, diff_level))

    n_cross = sum(
        1 for ap, dmap in audio_map.items() if len(dmap) > 1
    )
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
        Explicitly models local patterns before attention:
          k=3  → note pairs (jacks, 2-note bursts)
          k=5  → trills and short streams
          k=9  → longer streams (8-note runs)
        """
        def __init__(self, d, drop=0.1):
            super().__init__()
            self.convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(d, d, k, padding=k // 2, groups=d),  # depthwise
                    nn.Conv1d(d, d, 1),                              # pointwise
                    nn.GELU(),
                )
                for k in [3, 5, 9]
            ])
            self.proj = nn.Linear(3 * d, d)
            self.norm = nn.LayerNorm(d)
            self.drop = nn.Dropout(drop)

        def forward(self, x):
            h = x.transpose(1, 2)                          # (B, D, T)
            outs = [conv(h).transpose(1, 2) for conv in self.convs]  # 3×(B,T,D)
            h = torch.cat(outs, dim=-1)                    # (B, T, 3D)
            return self.norm(x + self.drop(self.proj(h)))  # residual

    class ManiaTransformerV3(nn.Module):
        """
        v3: PatternConvBlock + Transformer + 4 per-difficulty heads.

        The PatternConvBlock sees local context explicitly (streams, jacks,
        trills) before the global attention layers.  Four separate output heads
        let each difficulty learn its own note-density and pattern signature
        while sharing the full audio encoder.

        Input : x     (B, T, feat_dim)
                diff  (B,)  — difficulty index 0-3
        Output: list of 4 tensors (B, T, 4) — one per difficulty head
        """
        def __init__(self):
            super().__init__()
            self.diff_emb   = nn.Embedding(diff_levels, diff_emb)
            self.input_proj = nn.Linear(feat_dim + diff_emb, d_model)
            self.conv_block = PatternConvBlock(d_model, drop=dropout)
            self.pos_enc    = PositionalEncoding(d_model, drop=dropout)
            enc_layer       = nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward=dim_ff,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
            self.norm        = nn.LayerNorm(d_model)
            # 4 difficulty heads — deeper than v2 for better specialisation
            self.heads = nn.ModuleList([
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

        feats, lbls, diff_level, ap = extract_features_and_labels(
            audio_path, groups, version_str)
        if feats is None:
            continue

        raw_sequences.append((feats, lbls, diff_level, ap))
        diff_counts[diff_level] += 1

    print(f"\nUsable maps : {len(raw_sequences)} / {maps_tried}")
    print(f"  Easy:{diff_counts[0]}  Normal:{diff_counts[1]}  "
          f"Hard:{diff_counts[2]}  Insane:{diff_counts[3]}")

    if len(raw_sequences) < 5:
        print("ERROR: Not enough usable maps."); sys.exit(1)

    # Build cross-difficulty context features
    print("\nBuilding cross-difficulty context...")
    sequences = build_cross_diff_context(raw_sequences)

    dataset = make_dataset(sequences)
    loader  = DataLoader(dataset, batch_size=16, shuffle=True,
                         num_workers=0, drop_last=True)
    print(f"Dataset     : {len(dataset)} sequences × {SEQ_LEN} steps\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model()
    model.to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model       : ManiaTransformerV3  |  device: {device}  |  params: {params:,}")
    print(f"Features    : mel({N_MEL}) + onset + SC({N_SC}) + chroma({N_CHROMA}) "
          f"+ rms + flat + phases(6) + ctx({FEAT_DIM_CTX}) = {FEAT_DIM}")
    print(f"Grid        : SUBDIV={SUBDIV}  |  4 difficulty heads\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    # Warmup for first 3 epochs, then cosine decay
    def lr_lambda(ep):
        if ep < 3:
            return (ep + 1) / 3
        return 0.5 * (1 + np.cos(np.pi * (ep - 3) / max(epochs - 3, 1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    SIBLING_WEIGHT = 0.25   # auxiliary loss weight for sibling difficulty heads

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for X, y_note, diff in loader:
            X, y_note, diff = X.to(device), y_note.to(device), diff.to(device)

            note_mean  = y_note.mean().item()
            pw_val     = min((1.0 - note_mean) / (note_mean + 1e-6), 15.0)
            pw_val     = max(pw_val, 2.0)
            pos_weight = torch.tensor([pw_val] * 4, device=device)

            optimizer.zero_grad()
            all_logits = model(X, diff)   # list of 4 tensors (B, T, 4)

            # Primary loss: each sample's target difficulty head
            loss = torch.tensor(0.0, device=device)
            for d in range(DIFF_LEVELS):
                mask = (diff == d)
                if mask.sum() == 0:
                    continue
                loss = loss + F.binary_cross_entropy_with_logits(
                    all_logits[d][mask], y_note[mask],
                    pos_weight=pos_weight
                )

            # Auxiliary multi-task loss: sibling heads see target labels too
            # (cross-diff context in input features already provides signal;
            #  this pushes all heads to agree on active positions)
            if SIBLING_WEIGHT > 0:
                for d in range(DIFF_LEVELS):
                    for d2 in range(DIFF_LEVELS):
                        if d2 == d:
                            continue
                        mask = (diff == d)
                        if mask.sum() == 0:
                            continue
                        loss = loss + SIBLING_WEIGHT * F.binary_cross_entropy_with_logits(
                            all_logits[d2][mask], y_note[mask],
                            pos_weight=pos_weight
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
        "model_state":    model.state_dict(),
        "model_type":     "ManiaTransformerV3",
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
        "maps_trained":   len(sequences),
    }, out_path)

    print(f"\nModel saved → {out_path}")
    print(f"  Maps : {len(sequences)}   Sequences : {len(dataset)}")
    print(f'\nNext: python ManiaMapper.py --ui')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train osu!mania 4K Transformer v3")
    parser.add_argument("--maps-dir", default=DEFAULT_MAPS_DIR)
    parser.add_argument("--out",      default=DEFAULT_OUT)
    parser.add_argument("--max-maps", type=int, default=3000)
    parser.add_argument("--epochs",   type=int, default=40)
    args = parser.parse_args()

    if not os.path.isdir(args.maps_dir):
        print(f"ERROR: Directory not found: {args.maps_dir}"); sys.exit(1)

    print(f"Maps dir : {args.maps_dir}")
    print(f"Output   : {args.out}")
    print(f"Max maps : {args.max_maps}  |  Epochs : {args.epochs}")
    train(args.maps_dir, args.out, args.max_maps, args.epochs)


if __name__ == "__main__":
    main()
