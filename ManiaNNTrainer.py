"""
ManiaNNTrainer.py  —  Style-aware osu!mania 4K trainer
=======================================================
Trains a style-conditioned LSTM on (audio, notes, style) triples.

Produces mania_style_model.pt which supports:
  • Style recommendation  — given a new song, score all 10 styles
  • Style-conditioned generation — given audio + chosen style → note placement

Data layout
-----------
Organise your training maps under a styles directory like:
    C:\\ManiaStyles\\
        tech\\
            song_folder_1\\
                map.osu
                audio.mp3
        streams\\
            ...
        complex_streams\\   jump_streams\\   hand_streams\\
        ln\\                complex_ln\\      ranked_allround\\
        chord_jacks\\       chord_jacks_doubles\\

Subfolder names must match STYLE_CLASSES (case-insensitive).
Each subfolder is scanned recursively for .osu + audio pairs.

Usage
-----
    python ManiaNNTrainer.py --styles-dir C:\\ManiaStyles
    python ManiaNNTrainer.py --styles-dir C:\\ManiaStyles --epochs 20 --maps-per-style 300

Requirements: pip install torch librosa numpy tqdm
"""

import os, sys, json, argparse, warnings
from collections import defaultdict
import numpy as np

warnings.filterwarnings("ignore")

# ── Style definitions ──────────────────────────────────────────────────────────
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
NUM_STYLES  = len(STYLE_CLASSES)
STYLE_TO_ID = {s: i for i, s in enumerate(STYLE_CLASSES)}

DEFAULT_STYLES_DIR = r"C:\ManiaStyles"
DEFAULT_OUT        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mania_style_model.pt")

# ── Feature config (must match ManiaMapper inference) ─────────────────────────
N_MFCC    = 20
SUBDIV    = 4        # 16th-note grid
HOP       = 512
FEAT_DIM  = N_MFCC + 5    # mfcc + bass + mid + high + sin(phase) + cos(phase)
STYLE_DIM = 16             # style embedding dimension
SEQ_LEN   = 128            # LSTM chunk length


# ── osu! parser ───────────────────────────────────────────────────────────────

def x_to_col(x: int) -> int:
    return min(3, max(0, x * 4 // 512))


def parse_osu(path: str):
    """
    Parse a .osu file.
    Returns (audio_filename, groups, has_ln) or None.
    groups = list of (time_ms, [cols])
    has_ln = True if the map contains long notes
    """
    mode = audio_file = None
    keys = None
    in_hitobjects = False
    raw_notes = []
    has_ln = False

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line.startswith("AudioFilename:"):
                    audio_file = line.split(":", 1)[1].strip()
                elif line.startswith("Mode:"):
                    try:    mode = int(line.split(":")[1].strip())
                    except ValueError: pass
                elif line.startswith("CircleSize:"):
                    try:    keys = float(line.split(":")[1].strip())
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
                        x    = int(parts[0])
                        time = int(parts[2])
                        raw_notes.append((time, x_to_col(x)))
                        # LN: type bit 7 set and last field has endtime:0
                        if len(parts) >= 6 and int(parts[3]) & 128:
                            has_ln = True
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

    return audio_file, groups, has_ln


# ── Audio feature extraction ──────────────────────────────────────────────────

def extract_features_and_labels(audio_path: str, groups: list):
    """
    Returns (features, labels) or (None, None).
    features : np.ndarray (T, FEAT_DIM)
    labels   : np.ndarray (T, 4)   binary per column
    """
    try:
        import librosa
    except ImportError:
        print("ERROR: pip install librosa")
        sys.exit(1)

    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
    except Exception:
        return None, None

    duration_ms = len(y) / sr * 1000

    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr) * 1000
        bpm = float(np.atleast_1d(tempo)[0])
    except Exception:
        return None, None

    beat_length = 60000.0 / bpm
    step_ms     = beat_length / SUBDIV

    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP)
        mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-9)
    except Exception:
        return None, None

    try:
        S     = np.abs(librosa.stft(y, hop_length=HOP))
        freqs = librosa.fft_frequencies(sr=sr)
        def _norm(x):
            mn, mx = x.min(), x.max()
            return (x - mn) / (mx - mn + 1e-9)
        bass_e = _norm(S[freqs < 300, :].mean(axis=0))
        mid_e  = _norm(S[(freqs >= 300) & (freqs < 3000), :].mean(axis=0))
        high_e = _norm(S[freqs >= 3000, :].mean(axis=0))
    except Exception:
        return None, None

    frame_times = librosa.frames_to_time(np.arange(mfcc.shape[1]), sr=sr, hop_length=HOP) * 1000
    n_frames    = mfcc.shape[1]

    def feat_at(t_ms):
        idx = min(int(np.searchsorted(frame_times, t_ms)), n_frames - 1)
        f   = np.zeros(FEAT_DIM, dtype=np.float32)
        f[:N_MFCC]    = mfcc[:, idx]
        f[N_MFCC]     = bass_e[idx]
        f[N_MFCC + 1] = mid_e[idx]
        f[N_MFCC + 2] = high_e[idx]
        phase          = (t_ms % (beat_length * 4)) / (beat_length * 4 + 1e-9)
        f[N_MFCC + 3] = float(np.sin(2 * np.pi * phase))
        f[N_MFCC + 4] = float(np.cos(2 * np.pi * phase))
        return f

    positions = []
    for t_beat in beat_times:
        for s in range(SUBDIV):
            t = t_beat + s * step_ms
            if t > duration_ms:
                break
            positions.append(t)

    if len(positions) < 64:
        return None, None

    features = np.stack([feat_at(t) for t in positions])

    labels   = np.zeros((len(positions), 4), dtype=np.float32)
    TOL_MS   = max(step_ms * 0.5, 25.0)
    pos_arr  = np.array(positions)

    for t_note, cols in groups:
        dists   = np.abs(pos_arr - t_note)
        closest = int(np.argmin(dists))
        if dists[closest] <= TOL_MS:
            for c in cols:
                labels[closest, c] = 1.0

    ratio = labels.sum() / (labels.size + 1e-9)
    if ratio < 0.03 or ratio > 0.75:
        return None, None

    return features, labels


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model(hidden_dim=256, num_layers=3, dropout=0.3):
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("ERROR: pip install torch")
        sys.exit(1)

    class ManiaStyleLSTM(nn.Module):
        """
        Style-conditioned LSTM for osu!mania note placement.

        Two heads:
          note_head  — per-timestep, predicts which columns have notes
          style_head — global (mean-pooled), classifies the mapping style

        At training:  style label is known → embed it and concatenate with audio features
        At inference (style recommend): pass style_id=-1 → zero embedding → read style_head
        At inference (mapping):         pass chosen style_id → note_head drives generation
        """
        def __init__(self):
            super().__init__()
            self.style_embed = nn.Embedding(NUM_STYLES, STYLE_DIM)
            self.lstm        = nn.LSTM(FEAT_DIM + STYLE_DIM, hidden_dim, num_layers,
                                       batch_first=True, dropout=dropout)
            self.norm        = nn.LayerNorm(hidden_dim)
            self.note_head   = nn.Linear(hidden_dim, 4)
            self.style_head  = nn.Linear(hidden_dim, NUM_STYLES)

        def forward(self, x, style_ids):
            """
            x          : (B, T, FEAT_DIM)
            style_ids  : (B,) LongTensor  — style class index per sample
            Returns:
              note_logits  : (B, T, 4)
              style_logits : (B, NUM_STYLES)   from mean-pooled hidden state
            """
            import torch
            emb   = self.style_embed(style_ids)            # (B, STYLE_DIM)
            emb_t = emb.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, T, STYLE_DIM)
            inp   = torch.cat([x, emb_t], dim=-1)          # (B, T, FEAT_DIM+STYLE_DIM)
            out, _ = self.lstm(inp)                         # (B, T, hidden)
            out   = self.norm(out)
            note_logits  = self.note_head(out)              # (B, T, 4)
            style_logits = self.style_head(out.mean(dim=1)) # (B, NUM_STYLES)
            return note_logits, style_logits

        def recommend_style(self, x):
            """
            x : (1, T, FEAT_DIM)  — audio features for a new song
            Returns probability vector of shape (NUM_STYLES,)
            """
            import torch
            dummy_style = torch.zeros(1, dtype=torch.long, device=x.device)
            with torch.no_grad():
                _, style_logits = self.forward(x, dummy_style)
                return torch.softmax(style_logits[0], dim=0).cpu().numpy()

        def generate_notes(self, x, style_id: int, threshold: float = 0.5):
            """
            x        : (1, T, FEAT_DIM)
            style_id : int
            Returns binary note matrix (T, 4)
            """
            import torch
            sid = torch.tensor([style_id], dtype=torch.long, device=x.device)
            with torch.no_grad():
                note_logits, _ = self.forward(x, sid)
                return (torch.sigmoid(note_logits[0]) > threshold).cpu().numpy()

    return ManiaStyleLSTM()


# ── Dataset ───────────────────────────────────────────────────────────────────

def make_dataset(sequences):
    """sequences: list of (features, labels, style_id)"""
    try:
        import torch
        from torch.utils.data import Dataset
    except ImportError:
        print("ERROR: pip install torch")
        sys.exit(1)

    class ManiaStyleDS(Dataset):
        def __init__(self, seqs, seq_len=SEQ_LEN):
            self.samples = []
            stride = seq_len // 2
            for feats, lbls, style_id in seqs:
                T = len(feats)
                for start in range(0, T - seq_len, stride):
                    end = start + seq_len
                    self.samples.append((
                        torch.tensor(feats[start:end], dtype=torch.float32),
                        torch.tensor(lbls[start:end],  dtype=torch.float32),
                        torch.tensor(style_id,          dtype=torch.long),
                    ))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            return self.samples[i]

    return ManiaStyleDS(sequences)


# ── Style directory scanner ───────────────────────────────────────────────────

def scan_style_dirs(styles_dir: str, max_per_style: int):
    """
    Scan styles_dir/  for subfolders matching STYLE_CLASSES.
    Returns list of (osu_path, audio_path, style_id).
    """
    found = []
    styles_dir = os.path.abspath(styles_dir)

    for entry in os.scandir(styles_dir):
        if not entry.is_dir():
            continue
        style_name = entry.name.lower()
        if style_name not in STYLE_TO_ID:
            print(f"  [WARN] Unknown style folder '{entry.name}' — skipping. "
                  f"Valid: {STYLE_CLASSES}")
            continue

        style_id   = STYLE_TO_ID[style_name]
        style_maps = []

        for root, _, files in os.walk(entry.path):
            for f in files:
                if f.endswith(".osu"):
                    style_maps.append(os.path.join(root, f))

        style_maps = style_maps[:max_per_style]
        print(f"  {style_name:25s}  {len(style_maps):4d} .osu files")
        for osu_path in style_maps:
            found.append((osu_path, style_id))

    return found


# ── Training ──────────────────────────────────────────────────────────────────

def train(styles_dir: str, out_path: str, max_per_style: int, epochs: int,
          style_loss_weight: float = 0.3):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError:
        print("ERROR: pip install torch")
        sys.exit(1)

    try:
        from tqdm import tqdm
        _tqdm = tqdm
    except ImportError:
        _tqdm = lambda x, **kw: x

    # ── Scan style dirs ────────────────────────────────────────────────────
    print(f"\nScanning styles directory: {styles_dir}")
    candidates = scan_style_dirs(styles_dir, max_per_style)
    if not candidates:
        print("ERROR: No maps found. Check --styles-dir and subfolder names.")
        sys.exit(1)
    print(f"\nTotal candidates: {len(candidates)}\n")

    # ── Extract features ───────────────────────────────────────────────────
    sequences  = []
    style_counts = defaultdict(int)
    maps_tried = 0

    for osu_path, style_id in _tqdm(candidates, desc="Extracting features"):
        maps_tried += 1
        result = parse_osu(osu_path)
        if result is None:
            continue

        audio_filename, groups, _ = result
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

        feats, lbls = extract_features_and_labels(audio_path, groups)
        if feats is None:
            continue

        sequences.append((feats, lbls, style_id))
        style_counts[STYLE_CLASSES[style_id]] += 1

    print(f"\nExtracted {len(sequences)} maps  (tried {maps_tried})")
    print("Per-style breakdown:")
    for s in STYLE_CLASSES:
        print(f"  {s:25s}: {style_counts[s]}")

    if len(sequences) < 20:
        print("ERROR: Not enough usable maps. Need audio files alongside .osu.")
        sys.exit(1)

    # ── Dataset & loader ───────────────────────────────────────────────────
    dataset = make_dataset(sequences)
    loader  = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
    print(f"\nDataset: {len(dataset)} sequences of length {SEQ_LEN}")

    # ── Model ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model()
    model.to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: ManiaStyleLSTM  |  device: {device}  |  params: {params:,}\n")

    pos_weight     = torch.tensor([5.0] * 4, device=device)
    note_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    style_criterion= nn.CrossEntropyLoss()
    optimizer      = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler      = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Train ──────────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss  = 0.0
        note_loss_t = 0.0
        style_loss_t= 0.0
        style_correct = 0
        style_total   = 0

        for X, y_note, y_style in loader:
            X, y_note, y_style = X.to(device), y_note.to(device), y_style.to(device)
            optimizer.zero_grad()

            note_logits, style_logits = model(X, y_style)

            loss_note  = note_criterion(note_logits, y_note)
            loss_style = style_criterion(style_logits, y_style)
            loss       = loss_note + style_loss_weight * loss_style

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss   += loss.item()
            note_loss_t  += loss_note.item()
            style_loss_t += loss_style.item()

            preds = style_logits.argmax(dim=1)
            style_correct += (preds == y_style).sum().item()
            style_total   += y_style.size(0)

        scheduler.step()
        n = len(loader)
        print(f"  Epoch {epoch:2d}/{epochs}  "
              f"total: {total_loss/n:.4f}  "
              f"note: {note_loss_t/n:.4f}  "
              f"style: {style_loss_t/n:.4f}  "
              f"style_acc: {style_correct/style_total:.3f}")

    # ── Save ───────────────────────────────────────────────────────────────
    torch.save({
        "model_state":       model.state_dict(),
        "style_classes":     STYLE_CLASSES,
        "hidden_dim":        256,
        "num_layers":        3,
        "feat_dim":          FEAT_DIM,
        "style_dim":         STYLE_DIM,
        "n_mfcc":            N_MFCC,
        "subdiv":            SUBDIV,
        "maps_trained":      len(sequences),
        "style_counts":      dict(style_counts),
    }, out_path)

    print(f"\nModel saved: {out_path}")
    print(f"  Styles     : {NUM_STYLES}")
    print(f"  Maps used  : {len(sequences)}")
    print(f"  Sequences  : {len(dataset)}")
    print(f"\nNext steps:")
    print(f"  1. Run ManiaMapper.py --style recommend  <song.mp3>   → get style suggestion")
    print(f"  2. Run ManiaMapper.py --style <name>     <song.mp3>   → generate .osz")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train style-conditioned osu!mania 4K model")
    parser.add_argument("--styles-dir",     default=DEFAULT_STYLES_DIR,
                        help=f"Root folder with style subfolders. Default: {DEFAULT_STYLES_DIR}")
    parser.add_argument("--out",            default=DEFAULT_OUT,
                        help="Output .pt path")
    parser.add_argument("--maps-per-style", type=int, default=300,
                        help="Max maps to use per style (default: 300)")
    parser.add_argument("--epochs",         type=int, default=15,
                        help="Training epochs (default: 15)")
    parser.add_argument("--style-loss-weight", type=float, default=0.3,
                        help="Weight for style classification loss (default: 0.3)")
    args = parser.parse_args()

    if not os.path.isdir(args.styles_dir):
        print(f"ERROR: Styles directory not found: {args.styles_dir}")
        print(f"\nCreate it with subfolders named after each style:")
        for s in STYLE_CLASSES:
            print(f"  {args.styles_dir}\\{s}\\")
        sys.exit(1)

    print(f"Styles dir       : {args.styles_dir}")
    print(f"Output           : {args.out}")
    print(f"Maps per style   : {args.maps_per_style}  |  Epochs: {args.epochs}")
    print(f"Style loss weight: {args.style_loss_weight}")
    print(f"\nStyle classes ({NUM_STYLES}):")
    for i, s in enumerate(STYLE_CLASSES):
        print(f"  {i:2d}  {s}")

    train(args.styles_dir, args.out, args.maps_per_style, args.epochs,
          args.style_loss_weight)


if __name__ == "__main__":
    main()
