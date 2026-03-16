"""
ManiaNNTrainer.py  —  osu!mania 4K trainer
==========================================
Trains an LSTM on the relationship between audio features and note placement.

No style labels. No categories. The model sees audio and notes from many real
maps and learns on its own: what does the music sound like when streams appear?
When long notes appear? When chords cluster on the downbeat? It absorbs all of
this from the training data directly.

The folder structure under --maps-dir is ignored during training — subfolders
are just for your own organisation. Every 4K .osu file with a matching audio
file is used regardless of which folder it lives in.

Data layout
-----------
    C:\\ManiaStyles\\
        anything\\
            song_folder\\
                map.osu
                audio.mp3   (or .ogg / .wav / .flac)
        more_folders\\
            ...

Usage
-----
    python ManiaNNTrainer.py --maps-dir C:\\ManiaStyles
    python ManiaNNTrainer.py --maps-dir C:\\ManiaStyles --epochs 20 --max-maps 1000

Requirements: pip install torch librosa numpy tqdm
"""

import os, sys, argparse, warnings
import numpy as np

warnings.filterwarnings("ignore")

# ── Feature config (must match ManiaMapper.py exactly) ────────────────────────
N_MFCC   = 20
SUBDIV   = 4        # 16th-note grid
HOP      = 512
FEAT_DIM = N_MFCC + 5   # mfcc(20) + bass + mid + high + sin(phase) + cos(phase)
SEQ_LEN  = 128

DEFAULT_MAPS_DIR = r"C:\Users\Aravind Dora\Desktop\ManiaMapper\ManiaStyles"
DEFAULT_OUT      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mania_model.pt")


# ── osu! parser ───────────────────────────────────────────────────────────────

def x_to_col(x):
    return min(3, max(0, x * 4 // 512))


def parse_osu(path):
    """
    Parse a 4K .osu file.
    Returns (audio_filename, groups) or None if not usable.
    groups = list of (time_ms, [cols])
    """
    mode = audio_file = None
    keys = None
    in_hitobjects = False
    raw_notes = []

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line.startswith("AudioFilename:"):
                    audio_file = line.split(":", 1)[1].strip()
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

    return audio_file, groups


# ── Audio feature extraction ──────────────────────────────────────────────────

def extract_features_and_labels(audio_path, groups):
    """
    Returns (features, labels) arrays or (None, None).
    features : (T, FEAT_DIM)  — audio features at each 16th-note position
    labels   : (T, 4)         — binary, which columns have a note
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
        bpm        = float(np.atleast_1d(tempo)[0])
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


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(feat_dim=FEAT_DIM, hidden_dim=256, num_layers=3, dropout=0.3):
    try:
        import torch.nn as nn
    except ImportError:
        print("ERROR: pip install torch")
        sys.exit(1)

    class ManiaMapperLSTM(nn.Module):
        """
        LSTM that maps audio features → note placement probabilities.

        Input : (batch, time, FEAT_DIM)  — audio features at each 16th-note step
        Output: (batch, time, 4)         — logits for each of the 4 columns

        No style labels. The model learns the audio-to-pattern relationship
        directly from the training data.
        """
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(feat_dim, hidden_dim, num_layers,
                                batch_first=True, dropout=dropout)
            self.norm = nn.LayerNorm(hidden_dim)
            self.fc   = nn.Linear(hidden_dim, 4)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(self.norm(out))

    return ManiaMapperLSTM()


# ── Dataset ───────────────────────────────────────────────────────────────────

def make_dataset(sequences):
    try:
        import torch
        from torch.utils.data import Dataset
    except ImportError:
        print("ERROR: pip install torch")
        sys.exit(1)

    class ManiaDS(Dataset):
        def __init__(self, seqs, seq_len=SEQ_LEN):
            self.samples = []
            stride = seq_len // 2
            for feats, lbls in seqs:
                T = len(feats)
                for start in range(0, T - seq_len, stride):
                    end = start + seq_len
                    self.samples.append((
                        torch.tensor(feats[start:end], dtype=torch.float32),
                        torch.tensor(lbls[start:end],  dtype=torch.float32),
                    ))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            return self.samples[i]

    return ManiaDS(sequences)


# ── Scanner ───────────────────────────────────────────────────────────────────

def scan_all_maps(maps_dir, max_maps):
    """
    Recursively find all .osu files under maps_dir.
    Folder names are completely ignored — everything is treated equally.
    """
    found = []
    for root, _, files in os.walk(maps_dir):
        for f in files:
            if f.endswith(".osu"):
                found.append(os.path.join(root, f))
    if max_maps:
        found = found[:max_maps]
    print(f"  Found {len(found)} .osu files (scanning all subfolders)")
    return found


# ── Training ──────────────────────────────────────────────────────────────────

def train(maps_dir, out_path, max_maps, epochs):
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

    print(f"\nScanning: {maps_dir}")
    candidates = scan_all_maps(maps_dir, max_maps)
    if not candidates:
        print("ERROR: No .osu files found.")
        sys.exit(1)
    print(f"  Total candidates: {len(candidates)}\n")

    sequences  = []
    maps_tried = 0

    for osu_path in _tqdm(candidates, desc="Extracting features"):
        maps_tried += 1
        result = parse_osu(osu_path)
        if result is None:
            continue

        audio_filename, groups = result
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

        sequences.append((feats, lbls))

    print(f"\nUsable maps: {len(sequences)} / {maps_tried}")

    if len(sequences) < 10:
        print("ERROR: Not enough usable maps. Make sure audio files are next to the .osu files.")
        sys.exit(1)

    dataset = make_dataset(sequences)
    loader  = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
    print(f"Dataset: {len(dataset)} sequences of length {SEQ_LEN}\n")

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model      = build_model()
    model.to(device)
    params     = sum(p.numel() for p in model.parameters())
    print(f"Model: ManiaMapperLSTM  |  device: {device}  |  params: {params:,}\n")

    pos_weight = torch.tensor([5.0] * 4, device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for X, y_note in loader:
            X, y_note = X.to(device), y_note.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y_note)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        print(f"  Epoch {epoch:2d}/{epochs}  loss: {total_loss / len(loader):.4f}")

    torch.save({
        "model_state":  model.state_dict(),
        "feat_dim":     FEAT_DIM,
        "hidden_dim":   256,
        "num_layers":   3,
        "maps_trained": len(sequences),
    }, out_path)

    print(f"\nModel saved → {out_path}")
    print(f"  Maps used : {len(sequences)}")
    print(f"  Sequences : {len(dataset)}")
    print(f"\nNext step:")
    print(f'  python ManiaMapper.py "song.mp3"')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train osu!mania 4K LSTM")
    parser.add_argument("--maps-dir", default=DEFAULT_MAPS_DIR,
                        help=f"Root folder with maps. Default: {DEFAULT_MAPS_DIR}")
    parser.add_argument("--out",      default=DEFAULT_OUT,
                        help="Output .pt path")
    parser.add_argument("--max-maps", type=int, default=3000,
                        help="Max maps to use (default: 3000)")
    parser.add_argument("--epochs",   type=int, default=15,
                        help="Training epochs (default: 15)")
    args = parser.parse_args()

    if not os.path.isdir(args.maps_dir):
        print(f"ERROR: Directory not found: {args.maps_dir}")
        sys.exit(1)

    print(f"Maps dir  : {args.maps_dir}")
    print(f"Output    : {args.out}")
    print(f"Max maps  : {args.max_maps}  |  Epochs: {args.epochs}")

    train(args.maps_dir, args.out, args.max_maps, args.epochs)


if __name__ == "__main__":
    main()
