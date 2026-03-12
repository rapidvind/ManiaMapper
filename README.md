# ManiaMapper

An AI-powered osu!mania 4K beatmap generator. Feed it an audio file, get a playable `.osz` beatmap.

It uses a style-conditioned LSTM trained on real maps to place notes in a musically-aware way — responding to beats, onsets, energy, and spectral content across bass, mid, and high frequency bands.

---

## Features

- **AI note generation** via a 3-layer LSTM (256 hidden units) trained on 10 distinct mapping styles
- **Audio analysis** with Librosa: BPM detection, beat tracking, onset detection, 20-coefficient MFCC
- **4 difficulty presets**: Easy, Normal, Hard, Insane
- **Anti-pattern rules**: min-gap enforcement, same-hand ABA filtering, breathe cooldowns
- **Pattern library**: 11 trill types, chord jacks, staircases
- **Long notes** (LNs) with energy-dependent probability
- **Optional SV** (speed variation) that reacts to song energy (0.70–1.30×)
- **Rule-based fallback** with Markov chain column transitions if no model is loaded
- Outputs standard `.osz` files — double-click to import into osu!

---

## Requirements

```
pip install librosa numpy torch tqdm
```

Python 3.6+. CUDA is supported but optional (falls back to CPU).

Audio formats: MP3, OGG, WAV, FLAC.

---

## Usage

### Generate a beatmap

```bash
python ManiaMapper.py song.mp3
python ManiaMapper.py song.mp3 --output my_map.osz
python ManiaMapper.py song.mp3 --nn mania_nn_model.pt
```

You'll be prompted for song title, artist, difficulty level, and whether to add speed variation.

### Train a custom model

Organize your training maps like this:

```
ManiaStyles/
├── tech/
│   ├── song_1/
│   │   ├── map.osu
│   │   └── audio.mp3
│   └── song_2/
├── streams/
├── ln/
└── ...
```

Supported style folders: `tech`, `streams`, `complex_streams`, `jump_streams`, `hand_streams`, `ln`, `complex_ln`, `ranked_allround`, `chord_jacks`, `chord_jacks_doubles`

```bash
python ManiaNNTrainer.py --styles-dir C:\ManiaStyles
python ManiaNNTrainer.py --styles-dir C:\ManiaStyles --epochs 20 --maps-per-style 300
```

This produces `mania_nn_model.pt`, which you pass to `ManiaMapper.py` via `--nn`.

---

## How It Works

```
Audio File
    ↓
Librosa Analysis  →  BPM, beats, onsets, MFCC, spectral bands
    ↓
Beat Grid         →  Timing subdivision aligned to detected beats
    ↓
LSTM Inference    ←  mania_nn_model.pt
    ↓
Note Assignment   →  Column selection + anti-pattern filtering
    ↓
SV Generation     →  (optional) Energy-reactive speed points
    ↓
.osu Writer       →  Standard osu!mania format
    ↓
.osz Package      →  Ready to import
```

### Neural Network Architecture

```
Input:  (B, T, 25)   — 20 MFCC + 3 spectral + 2 phase features
Style Embedding: 16-dim
LSTM: 256 hidden, 3 layers, dropout=0.3
├── Note Head:  (B, T, 4)          — per-column predictions
└── Style Head: (B, NUM_STYLES)    — style classification
```

---

## Difficulty Presets

| Level  | HP | OD | Subdivision | Note Density |
|--------|----|----|-------------|--------------|
| Easy   | 6  | 6  | 2nd notes   | 40%          |
| Normal | 7  | 7  | 4th notes   | 55%          |
| Hard   | 8  | 8  | 8th notes   | 70%          |
| Insane | 9  | 9  | 8th notes   | 100%         |

---

## Files

| File | Description |
|------|-------------|
| `ManiaMapper.py` | Main beatmap generator |
| `ManiaNNTrainer.py` | LSTM trainer for custom models |
| `mania_nn_model.pt` | Trained model weights (not included — train your own) |
