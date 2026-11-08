# ManiaMapper

An AI-powered osu!mania 4K beatmap generator — feed it an audio file, get a playable `.osz` beatmap.

---

## Demo

*Screenshots coming soon:*
- *CLI generating a map end-to-end*
- *A generated map open in the osu! editor*
- *The web app UI*

---

## Features

- **AI note generation** via an autoregressive encoder/decoder network (~6.97M params) trained on real maps — one shared model across all training data, not conditioned on a selectable style (see [Limitations](#limitations))
- **Audio analysis** with Librosa: BPM detection, beat tracking, onset detection (full + harmonic + percussive), mel spectrogram, chroma, spectral contrast/flatness/centroid, zero-crossing rate
- **4 difficulty presets**: Easy, Normal, Hard, Insane, each with its own output head and target note rate
- **Post-generation refinement**: min-gap enforcement, column rebalancing, pattern diversification
- **Long notes** (LNs): model-predicted placement probability, thresholded per difficulty to hit a target LN percentage
- **Optional SV** (speed variation) that reacts to song energy (0.70–1.30×)
- **Tkinter GUI** (`--ui`) alongside the CLI, plus a [web app](#usage-cli--web-app)
- Outputs standard `.osz` files — double-click to import into osu!

---

## How it works

```
Audio File
    ↓
Librosa Analysis   →  BPM, beats, onsets, mel/chroma/spectral features
    ↓
Beat Grid          →  Timing subdivision aligned to detected beats
    ↓
Encoder + Decoder  ←  mania_model.pt (difficulty-conditioned, autoregressive)
    ↓
Note Assignment    →  Column selection, min-gap enforcement, column rebalancing
    ↓
SV Generation      →  (optional) Energy-reactive speed points
    ↓
.osu Writer        →  Standard osu!mania format
    ↓
.osz Package       →  Ready to import
```

### Neural network architecture

```
Input:  (B, T, 130)  — 114 audio features + 16 cross-difficulty context
Difficulty Embedding: 4 levels, 32-dim

Encoder (bidirectional, full-sequence context):
  PatternConvBlock ×2   — depthwise multi-scale conv (kernels 3/5/9/13/25)
  BiLSTM: 4 layers, 384-dim (192 hidden × 2 directions)

Decoder (causal, autoregressive):
  Event Embedding: 64-dim (previously-placed event; teacher-forced in training)
  LSTM: 2 layers, 256 hidden

Output heads (one set per difficulty level, ×4):
  ├── Hit Head:     (B, T, 4)   — per-column note placement
  ├── LN Head:      (B, T, 4)   — per-column LN start
  └── LN-Dur Head:  (B, T, 4)   — LN duration (sigmoid × 16 beats)
Pattern Head: (B, T, 18)        — auxiliary pattern-type classification
```

The `ManiaStyles/` training folders (`tech`, `streams`, `ln`, etc.) organize training data for variety; they are not a conditioning input to the model — see [Limitations](#limitations).

---

## Setup

```
pip install -r requirements.txt
```

Training additionally needs `tqdm` (imported by `ManiaNNTrainer.py`, not currently listed in `requirements.txt`):

```
pip install tqdm
```

Python 3.6+. CUDA is supported but optional (falls back to CPU).

Audio formats: MP3, OGG, WAV, FLAC (the web app also accepts M4A).

---

## Usage (CLI + web app)

### Generate a beatmap (CLI)

```bash
python ManiaMapper.py song.mp3
python ManiaMapper.py song.mp3 --output my_map.osz
python ManiaMapper.py song.mp3 --nn mania_model.pt
python ManiaMapper.py --ui
```

You'll be prompted for song title, artist, difficulty level, and whether to add speed variation (unless `--title`/`--artist`/`--difficulty`/`--no-sv` are passed to skip the prompts). `--ui` opens the Tkinter GUI instead.

**A trained model is required** — `ManiaMapper.py` looks for `mania_model.pt` next to the script (or at the path passed via `--nn`) and exits with an error if it isn't found. No trained weights are included in this repo; train your own (below).

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

These style folders are just how training data gets organized on disk for variety — they are not fed to the model as a style label (see [Neural network architecture](#neural-network-architecture)).

```bash
python ManiaNNTrainer.py --maps-dir C:\ManiaStyles
python ManiaNNTrainer.py --maps-dir C:\ManiaStyles --epochs 20 --max-maps 3000
```

This produces `mania_model.pt` (the `--out` flag changes the path), which you pass to `ManiaMapper.py` via `--nn`.

### Web app

`app.py` is a FastAPI backend (`uvicorn app:app`) that wraps the CLI generator behind an HTTP API, with `static/index.html` as its frontend. It previously ran on a GCP VM at maniamapper.com; that deployment has since been retired, but the code here is what ran it.

- `POST /api/generate` — accepts an uploaded audio file (`.mp3`/`.ogg`/`.wav`/`.flac`/`.m4a`) plus `difficulty`/`title`/`artist` form fields, validates them, and shells out to `ManiaMapper.py` as a subprocess (`--nn mania_model.pt --no-sv`, 120s timeout) to generate the map. Returns the resulting `.osz` as a file download.
- Best-effort album art: before generating, it queries the iTunes Search API for `artist + title`, downloads the top result's artwork, and passes it to `ManiaMapper.py` as the beatmap background — silently skipped if the lookup, download, or image validation fails.
- `GET /api/health` — reports whether `mania_model.pt` is present on the server.
- Each request gets its own job directory under `/tmp/maniamapper_jobs`, cleaned up in the background after the response is sent.
- CORS is wide open (`allow_origins=["*"]`) — fine for a public generate-and-download API with no auth or user state, not something to carry into a service that has either.

---

## Difficulty Presets

| Level  | HP | OD | Max chord | Target notes/sec | Target LN % |
|--------|----|----|-----------|-------------------|-------------|
| Easy   | 6  | 6  | 1         | 2.5               | 4%          |
| Normal | 7  | 7  | 2         | 4.5               | 7%          |
| Hard   | 8  | 8  | 2         | 9.5               | 10%         |
| Insane | 9  | 9  | 3         | 16.0              | 12%         |

---

## Results

Numbers below are from `training_log.txt`, the log of the training run that also produced `mania_model_training_report.png` (both committed in this repo). No model weights are included — see [Train a custom model](#train-a-custom-model) above.

**Dataset:** 127 usable maps out of 151 scanned `.osu` files (Easy: 14, Normal: 21, Hard: 52, Insane: 40), built into 2,408 training sequences of 256 steps each. 222,013 regular notes and 45,807 LN notes (LN ratio 0.171). 20 audio files had 2+ difficulties available, used for cross-difficulty context.

**Training:** 50/50 epochs, on GPU (cuda), ~6.97M parameters. Final epoch — total loss 0.3044 (hit-placement BCE 0.016, LN-placement BCE 0.167, pattern classification CE 0.076), pattern-classification accuracy 90.1%. The training report below additionally plots an LN-duration MSE curve not broken out in the text log.

That 90.1% is **training-set accuracy** — the trainer doesn't hold out a validation split, so it measures fit to the training data, not generalization to unseen maps.

![Training report: loss curves, pattern accuracy, confusion matrix, and dataset diagnostics](mania_model_training_report.png)

---

## Limitations

- **4K only.** `KEYS = 4` is hardcoded in `ManiaMapper.py`; there's no path to 5K/6K/7K without code changes.
- **No style conditioning at inference time.** The `ManiaStyles/` folders (`tech`, `streams`, `ln`, etc.) are only how training data happens to be organized on disk — the model has no style input or style head, so generated maps reflect whatever pattern mix the training set as a whole contains, not a style you can select. A style under-represented in training will be under-represented in output.
- **No objective playability metric.** Training is evaluated on loss and pattern-classification accuracy against the training set — there's no automated check for whether a generated map is actually fun or fair to play; that's a human judgment call per generated map.
- **No held-out validation split.** The 90.1% pattern accuracy in [Results](#results) is measured on the training data itself, not on unseen maps — it's evidence the model fit the training set, not evidence it generalizes.
- **One documented training run.** The numbers in [Results](#results) are from a single run (127 maps, 50 epochs) — no hyperparameter sweep or cross-validation was done.

---

## Files

| File | Description |
|------|-------------|
| `ManiaMapper.py` | Beatmap generator — CLI and Tkinter GUI (`--ui`) |
| `ManiaNNTrainer.py` | Trains the model from a folder of `.osu` maps |
| `app.py` | FastAPI backend serving the web app |
| `static/index.html` | Web app frontend, served by `app.py` |
| `requirements.txt` | Python dependencies |
| `mania_model_training_report.png` | Diagnostics from the last training run (see [Results](#results)) |
| `training_log.txt` | Raw console log from the last training run (see [Results](#results)) |
| `mania_model.pt` | Trained model weights — **not included**, gitignored; train your own |

---

## Tech stack

Python · PyTorch · Librosa · NumPy · FastAPI + Uvicorn (web backend) · Tkinter (desktop GUI) · Matplotlib + Pillow (training diagnostics)
