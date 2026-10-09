# Makam Recognition

Recognizing the makam (melodic mode) of Ottoman-Turkish music recordings from their melody.
Work in progress — milestone M1 (data exploration).

## Setup (Mac M1)

```bash
# 1. get the dataset next to this repo (~440 MB)
git clone https://github.com/MTG/otmm_makam_recognition_dataset.git ../otmm_makam_recognition_dataset

# 2. virtual environment + this package in editable mode
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 3. check everything works
pytest -q
python scripts/m1_explore.py --data ../otmm_makam_recognition_dataset
```

The first run parses 1,000 text files and saves binary copies in `cache/`; later runs are faster.

## Layout

```
src/makam/data.py       load annotations, metadata and pitch tracks
src/makam/features.py   Hz -> cents, octave folding, pitch-class histograms
scripts/m1_explore.py   dataset stats + figures in figures/
tests/                  unit tests for the feature code
```

## Data

OTMM Makam Recognition Dataset (Karakurt, Şentürk & Serra, 2016), CC BY-NC-SA 4.0:
20 makams × 50 recordings, distributed as predominant-melody pitch tracks (no audio).
