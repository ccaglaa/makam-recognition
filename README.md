# Makam Recognition

Recognizing the makam (melodic mode) of Ottoman-Turkish music recordings from their melody.
Work in progress, milestone M2 (baseline classifier).

## Setup (Mac M1)

```bash
# 1. get the dataset next to this repo (approx. 440 MB)
git clone https://github.com/MTG/otmm_makam_recognition_dataset.git ../otmm_makam_recognition_dataset

# 2. virtual environment + this package in editable mode
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 3. check everything works
pytest -q
python scripts/m1_explore.py --data ../otmm_makam_recognition_dataset
python scripts/m2_baseline.py --data ../otmm_makam_recognition_dataset
```

The first run parses 1,000 text files and saves binary copies in `cache/`; later runs are faster.

## Layout

```
src/makam/data.py       load annotations, metadata and pitch tracks
src/makam/features.py   Hz -> cents, octave folding, pitch-class histograms
src/makam/classify.py   distances, kNN and nearest-template classifiers
src/makam/evaluate.py   composition groups, stratified and grouped cross-validation
scripts/m1_explore.py   dataset stats + figures in figures/
scripts/m2_baseline.py  accuracy table (results/m2_accuracy.csv) + confusion matrix
tests/                  unit tests
```

## Data

OTMM Makam Recognition Dataset (Karakurt, Şentürk & Serra, 2016), CC BY-NC-SA 4.0:
20 makams x 50 recordings, distributed as predominant-melody pitch tracks (no audio).

## Results so far (tonic known, 10-fold CV, seed 0)

| Method | Stratified | Grouped by composition |
|---|---|---|
| Nearest template, Bhattacharyya | 72.5% +- 3.1 | 73.0% +- 5.6 |
| kNN k=10, Bhattacharyya | 70.5% +- 4.6 | 68.3% +- 4.0 |
| kNN k=1, Bhattacharyya | 65.1% +- 4.2 | 62.0% +- 4.6 |
| Published MORTY baseline (Karakurt et al. 2016) | 71.8% | not reported |

Across seeds 0 to 2 the best method ranges from 71.9% to 73.3% (grouped), so
differences under ar. 2 points are noise. The best configuration was picked
among 18, which makes its number slightly optimistic.
