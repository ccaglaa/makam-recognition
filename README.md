# Makam Recognition

Recognizing the makam (melodic mode) of Ottoman-Turkish music recordings from their melody.
Work in progress, milestone M3 (unknown tonic).

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
python scripts/m3_unknown_tonic.py --data ../otmm_makam_recognition_dataset
```

The first run parses 1,000 text files and saves binary copies in `cache/`; later runs are faster.

## Layout

```
src/makam/data.py       load annotations, metadata and pitch tracks
src/makam/features.py   Hz -> cents, octave folding, pitch-class histograms
src/makam/classify.py   distances, kNN and nearest-template classifiers
src/makam/evaluate.py   composition groups, stratified and grouped cross-validation
src/makam/tonic.py      tonic estimation by rotating histograms against templates
scripts/m1_explore.py   dataset stats + figures in figures/
scripts/m2_baseline.py  accuracy table (results/m2_accuracy.csv) + confusion matrix
scripts/m3_unknown_tonic.py  tonic and makam without a given tonic + tonic error plot
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

Effect of histogram resolution (template, Bhattacharyya, grouped CV): 72 to 73% for
bins from 3.75 to 50 cents, but 56.8% with 100-cent bins (the piano grid). Forcing
makam music onto 12 semitones costs ar. 16 points.

## Tonic unknown (M3, nearest template + Bhattacharyya, seed 0)

Test recordings are measured against an arbitrary 440 Hz reference, and the
histogram is rotated to every position against every makam template.
A tonic counts as correct within 25 cents, octave errors ignored.

| Task | Stratified | Grouped | MORTY (2016) |
|---|---|---|---|
| Tonic, makam known | 96.4% | 96.1% | 95.8% |
| Makam, tonic unknown | 67.7% | 66.9% | not reported |
| Makam and tonic both right | 66.9% | 65.9% | 63.6% |

MORTY's exact tonic tolerance may differ from ours, so compare loosely.
Losing the tonic costs ar. 6 points of makam accuracy (73.0% -> 66.9%).
Most wrong tonics are off by a fourth or a fifth (the most prominent non-tonic notes).
The makam annotations' tonics agree with the corrected `otmm_tonic_dataset` within
16 cents for all 998 shared recordings, so we keep them.
