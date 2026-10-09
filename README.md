# Makam Recognition

Recognizing the makam (melodic mode) of Ottoman-Turkish music recordings from their melody.
Work in progress, milestone M4 (does melodic order help?).

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
python scripts/m4a_order_register.py --data ../otmm_makam_recognition_dataset

# 4. neural network (M4b), needs PyTorch
pip install -e ".[dl]"
python scripts/m4b_cnn.py --data ../otmm_makam_recognition_dataset --quick
python scripts/m4b_cnn.py --data ../otmm_makam_recognition_dataset
```

The first run parses 1,000 text files and saves binary copies in `cache/`; later runs are faster.

## Layout

```
src/makam/data.py       load annotations, metadata and pitch tracks
src/makam/features.py   Hz -> cents, octave folding, pitch-class histograms
src/makam/classify.py   distances, kNN and nearest-template classifiers
src/makam/evaluate.py   composition groups, stratified and grouped cross-validation
src/makam/tonic.py      tonic estimation by rotating histograms against templates
src/makam/sequence.py   pitch-class-gram: the melody as a (pitch bins x time) array
src/makam/cnn.py        1D CNN and its training loop (PyTorch)
scripts/m1_explore.py   dataset stats + figures in figures/
scripts/m2_baseline.py  accuracy table (results/m2_accuracy.csv) + confusion matrix
scripts/m3_unknown_tonic.py  tonic and makam without a given tonic + tonic error plot
scripts/m4a_order_register.py  register vs melodic order, logistic regression (ar. 5 to 15 min)
scripts/m4b_cnn.py      1D CNN on the melody in time order, same folds
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

## Does order help? (M4a, tonic known, logistic regression, grouped CV, seed 0)

The histogram throws away WHEN notes happen (the seyir, the melodic path of a makam)
and, once folded, WHERE in the register they happen. Same classifier, four inputs:

| Input | Order | Register | Accuracy |
|---|---|---|---|
| F0 folded histogram (M2 input) | no | no | 73.3% |
| F1 3-octave histogram | no | yes | 71.8% |
| F2 folded histogram per third of the piece | yes | no | **77.4%** |
| F3 3-octave histogram per third | yes | yes | 73.4% |

Order helps: F2 beats F0 by +4.1 (seed 0), +3.1 (seed 1) and +3.7 points (seed 2),
winning 7 to 8 of 10 folds. Register does not help, and adding it to order hurts
(720 features for 900 training recordings). The biggest gains are on the pairs the
histogram confuses: Ussak 44% -> 62%, Muhayyer 56% -> 72%, Nihavent 54% -> 70%.

For ar. a quarter of recordings the annotated tonic is one octave away from the
melody. Folded features never noticed; register features normalize the octave
from the melody itself (`normalize_octave`).
Results can move by a few tenths of a point between machines (parallel solver).

Which part of the piece matters (same classifier, 15-cent folded histograms):

| Input | Accuracy |
|---|---|
| Whole piece, one histogram | 72.6% |
| Last third only | 61.4% |
| First third only | **77.4%** |
| 2 / 3 / 5 / 10 sections | 74.3% / 77.4% / 77.7% / 75.8% |

The opening carries the information, the ending does not: every makam ends on its
tonic (the karar), so endings look alike once measured from the tonic. More sections
help up to 3 to 5, then the feature count (800 for 10 sections) starts to overfit.
