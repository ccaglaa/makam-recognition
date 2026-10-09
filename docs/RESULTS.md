# Detailed results and lab notes

Every experiment of the project, milestone by milestone, with setup commands and
all the numbers. The short version is the [README](../README.md).

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
python scripts/m4b_cnn.py --data ../otmm_makam_recognition_dataset --region 0.333 --ensemble

# 5. error analysis, unseen scores and the audio demo (SymbTr scores, ar. 100 MB)
git clone https://github.com/MTG/SymbTr.git ../SymbTr
python scripts/m5_errors.py --data ../otmm_makam_recognition_dataset
python scripts/m5_scores.py --data ../otmm_makam_recognition_dataset --symbtr ../SymbTr
python scripts/m5_demo.py --data ../otmm_makam_recognition_dataset --symbtr ../SymbTr --makam Saba
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
src/makam/symbtr.py     read SymbTr scores, turn them into pitch tracks or audio
src/makam/pitch.py      YIN pitch tracker: audio -> melody, same format as the dataset
src/makam/pipeline.py   recognizer: melody in, tonic and ranked makams out
scripts/m1_explore.py   dataset stats + figures in figures/
scripts/m2_baseline.py  accuracy table (results/m2_accuracy.csv) + confusion matrix
scripts/m3_unknown_tonic.py  tonic and makam without a given tonic + tonic error plot
scripts/m4a_order_register.py  register vs melodic order, logistic regression (ar. 5 to 15 min)
scripts/m4b_cnn.py      1D CNN on the melody in time order, same folds
scripts/m5_errors.py    where the errors are, and is the confidence trustworthy
scripts/m5_scores.py    trained on performances, tested on 1206 unseen scores
scripts/m5_demo.py      score -> audio (.wav) -> YIN -> tonic and makam
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

## Can a neural network learn the order? (M4b, 1D CNN, tonic known, grouped CV)

The melody becomes a pitch-class-gram (48 bins of 25 cents x 0.2 s steps) plus a
channel giving the position in the piece. A small 1D CNN (3 conv blocks, 64 filters)
trains on random 30 s windows and predicts a recording by averaging its windows.
Hyperparameters were fixed in advance, not tuned on test folds.

| Model | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| CNN, whole piece (train 84%) | 71.9% | | |
| CNN, opening third only (train 94%) | 72.0% | 72.1% | 71.0% |
| F2 per-third histograms (M4a) | 77.4% | 77.4% | 77.2% |
| **CNN + F2, probabilities averaged** | **78.7%** | **79.7%** | **78.6%** |

Diagnostics, each from one measurement:
* Predicting a whole recording in one pass scored 68.2%; averaging 30 s windows
  like in training scored 71.9%. The train/test mismatch cost 3.7 points
* 100 epochs instead of 40 raised training accuracy (84% -> 91%) but not test
  accuracy (71.5%): the network overfits, more training does not help
* Alone, the CNN only matches a whole-piece histogram (72.6%). With ar. 900
  training recordings it does not learn on its own what the per-third
  histograms encode by design
* Combined with F2 it adds a small but consistent gain: +1.3, +2.3 and +1.4
  points on three seeds, 16 wins, 9 losses and 5 ties over 30 folds.
  The gain does not come from the hardest pair (Ussak and Muhayyer get worse)

## Error analysis (M5, F2, tonic known, grouped CV)

Errors are spread evenly: by length 75 to 79%, by octave of the annotated tonic
75 to 78%, by instrumentation 72% (solo instrumental) to 84% (duet).
Confidence is meaningful on recordings: when the model is 80 to 95% sure it is
right 96% of the time, above 95% it is right 100% of the time (76 recordings).

## From performances to scores (M5)

Trained on all 1000 recordings, tested on the SymbTr scores of the same 20 makams
whose composition is NOT among the recordings (1206 of 1454; 248 removed as
leakage). Each score becomes a pitch track at a random tonic between 180 and 330 Hz.

| Task | Scores | Recordings (CV) |
|---|---|---|
| Tonic found (within 25 cents) | 80.5% | 96.1% |
| Makam, tonic given (F2) | 69.2% | 77.4% |
| Makam, tonic unknown (template + F2) | 55.7% | ar. 67% |
| Balanced accuracy, tonic unknown | 60.3% | - |

Two separate failure modes:
* Tonic: Hicaz (45%) and Acemkurdi (17%) tonics are mostly placed a fourth too
  high, on the dominant. With the tonic given, both are 83 to 94% correct
* Same-scale pairs: Nihavent (17%, read as Sultaniyegah), Ussak (15%, read as
  Beyati or Muhayyer), Mahur (40%, read as Acemasiran) fail even with the tonic
  given. Notated and performed pitches differ by less than 10 cents here, so the
  cues that separate these pairs in recordings are probably performance habits
  (improvised openings, ornaments) that scores do not contain. A hypothesis, untested
* Confidence does not transfer: on scores, predictions made with more than 95%
  confidence are right only 73% of the time (100% on recordings)

The demo synthesizes 60 s of an unseen score, extracts the melody with YIN
(median error under 0.1 cent on synthetic audio) and recognizes it from the
audio alone. Example runs (seed 1): Saba right at 100%; Hicaz wrong, tonic a
fourth too high; Ussak tonic right but makam ranked 3rd after Beyati and Muhayyer.

## Limitations

* One dataset of 1000 recordings, 20 makams, predominant-melody tracks only (no audio)
* Tonic octave is unreliable in ar. 25% of annotations; folded features avoid it
* Scores are synthesized with exact pitches, no ornaments or vibrato
* The demo has not been run on real recorded audio yet
