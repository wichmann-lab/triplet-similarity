# Same stimuli, different spaces

Data, code and materials for

> Aguilar, G., Schönmann, I., Mühlbauer, B. F., Künstle, D.-E., & Wichmann, F. A.
> *Same stimuli, different spaces: how task, observer and amount of data
> influence geometry and reliability of mental representations.*

Observers judged the similarity of 54 images (three sets of 18: *action*,
*taxonomic*, *greebles*) in two triplet tasks, *classic* and *odd one out*.
Experiment 1: five observers (O1 - O5) answered the same trials three times.
Experiment 2: two observers (P1, P2) answered all possible trials; P1 is O5.

## Contents

| Folder | Content |
|---|---|
| `experiment/`, `experiment_mid_lab/` | PsychoPy experiments of the two lab setups, condition files, stimuli and raw data (one CSV per session). |
| `resources/` | Image metadata, checksums of the THINGS images, instruction video (Git LFS). |
| `scripts/` | Analysis modules, notebooks and unit tests (`scripts/tests`). |
| `data/` | Fitted embeddings, cross-validation results and simulation outputs. |
| `plots/` | Figures. |

## Quick start

```bash
conda env create -f environment.yml
conda activate triplet-similarity
cd scripts
python -m pytest tests
SOURCE_DATE_EPOCH=0 jupyter nbconvert --to notebook --execute --inplace analysis_reliability.ipynb
```

Run notebooks from `scripts/`. `SOURCE_DATE_EPOCH=0` makes the PDF figures
byte-identical between runs. Figures use the Arial font.

Raw data are read only through `scripts/load_data.py::load_all()`, which
cleans them (training and timeout trials, left-right swap, triplet
conversion) and returns one row per session and image set.

## Manuscript figures and tables

| Item | Notebook in `scripts/` |
|---|---|
| Figures 1, 2, S3, S4, S5; Tables 1, 2, S4, S5, S6 | `analysis_reliability.ipynb` |
| Session timing (Methods) | `time_btwn_sessions.ipynb` |
| Embeddings and dimensionality of all observers (about 3 h) | `finding_dimensionality_aggregated_per_observer.ipynb` |
| Tables 3, 5, S7, S10 | `reads-and-prints-dimensionalities.ipynb` |
| Figures 3, S6, S7 | `plot_all_embeddings.ipynb` (one image set and criterion per run) |
| Figure 4 | `triplet_accuracy_for_diff_tasks_and_obs.ipynb` |
| Figure 5; Tables S8, S9 | `embedding_analysis_subsampling_P1_P2.ipynb`, `visualize_P1_P2_subsampling_analysis.ipynb` |
| Table 4 (pooling designs A, B, C) | `pooled_design_{A,B,C}_*.ipynb`, `pooled_triplet_accuracy_vs_dimension.ipynb` |
| Figure 6 (effect-size criterion) | `analysis-effectsize-criterion.ipynb` |
| Figure S1; Table S3 | `Triplet_error_over_repetition.ipynb` |
| Figure S2 (about 25 min) | `simulate_error_in_triplet_sampling.ipynb` |
| Figures S8 - S14 | `visualize_triplet_accuracy_vs_dimensionality.ipynb` (one observer per run) |
| Figure S15 | `make_radar_plots.ipynb` |
| Condition files of the experiment | `create_conditions.ipynb` |

Figures 3, S6 and S7 were finished in Inkscape (legends and stimulus
thumbnails); their sources are the `.svg` files next to `*-edited.pdf` in
`plots/`. The pooling notebooks refit only when their results in `data/` are
missing (about 8 h in total); otherwise they check the rebuilt data against the
saved runs.

## Reproducibility

- Sessions are combined in the fixed order of `data/original_session_order.txt`.
- Every fit and every random step takes an integer seed (`SEED = 0`), and fits
  run on 10 parallel jobs; the number of jobs does not change the results.
- Three conditions change between one and two dimensions with the seed (O2,
  odd one out, greebles; O4, odd one out, taxonomic; P1, odd one out, action).
  For these we report the majority over SOE seeds 0 - 9
  (`seed_stability.py`, `data/dimensionality_seed_stability_published_inputs.csv`).
- The Experiment 2 subsampling runs (`data/subsampling_analysis-*.csv`) take
  days to refit. Setting `do_simulation = True` in
  `embedding_analysis_subsampling_P1_P2.ipynb` refits them with a fixed seed,
  which reproduces them up to the random variation of the subsamples.

## Stimuli

The 29 images taken from the THINGS database may not be redistributed and are
not included. To run the experiment, download `images_THINGS.zip` from
<https://osf.io/jum2f/>, unzip it, and run from `scripts/`:

```bash
python restore_things_images.py <folder with the THINGS images>
```

which copies the images into both experiment folders after checking their
checksums. Sources and licences of all stimuli are in [STIMULI.md](STIMULI.md).
The analysis does not need the images.

## Licence

Code: GNU General Public License v3.0 ([LICENSE](LICENSE)). Data and figures:
Creative Commons Attribution 4.0 ([LICENSE-data](LICENSE-data)). Stimuli keep
the licences of their sources ([STIMULI.md](STIMULI.md)).

## Citation

See [CITATION.cff](CITATION.cff).
