# Decoupling Representation Learning from Decision Heads in Deep Tabular Models for Breast Cancer Cytology

Code, data and saved results for the paper
*"Decoupling Representation Learning from Decision Heads in Deep Tabular Models for Breast Cancer Cytology:
A Calibration-, Geometry- and Significance-Aware Evaluation"* (M. F. Almufareh, Jouf University).

Every number and figure in the paper is computed from the saved predictions in `results_v3/` (WDBC) and
`results_wbco/` (Wisconsin Original). These folders, and the `*log*.txt` files, are the original outputs and logs
of the runs used for the paper (NVIDIA RTX 4000 Ada).

## Repository layout

```
wdbc_rerun.py                  full experiment pipeline (training, evaluation, statistics, figures)
extra_experiments.py           CatBoost/TabPFN baselines, feature-wise robustness, Friedman/Nemenyi, decision curves
make_figures.py                re-draws all paper figures from the saved results (no retraining)
breast-cancer-wisconsin.data   Wisconsin Breast Cancer (Original) data file (UCI id 15)
fonts/                         Inter font used in the figures (SIL Open Font License)
results_v3/                    outputs of the WDBC run (569 samples, 30 features); extra/ = extra_experiments.py
results_wbco/                  outputs of the Wisconsin Original run (699 samples, 9 features); extra/ likewise
figures_paper/                 the paper figures made by make_figures.py
*log*.txt                      console logs of the runs
requirements.txt               Python packages
```

## Data

| Dataset | Samples | Features | Source |
|---|---|---|---|
| Wisconsin Diagnostic Breast Cancer (WDBC) | 569 (357 benign, 212 malignant) | 30 | UCI, https://doi.org/10.24432/C5DW2B (loaded with `sklearn.datasets.load_breast_cancer`, which is the same data) |
| Wisconsin Breast Cancer (Original) (WBCO) | 699 (458 benign, 241 malignant) | 9 | UCI, https://doi.org/10.24432/C5HP4Z (file `breast-cancer-wisconsin.data`) |

Malignant is the positive class. The 16 missing *Bare Nuclei* values in WBCO are replaced by the
median of the training part of each split.

## Install

Python 3.10 or newer. Install the CUDA build of PyTorch for your GPU first
(https://pytorch.org/get-started/locally/), then:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the results

```bash
# quick check that everything runs (a few minutes; numbers are NOT valid)
python wdbc_rerun.py --quick --out quick_test

# WDBC (about 13 min on one NVIDIA RTX 4000 Ada)
python wdbc_rerun.py --out results_v3

# Wisconsin Original (about 11 min)
python wdbc_rerun.py --dataset wbco --out results_wbco

# all paper figures for both datasets -> figures_paper/ and figures_paper.zip
python make_figures.py --both

# extra analyses (same splits; written to <results folder>/extra/)
python extra_experiments.py --dataset wdbc --res results_v3
python extra_experiments.py --dataset wbco --res results_wbco
```

If a run stops, start the same command again: finished cross-validation folds are reloaded from
`<out>/cv_cache/`. TabPFN is optional (`--tabpfn`) because it needs a one-time model licence/download;
it is run separately in `extra_experiments.py`.

## Settings (Table 2 of the paper)

| Setting | Value |
|---|---|
| Split | stratified 70 / 15 / 15 (test n = 86 for WDBC, 105 for WBCO) |
| Repeated cross-validation | 10 × stratified 5-fold, 15% of each training fold held out for validation |
| Preprocessing | standardisation and mutual-information ranking fitted on training data only; number of features chosen on validation AUC from {10, 15, 20, 25, 30} (WDBC) or {3, 5, 7, 9} (WBCO) |
| Backbones | FT-Transformer, DCN-V2, TabNet, TabMLP-Mixer; 128-d embedding from the penultimate layer |
| Training | Adam, learning rate 1e-4, weight decay 1e-5, batch 32, max 100 epochs, early stopping (patience 10, validation loss), dropout 0.3, 4 attention heads, gradient clipping 1.0 |
| Decision heads on frozen embeddings | XGBoost (300 trees, depth 6, lr 0.05), ExtraTrees (500 trees), kNN (k and weighting chosen on validation) |
| Baselines | the same heads on raw features, Logistic Regression, Random Forest, SVM-RBF, HistGradientBoosting, LightGBM; CatBoost (early stopping on the validation split) and TabPFN in `extra_experiments.py` |
| Statistics | exact McNemar and DeLong tests, Holm correction |
| Calibration | Brier score, ECE (10 equal-width bins), calibration slope and intercept |
| Robustness | Gaussian noise σ from 0 to 1.0, 100 repeats |
| Explainability | SHAP (KernelExplainer) and LIME on the test set, Spearman correlation of importances |
| Seed | 42 |

## Main output files (per dataset)

| File | Content |
|---|---|
| `single_split_test_metrics.csv` | test-set metrics and confusion-matrix counts of every model |
| `cv_metrics_summary.csv`, `cv_metrics_per_fold.csv` | repeated-CV mean ± SD, 95% CI, and every fold |
| `cv_pooled_oof_calibration.csv` | calibration on pooled out-of-fold predictions |
| `stats_cv_oof_repeat0.csv`, `stats_single_split_test.csv` | McNemar and DeLong tests (raw and Holm-corrected p-values) |
| `geometry_raw_vs_embeddings.csv` | silhouette, Davies–Bouldin, intra/inter-class distances |
| `robustness_noise.csv` | AUC under Gaussian noise |
| `xai_feature_importance_test.csv`, `xai_shap_lime_agreement.csv` | SHAP and LIME importances and their agreement |
| `latency.csv` | inference time per sample |
| `*_probabilities*.csv` | saved predicted probabilities (all metrics can be recomputed from these) |
| `env.json`, `run_log.txt` | software versions, settings and log of the run |

## Hardware used

NVIDIA RTX 4000 Ada Generation (CUDA 12.6), Python 3.14, PyTorch 2.14, XGBoost 3.4, scikit-learn 1.9.

## Citation

The paper is under review. Please cite it once it is published.
