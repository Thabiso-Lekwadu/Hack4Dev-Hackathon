# Hack4Dev 2026 — Building Age Classification: Solution

Transferable Madrid → Amsterdam building-age classifier from 42 years of Landsat time series (1984–2025).

A LightGBM model learns building eras from Madrid (Stage 1); three light learners then adapt it to Amsterdam from 5–200 labelled pixels per era (Stage 2).

## Results (seed 42, executed end to end)

| Model | n per class | Macro F1 (mean ± std) | Notebook 4 baseline |
|---|---|---|---|
| Madrid, Stage 1 (5-fold × 5-repeat CV) | all | **0.783 ± 0.004** | 0.618 |
| Amsterdam, Stage 2 | 5 | **0.645 ± 0.025** | 0.544 |
| Amsterdam, Stage 2 | 25 | **0.697 ± 0.014** | 0.599 |
| Amsterdam, Stage 2 | 50 | **0.710 ± 0.013** | 0.608 |
| Amsterdam, Stage 2 | 100 | **0.740 ± 0.010** | 0.613 |
| Amsterdam, Stage 2 | 200 | **0.755 ± 0.007** | 0.615 |

Zero-shot Amsterdam (no labels): 0.486 (baseline 0.343).
Prize metrics: average F1 0.709 · peak F1 0.755 · low-data (n = 25) F1 0.697.

**Spatially blocked evaluation** (whole 2 km districts held out, 360 m buffer; Notebook 7, Section 7.7): Madrid 0.634 (baseline 0.588); Amsterdam averaged over 5–200 labels 0.693 (baseline 0.588). Random-pixel CV is optimistic because neighbouring pixels are near-duplicates; use the blocked numbers for unseen districts.

## How to run

1. Put `madrid_train.parquet` and `amsterdam_data.parquet` in `data/` (not included in this repository; see `data/README.md`).
2. `pip install -r requirements.txt`
3. Run `5-Improved_Feature_Engineering.ipynb` (~3 min), then `6-Improved_Modelling_and_Transfer.ipynb` (~60 min on 2 cores; most of it is the 25-fit Madrid CV). Optionally run `7-Decision_Evidence_and_Leakage_Audit.ipynb` (~2 h) to reproduce the evidence for every decision and the leakage audit. Set the environment variable `H4D_QUICK=1` for a few-minute smoke run of Notebooks 6 and 7 (scores not representative).

To adapt the Stage 1 model with a different budget, use `adapt_to_city(stage1, X_city, pix_city, support_idx, y_support, n)` in Notebook 6, Section 6.7.

## Files

| Path | Contents |
|---|---|
| `5-Improved_Feature_Engineering.ipynb` | Pixel × year × band cube, city-year anomalies, temporal + change-point + spatial features |
| `6-Improved_Modelling_and_Transfer.ipynb` | Stage 1 CV and final model, zero-shot, transfer procedure, Amsterdam CV, classification reports, per-class F1 heatmap, prediction and error maps |
| `7-Decision_Evidence_and_Leakage_Audit.ipynb` | Each decision re-tested against its alternative (RF vs LightGBM, class weights, feature ablation, scaling, Stage 2 variants), leakage audit with blocked-CV maps, decision-evidence summary table |
| `models/stage1_madrid_lgbm.txt`, `models/stage1_madrid_meta.json` | Trained Stage 1 model + feature list, parameters, settings |
| `results/` | `results_table.csv`, `results_summary.json` (all fold scores, confusion proportions), figures |
| `docs/` | Plain-language report on feature engineering, scaling, leakage and transfer (PDF and Word), and the 300-word written justification |
| `run_all.sh` | Runs Notebooks 5–7 in order |
| `solution/` | Experiment scripts behind the design decisions (`exp1`–`exp4`), shared helpers, notebook builders |

Every design decision is commented in the notebooks (search for `DECISION`). The plain-language write-up and the 300-word justification are in `docs/`.
