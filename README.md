# Ten sensors provide the strongest warning signs of defective chips, giving engineers a focused place to investigate first

**Course:** MSDS 696 — Data Science Practicum II · Regis University · August 2026
**Author:** Ilse Severance

---

## Research Question

> *Which of the 590 manufacturing process sensors most strongly distinguish failed chips from passed chips, and how should quality engineers prioritize investigating them to reduce defect rates?*

---

## Project Overview

As a quality engineer, the goal of this project is to move machine learning beyond a simple pass/fail classifier and toward something a quality team can act on: a **ranked, defensible shortlist of sensors** worth investigating first. Using the SECOM semiconductor manufacturing dataset — 590 anonymized sensor readings per chip, with a ~6.6% Fail rate — a **Random Forest** model is trained to detect defective chips, and its **permutation importance** is cross-checked against an independent **Mann-Whitney U test** to surface the sensors both signals agree matter. The result is a prioritized list quality engineers can use to focus process-control investigation where it's most likely to reduce scrap, rework, and manufacturing cost.

---

## Key Results

| Metric | Value |
|---|---|
| Best model | Random Forest (tuned) |
| PR-AUC (5-fold CV) | 0.285 |
| PR-AUC (holdout test set) | 0.245 |
| ROC-AUC (holdout test set) | 0.769 |
| Chosen operating threshold | 0.324 (~35% recall in CV) |
| Precision / Recall at threshold (holdout) | 0.25 / 0.24 |
| Chips in dataset | 1,567 (104 Fail, 6.6%) |
| Sensor features (raw) | 590 |
| Features after cleaning + selection | 99 (75 sensors + 24 missingness flags) |
| Models compared | Logistic Regression, Random Forest, Histogram Gradient Boosting |

*Note: the holdout numbers above come directly from the notebook's final evaluation cell — the Conclusion narrative should be checked against these exact figures before final submission, since an earlier draft of the write-up cited slightly different numbers from a prior run.*

---

## Data Source

| Source | Description |
|---|---|
| **SECOM Dataset (UCI Machine Learning Repository)** | 590 anonymized sensor measurements per chip, collected during semiconductor manufacturing, paired with a Pass/Fail inspection label and a timestamp. No cost, scrap, or process-step metadata is included — sensor identities are not mapped to physical equipment. |

---

## Project Structure

```
.
├── MSDS696_Data_Science_Practicum_II - Ilse Severance.ipynb   # Main analysis notebook
├── MSDS696_Data_Science_Practicum_II-Final - Ilse Severance.pptx
├── Chip_Failure_Executive_Briefing.pptx / .pdf / .html        # Executive-facing presentation
├── LLM_Collaboration_Log_Enriched.md / .pdf                   # Running LLM collaboration log
├── secom.data / secom_labels.data / secom.names               # Raw SECOM source files
├── secom_combined.csv                                         # Combined features + labels + timestamp
└── Week 1 … Week 8 /                                          # Weekly status report folders
```

---

## Methodology Summary

**Target variable:** Pass/Fail inspection outcome (Fail = 1, ~6.6% prevalence)

**Preprocessing (all thresholds/statistics learned from the training split only, to avoid leakage):**
- Dropped 8 features exceeding 80% missing values
- Added missingness-indicator flags for the 20–80%-missing band (missingness itself may be informative)
- Median-imputed remaining missing values
- Dropped 19 near-zero-variance features (coefficient of variation < 0.01)
- Mann-Whitney U test narrowed 590 sensors to the top 75 most discriminative, plus 24 missingness flags (99 features total)

**Modeling:**
- 80/20 stratified train/test split; 5-fold stratified cross-validation for model selection
- Three models compared under `class_weight="balanced"`: Logistic Regression, Random Forest, Histogram Gradient Boosting
- Hyperparameters tuned via `RandomizedSearchCV` (20 iterations), scored on PR-AUC (average precision) — the primary metric given the class imbalance
- Classification threshold tuned from out-of-fold CV predictions to a ~35%-recall operating point, rather than using the default 0.5

**Feature importance:**
- Permutation importance (30 repeats) computed on the untouched holdout set, using the final fitted model
- Cross-checked against Mann-Whitney U p-values computed independently on the training set — sensors where both signals agree are treated as the most defensible leads

---

## Top Feature Importances

| Sensor | Permutation Importance (ΔPR-AUC) | Mann-Whitney p-value | Direction in Fail Units |
|---|---|---|---|
| feature_59 | 0.055 | 0.000001 | Higher |
| feature_341 | 0.029 | 0.000209 | Higher |
| feature_298 | 0.018 | 0.017352 | Higher |
| feature_205 | 0.018 | 0.000032 | Higher |
| feature_587 | 0.017 | 0.028323 | Higher |

**Key finding:** `feature_59`, `feature_477`, and `feature_205` show agreement between the model-based ranking and the independent statistical test — the strongest, most defensible starting point for engineering investigation.

---

## Limitations

- **Anonymized features.** Sensor names carry no physical or process meaning; this analysis narrows the search to *which* `feature_N` columns matter, not which physical tool or process step they correspond to.
- **Severe class imbalance.** Only 104 of 1,567 chips (6.6%) are labeled Fail, with just 21 in the holdout test set — enough to rank candidate sensors, but not enough to certify the ranking as fully stable.
- **No cost or scrap data.** The chosen decision threshold reflects a precision/recall trade-off, not an economically optimized one — the dataset has no dollar figures to weigh a missed defect against a false alarm.
- **Threshold instability.** The chosen operating point achieves ~35% recall in cross-validation but only ~19% on the small holdout set, underscoring how imprecise a single threshold estimate is with this few Fail examples.
- **Predictive, not causal.** This model produces a prioritized shortlist for engineering follow-up, not a causal diagnosis — top-ranked sensors should be validated against process/domain knowledge before any change is made on the line.
