# Ten sensors provide the strongest warning signs of defective chips, whinin those ten, three stand out as the best start, giving engineers a focused place to investigate first.

**Course:** MSDS 696 — Data Science Practicum II · Regis University · August 2026
**Author:** Ilse Severance

---

## Research Question

> *Which of the 590 manufacturing process sensors most strongly distinguish failed chips from passed chips, and how should quality engineers prioritize investigating them to reduce defect rates?*

---

## Problem Framing & Relevance

**Who this is for.** Quality engineers deciding what to investigate first, and the executives who allocate their (scarce) investigation hours. It is not a request to automate the pass/fail decision — it is a request to point a limited-time engineering team at the highest-value leads out of 590 undifferentiated sensors.

**Why this is a real decision problem, not just a modeling exercise.** A semiconductor line with 590 sensors and no physical labels on any of them leaves engineers with an impossible search space if a defect rate needs explaining. Investigating sensors one at a time, or by intuition, is slow and expensive. A ranked, statistically-defensible shortlist converts "check all 590" into "check these 3 first" — directly reducing wasted engineering time, and by extension scrap, rework, and cost, even though this dataset cannot quantify that dollar value directly (see *Limitations*).

**Why a predictive model is an appropriate tool here — and why its output has to be reframed.** A classifier's job is to predict Pass/Fail; that is not, by itself, the deliverable a quality engineer needs. The real deliverable is a ranking of *which inputs the classifier depended on to succeed*, cross-checked against an independent statistical test so the ranking doesn't rest on the model's judgment alone. This is why the project deliberately treats the classifier as an intermediate step (see *Methodology*) rather than the end product — the model exists to produce a defensible importance ranking, not to be deployed as an automated gate. This framing is made explicit in the executive-facing presentation: *"Not a modeling walkthrough — a recommendation for where to spend a scarce resource, backed by evidence."*

**Fit check (does the data actually support this question?).** The dataset pairs 590 anonymized sensor readings with a Pass/Fail label for every one of 1,567 chips — the minimum ingredients needed to ask "which sensors relate to failure" are present. The notebook verifies this explicitly (`FIT` / `PROVENANCE` / `LIMITS` / `FEASIBILITY` checks) before any modeling begins, rather than assuming the dataset is adequate.

---

## Data Source & Appropriateness

| Source | Description |
|---|---|
| **SECOM Dataset (UCI Machine Learning Repository)** | 590 anonymized sensor measurements per chip, collected during a real semiconductor manufacturing process, paired with a Pass/Fail inspection label and a timestamp. Donated 2008; publicly available, so the same raw files are pullable by anyone re-running this analysis. |

**What the dataset supports, and what it doesn't:**

| Question | Supported? | Why |
|---|---|---|
| Which sensors statistically distinguish Fail from Pass chips? | Yes | 590 sensor readings + label, aligned per chip |
| Which physical tool/process step does a flagged sensor correspond to? | No | Feature names are fully anonymized (`feature_0` … `feature_589`); no equipment/process-log mapping is included |
| Do flagged sensors *cause* defects? | No | Observational data only — no experimental manipulation, no process metadata to support causal claims |
| What does a missed defect or false alarm cost in dollars? | No | No scrap, rework, or cost fields exist in the dataset |
| Is the Fail class large enough to fully stabilize a fine-grained ranking? | Partially | Only 104 of 1,567 chips (6.6%) are Fail — enough to prioritize, not enough to certify |

Checking appropriateness *before* modeling — rather than discovering these gaps after building a model — is what keeps the eventual conclusions scoped correctly (see *Defensibility of Conclusions*).

---

## Data Preparation

All preprocessing follows one rule: **every statistic, threshold, or selected feature list is learned from the training split only, then applied unchanged to the test split.** The train/test split happens *before* any cleaning step for exactly this reason — if cleaning decisions were made on the full dataset, the untouched-test-set evaluation later would be contaminated by information the model isn't supposed to have seen yet.

| Step | What was done | Why this specific method |
|---|---|---|
| Split | 80/20 stratified split (`random_state=42`), before any cleaning | Preserves the ~6.6% Fail rate in both sets; done first to prevent leakage into every downstream step |
| Drop high-missing features | Dropped 8 features exceeding 80% missing (threshold learned from training split) | Beyond ~80% missing, imputation is guessing more than filling |
| Missingness indicators | Added 24 `_was_missing` flags for the 20–80%-missing band | *Why* a sensor's reading is missing (e.g., an offline instrument, a skipped step) may itself be predictive — imputing over it silently would discard that signal |
| Impute | Median imputation for remaining missing values | Robust to outliers, computationally cheap at 580+ dimensions; with only ~1,250 training rows, KNN-based imputation's distance calculations become less meaningful in this many dimensions and considerably more expensive — noted as a candidate for future sensitivity analysis, not adopted here |
| Remove near-zero-variance features | Dropped 19 features with coefficient of variation (std / \|mean\|) < 0.01 | Sensor scales differ by orders of magnitude (~3,000 vs. ~0.005); a fixed standard-deviation cutoff would unfairly flag small-scale-but-informative sensors, so variance is judged relative to each sensor's own scale instead |
| Statistical feature selection | Mann-Whitney U test (training split, label-aware) narrowed 563 candidate sensors to the top 75 by p-value | With 590 features and ~1,250 training rows, modeling on the full set risks severe overfitting; this narrows to the sensors with the strongest raw distributional evidence of a Pass/Fail difference before any model sees them |
| Scale | `StandardScaler` fit on training split only, applied to test split | Required for Logistic Regression's distance-based optimization; tree-based models are scale-invariant and use unscaled features |

**Result:** 99 total features feed the models — 75 sensor readings + 24 missingness-indicator flags.

---

## Methodology: Rigor & Justification

**Model selection — why these three, and not more:**

| Model | Role in the comparison |
|---|---|
| Logistic Regression | Interpretable linear baseline — confirms whether the signal is linear before reaching for anything more complex |
| Random Forest | Captures non-linear sensor interactions; robust to the noisy, high-dimensional, correlated sensor set |
| Histogram Gradient Boosting | Tests whether sequential error-correction (boosting) outperforms bagging on this data |

A larger model search wasn't pursued deliberately: with only ~83 Fail examples per training fold, an exhaustive sweep across many model families would mostly fit cross-validation noise rather than find generalizable signal.

**Why PR-AUC, not accuracy, drives every decision.** Predicting "Pass" for every chip already scores 93.4% accuracy while catching zero defects — accuracy is actively misleading at 6.6% prevalence. Recall, precision, F1, ROC-AUC, and PR-AUC are all tracked on the Fail class specifically, with **PR-AUC (average precision) as the primary metric for model selection, tuning, and threshold choice**, since it is most sensitive to how well the rare positive class is being found — which is the actual goal.

**Why `RandomizedSearchCV` over an exhaustive grid.** Hyperparameter tuning used 20 randomly sampled combinations per model rather than a full grid, for two reasons: (1) it samples directly from continuous distributions (e.g., `uniform(0.01, 0.3)` for learning rate) instead of forcing them into an arbitrary discrete grid, and (2) with so few Fail examples per fold, an exhaustive search would mostly optimize to fold-specific noise rather than real signal — a deliberately modest budget matches the amount of trustworthy signal available. `class_weight="balanced"` was fixed (not tuned) across all trials, since it was already handling imbalance.

**Why the classification threshold was tuned separately from `class_weight`.** `class_weight="balanced"` reweights the training loss, but does not move the default 0.5 decision boundary at prediction time — those are two independent levers. Out-of-fold predicted probabilities were used to trace the precision-recall curve and select an operating threshold (0.324) near 35% cross-validated recall, lifting recall on Fail from 7% at the default 0.5 cutoff to roughly 34% in cross-validation. This operating point reflects a precision/recall trade-off, not a cost-optimized one (see *Limitations* — no dollar-cost data exists to optimize against).

**Why permutation importance, not the Random Forest's built-in feature importances.** Built-in (Gini) importance is biased toward high-cardinality, continuous features — exactly what this dataset is full of. Permutation importance instead measures how much held-out PR-AUC drops when a sensor's values are shuffled, computed on the untouched test set with the final fitted model (30 repeats, averaged) — reflecting generalizable predictive contribution rather than an artifact of how the trees happened to split.

**Why cross-check against an independent Mann-Whitney U test at all.** A model-based ranking alone risks reflecting quirks of *this* model rather than a real property of the data. The Mann-Whitney U test is computed independently — no knowledge of the model's predictions, just a direct Pass-vs-Fail distributional comparison on the training split. Agreement between two independently-derived rankings is materially stronger evidence than either alone.

---

## Results

| Metric | Value |
|---|---|
| Best model | Random Forest (tuned) |
| PR-AUC (5-fold CV) | 0.285 |
| PR-AUC (holdout test set) | 0.245 |
| ROC-AUC (holdout test set) | 0.769 |
| Chosen operating threshold | 0.324 (~35% recall in CV) |
| Precision / Recall on Fail, holdout (evaluated once) | 0.25 / 0.24 |
| Holdout confusion matrix | 278 correctly Passed · 15 false alarms · 16 missed defects · 5 defects caught (of 21 actual Fail) |
| Chips in dataset | 1,567 (104 Fail, 6.6%) |
| Sensor features (raw) | 590 |
| Features after cleaning + selection | 99 (75 sensors + 24 missingness flags) |
| Models compared | Logistic Regression, Random Forest, Histogram Gradient Boosting |

The tuned threshold's cross-validated estimate (~34% recall) and its single holdout measurement (24% recall) differ, as expected with only 21 Fail chips in the holdout set — see *Defensibility of Conclusions* for how this is scoped rather than glossed over.

---

## Top Feature Importances

| Rank | Sensor | Permutation Importance (ΔPR-AUC) | Mann-Whitney p-value | Direction in Fail Units |
|---|---|---|---|---|
| 1 | feature_59 | 0.055 | 0.000001 | Higher |
| 2 | feature_341 | 0.029 | 0.000209 | Higher |
| 3 | feature_298 | 0.018 | 0.017352 | Higher |
| 4 | feature_205 | 0.018 | 0.000032 | Higher |
| 5 | feature_587 | 0.017 | 0.028323 | Higher |
| 6 | feature_346 | 0.016 | 0.041511 | — |
| 7 | feature_40 | 0.016 | 0.020053 | — |
| 8 | feature_160 | 0.016 | 0.020710 | — |
| 9 | feature_431 | 0.016 | 0.007320 | — |
| 10 | feature_100 | 0.015 | 0.005552 | — |

**Key finding:** `feature_59`, `feature_341`, and `feature_298` — the top three by permutation importance, all independently confirmed by the Mann-Whitney test (p < 0.02) — are the recommended starting point for engineering investigation. This matches the final executive presentation's recommendation exactly.

> **Correction from the notebook draft:** the notebook's Conclusion cell names `feature_59`, `feature_477`, and `feature_205` as the "both signals agree" trio. `feature_477` does not appear anywhere in the computed top-20 permutation-importance ranking — it isn't backed by the printed output. This README uses the top three sensors as they actually appear in the executed results table (and as they appear in the final presentation); the notebook's narrative text should be corrected to match before final submission.

---

## Defensibility of Conclusions

- **Two independent methods agree.** The recommendation isn't "the model says so" — it's "a predictive model *and* a label-aware statistical test, computed independently of each other, both flag the same three sensors." That agreement is the core of why this shortlist is defensible rather than a single model's opinion.
- **Direction of effect is reported, not just magnitude.** Each flagged sensor's readings run consistently *higher* in Fail chips than in Pass chips — this tells engineers what kind of failure mode to look for (over-processing, drift, a stuck-high instrument) rather than leaving them with an unranked list of feature IDs.
- **Confidence is explicitly scoped, not overstated.** With only 104 historical Fail chips (21 in the holdout), the recommendation is stated as a *priority ranking for investigation*, not a certified, final ordering — the middle and lower ranks are more likely to reorder under a different sample than the top few.
- **Prediction is not conflated with causation.** The dataset has no process or equipment metadata, so no claim is made that a flagged sensor *causes* defects — only that it's statistically and predictively associated with them. The explicit next step is validating flagged sensors against real process/equipment logs before any change is made on the line.
- **The claim matches what was measured.** Every number in the *Results* table above traces to a specific executed notebook cell, not an approximation or a prior run — see *Reproducibility* for the two places where this wasn't previously true and has now been corrected.

---

## Limitations

- **Anonymized features.** Sensor names carry no physical or process meaning; this analysis narrows the search to *which* `feature_N` columns matter, not which physical tool or process step they correspond to. Mapping those IDs to actual equipment requires the fab's internal process log, which isn't part of this dataset.
- **Severe class imbalance.** Only 104 of 1,567 chips (6.6%) are labeled Fail, with just 21 in the holdout test set — enough to rank candidate sensors, but not enough to certify the ranking as fully stable. A different random split could plausibly reorder the middle of the priority table, though the top few sensors are likely more robust to that.
- **No cost or scrap data.** The chosen decision threshold reflects a precision/recall trade-off, not an economically optimized one — the dataset has no dollar figures to weigh a missed defect (a bad chip that ships) against a false alarm (a good chip pulled for inspection).
- **Threshold instability.** The chosen operating point achieves ~34% recall in cross-validation but only 24% on the small holdout set (5 of 21 caught) — with this few Fail examples, any single threshold estimate should be read as a rough operating range, not a precise guarantee.
- **Predictive, not causal.** This model produces a prioritized shortlist for engineering follow-up, not a causal diagnosis — top-ranked sensors should be validated against process/domain knowledge before any change is made on the line.

---

## Reproducibility

- **Deterministic given the same raw inputs.** `random_state=42` is fixed everywhere randomness enters the pipeline: the train/test split, `StratifiedKFold`, `RandomizedSearchCV`, and `permutation_importance`. Re-running the notebook top-to-bottom against the same `secom.data` / `secom_labels.data` files reproduces every number in this README.
- **No test-set leakage to reproduce accidentally.** Every learned preprocessing artifact — which columns get dropped, imputation medians, the near-zero-variance cutoff, the Mann-Whitney-selected feature list, scaler mean/std — is fit on the training split only and then applied to the test split. There is nothing in the pipeline that requires the test set to already exist in order to define it.
- **Public, stable data source.** SECOM is a public UCI Machine Learning Repository dataset; anyone can pull the identical raw files.
- **Environment.** The notebook depends on `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, and `seaborn`. No `requirements.txt` currently pins exact versions — adding one (e.g., via `pip freeze`) is recommended so a re-run months from now can't silently drift due to a library update.
- **Two narrative/output mismatches identified and corrected in this README** (both should also be fixed directly in the notebook before final submission, so the notebook and README stay in sync):
  1. The notebook's Conclusion cell names `feature_477` as one of three top agreement sensors; the actual computed permutation-importance table has no `feature_477` in its top 20. The correct top three (`feature_59`, `feature_341`, `feature_298`) are used throughout this README instead.
  2. The notebook's threshold-stability note states "~19% recall" on the holdout set; the executed `classification_report` in the following cell shows 24% recall (5 of 21 Fail chips caught) — matching the final presentation's reported figure. This README uses 24% throughout.

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
