# toxicity-estimator-benchmark

A reproducible test of the SLTS+LASSO estimator (Nwaeme, Lukman et al., *Journal of Nigerian Society of Physical Sciences*) on public toxicity data, against LASSO, Random Forest and gradient boosting.

## What was tested
- **Data (public, not redistributed here):** Tox21 (12 assays), ClinTox (`CT_TOX`, failure in clinical trials for toxicity) and SIDER (`Hepatobiliary disorders`), from the MoleculeNet collection copy at `GLambard/Molecules_Dataset_Collection`. Download `tox21.csv`, `clintox.csv` and `sider.csv` from `https://raw.githubusercontent.com/GLambard/Molecules_Dataset_Collection/master/latest/` into the same folder as `run.py`.
- **Features:** about 700 numbers per molecule: 217 RDKit 2-D descriptors plus a 512-bin log Morgan count fingerprint (radius 2).
- **Splits:** randomized Bemis-Murcko scaffold splits (80/20), three seeds, so test molecules have core structures absent from training. 14 endpoints x 3 splits = 42 cases.
- **Stress test:** 0, 10 and 20 percent of *training* labels flipped at random; test labels are never changed.
- **Models:** LASSO (L1 logistic); SLTS+LASSO with 90 percent and 75 percent of records kept; Random Forest (200 trees); histogram gradient boosting. The penalty of both L1 models is chosen by 3-fold cross-validation on the training data only. Random Forest and gradient boosting use fixed settings and are **not tuned**.
- **Metric:** ROC-AUC (also PR-AUC in `results/results.csv`).

## How the published estimator was adapted
The published estimator was developed for a continuous response. Here, Sparse LTS (concentration steps with an L1-penalised linear fit to the 0/1 label) selects the records to keep, and a LASSO-penalised logistic model is fitted to them. Trimming on the raw residuals discarded most toxic compounds, because they are the minority class, so trimming is done **within each class**. This is an adaptation, not the paper's exact procedure.

## Results (mean ROC-AUC over 42 endpoint-split cases)
| model | 0% flipped | 10% | 20% | loss 0 to 20% |
|---|---|---|---|---|
| Random Forest | 0.818 | 0.775 | 0.726 | 0.092 |
| Gradient boosting | 0.807 | 0.736 | 0.688 | 0.119 |
| LASSO | 0.772 | 0.736 | 0.694 | 0.078 |
| SLTS+LASSO (keep 90%) | 0.742 | 0.723 | 0.687 | 0.055 |
| SLTS+LASSO (keep 75%) | 0.716 | 0.709 | 0.682 | 0.034 |

Random Forest was the most accurate model at every noise level. SLTS+LASSO lost the least accuracy as labels were corrupted and kept about 28 descriptors on average (LASSO: about 50), but it was less accurate than LASSO and Random Forest on clean labels. Full tables are in `results/summary.txt`.

## Limitations
- Random label flips are not the gross-outlier contamination the estimator was designed for.
- Baselines are untuned; a tuned comparison may change the ranking.
- FDA's DILIrank list was not used (it could not be retrieved for this study); ClinTox and SIDER labels are not drug-induced liver injury labels.
- Only three splits per endpoint; differences of about 0.01 AUC are within noise.

## Reproduce
```
pip install rdkit scikit-learn numpy pandas
python run.py tox21 clintox sider   # about one hour on two CPU cores
python analyze.py
```
