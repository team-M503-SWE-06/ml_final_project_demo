# bina.az price modelling with a from-scratch decision tree and SVM

MLE-AI-201 Machine Learning: final project. We clean the **bina.az** Baku real-estate
dump and frame it as two tasks:

- **Task A, regression:** predict `log(price)` (metrics: RMSE, MAE and R² on log price and on AZN).
- **Task B, classification:** predict the price tier (premium = price above the **training-set median**).
  Metrics: precision, recall, F1, ROC-AUC, PR-AUC and the confusion matrix.

Both models are **our own NumPy code**:

| File | What is in it |
|---|---|
| `src/decision_tree.py` | CART tree: Gini / entropy / MSE, vectorised threshold search, `max_depth`, `min_samples_split`, `min_samples_leaf`, `min_impurity_decrease`, `predict_proba`, impurity-decrease feature importances |
| `src/svm.py` | Pegasos soft-margin SVM (mini-batch sub-gradient on the hinge primal, 1/(λt) / invscaling / constant schedules, projection, iterate averaging, margin and support vectors) plus **random Fourier features** for the RBF kernel |
| `src/data_prep.py` | loading, cleaning (logged), leakage removal, train-only feature statistics, stratified split, tier label |
| `src/evaluate.py` | NumPy metrics (RMSE/MAE/R², confusion matrix, P/R/F1, ROC-AUC, PR-AUC), k-fold CV, bootstrap |
| `src/run_all.py` | **one command** that reproduces every number, table and figure |
| `tests/` | pytest suite: the tree reproduces sklearn's splits on a toy example, every hyperparameter changes the tree, the SVM matches `SVC` on a toy split, the metrics match sklearn |

scikit-learn is used **only** for the baselines (`DecisionTree*`, `SVC`, `SGDClassifier` in a scaling
`Pipeline`) and as a test oracle.

## Reproduce everything

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.get_data        # downloads data/house_sale.csv (see data/README.md)
python -m pytest -q           # 32 tests
python -m src.run_all         # ~15-20 min on a laptop: all numbers, tables, figures
```

`run_all` prints the cleaning log, the tuning sweeps, the final configs, the test-set results
table, descriptive post-selection fold variability, timings and the error analysis. The
fold results are not leakage-free performance estimates: the configurations were selected
using the validation labels, and feature statistics were fitted once on the original training
split before folds were created. It writes:

- `report/figures/*.png`: every figure in the report and slides
- `report/tables/*.tex`: the results tables and `numbers.tex`, which holds every number quoted in the report text
- `results/metrics.json`: everything in machine-readable form

To build the PDFs (MiKTeX/TeX Live `pdflatex`), run this after `run_all`:

```bash
cd report && pdflatex report.tex && pdflatex report.tex
cd ../presentation && pdflatex slides.tex && pdflatex slides.tex
```

## Protocol (no leakage)
- `unit_price` and `total_price` are dropped because both are functions of `price`. Identifiers, free text and `views` are dropped too.
- The split is 70 / 15 / 15, stratified on price deciles and seeded (`SEED = 42`).
- The tier threshold, imputation medians, one-hot vocabularies and standardisation statistics all come from the **training split only**.
- Hyperparameters are chosen on the **validation** split. The test split is used once, in step 5 of `run_all`.

## Repo layout
```
├── README.md  requirements.txt  contribution_report.md
├── data/            dataset goes here (not committed)
├── src/             from-scratch models + pipeline
├── tests/           pytest suite
├── report/          report.tex (+ generated figures/ and tables/)
└── presentation/    slides.tex -> slides.pdf
```
