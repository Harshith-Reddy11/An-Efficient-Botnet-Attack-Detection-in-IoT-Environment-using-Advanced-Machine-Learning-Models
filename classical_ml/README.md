# Classical ML implementation (IJERT methodology)

This directory adds a separate implementation of the paper's classical machine-learning approach while leaving the original notebooks, scripts, models, and web application untouched.

## Methodology represented

1. Load `Dataset/UNSW_NB15.csv` and use `attack_cat` as the multiclass target.
2. Remove exact duplicate rows; replace infinite values and median-impute missing numeric predictors.
3. Exclude `label` because it is a second target derived from attack status and would leak target information into an `attack_cat` classifier.
4. Split into stratified 70% training / 30% test partitions (the paper's Figure 3.4 specifies 70/30).
5. Remove highly correlated predictors (absolute Pearson correlation > 0.95), using training data only.
6. Rank/select features using Random Forest feature importance, fitted on training data only.
7. Apply SMOTE to the training partition only.
8. Train Logistic Regression, SVM, and Random Forest, then combine them with hard (majority) voting.
9. Evaluate accuracy, weighted precision, weighted recall, weighted F1, confusion matrices, and runtime on the untouched test partition.

## Run

From the repository root:

```bash
python -m pip install -r classical_ml/requirements.txt
python classical_ml/train_evaluate.py
```

Optionally specify a dataset path:

```bash
python classical_ml/train_evaluate.py --data "Dataset/UNSW_NB15.csv"
```

Outputs are written to `classical_ml/results/`. The script also saves trained estimators and preprocessing metadata for reproducibility.

## Important research note

The paper reports individual accuracies of approximately 91% (Logistic Regression), 93% (SVM), and 97% (Random Forest), and describes a majority-voting hybrid. It does not specify complete hyperparameters, the exact split seed, or a numerical accuracy for the hybrid in the results text. Consequently, this implementation uses documented, reproducible defaults; its measured results must not be represented as an exact reproduction of the paper's reported numbers. The paper's other pages contain a separate/inconsistent deep-learning discussion (ANN/CNN/LSTM/stacked models); this directory specifically implements the LR/SVM/RF methodology described in the abstract and Section III-E.

## Dataset and citation

The UNSW-NB15 dataset is not redistributed by this folder. Use the dataset file already present in `Dataset/` if you have the right to redistribute it, otherwise obtain it from its official source and follow its terms.

Golanakonda, S., Nallamala, S., & Rajalingari, H. R. (2026). *An Efficient Botnet Attack Detection in IoT Environment using Advanced Machine Learning Models*. International Journal of Engineering Research & Technology (IJERT), 15(08), IJERTV15IS080644. https://www.ijert.org/an-efficient-botnet-attack-detection-in-iot-environment-using-advanced-machine-learning-models-ijertv15is080644

## Fresh run in this repository

A run on the supplied `Dataset/UNSW_NB15.csv` (after exact-duplicate removal) produced the following held-out results. The target is the ten-class `attack_cat` field; 12,853 unique records remained, with a stratified 70/30 split and 3,856 test records.

| Model | Accuracy | Weighted precision | Weighted recall | Weighted F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 64.42% | 72.51% | 64.42% | 67.03% |
| SVM (RBF) | 68.88% | 76.10% | 68.88% | 70.98% |
| Random Forest | 79.02% | 79.52% | 79.02% | 79.23% |
| Hard majority voting | 72.20% | 76.90% | 72.20% | 73.73% |

These are the results of this newly executed, documented configuration—not the paper's reported values. They do not reproduce the paper's approximate 91%/93%/97% individual accuracies. The paper does not provide enough experimental detail to establish an exact reproduction, and the supplied CSV's class distribution changes substantially after duplicate removal (12,853 unique rows from 20,398 rows). Do not replace these measured values with the paper's figures in the repository.
