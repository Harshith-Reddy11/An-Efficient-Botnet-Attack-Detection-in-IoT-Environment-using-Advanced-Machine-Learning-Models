
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "Dataset" / "UNSW_NB15.csv"
OUT = Path(__file__).resolve().parent / "results"
RANDOM_STATE = 42


def prepare_data(csv_path: Path, test_size: float = 0.30):
    df = pd.read_csv(csv_path, low_memory=False)
    if "attack_cat" not in df.columns:
        raise ValueError("Expected target column 'attack_cat' in the dataset.")

    # Remove exact duplicate records and normalize missing/infinite values.
    df = df.drop_duplicates().replace([np.inf, -np.inf], np.nan)
    y_raw = df["attack_cat"].astype("string").str.strip()
    valid = y_raw.notna() & (y_raw != "")
    df, y_raw = df.loc[valid].copy(), y_raw.loc[valid]

    # 'label' is a binary target derived from attack status; exclude it to avoid
    # target leakage when predicting the multiclass 'attack_cat' field.
    X = df.drop(columns=["attack_cat", "label"], errors="ignore")
    X = X.select_dtypes(include=[np.number]).copy()
    if X.empty:
        raise ValueError("No numeric predictor columns were found.")

    # Median imputation is fitted on the training partition only.
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw.astype(str))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns, index=X_test.index)

    # Correlation-based redundancy removal, calculated on training data only.
    corr = X_train.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    redundant = [col for col in upper.columns if (upper[col] > 0.95).any()]
    X_train = X_train.drop(columns=redundant)
    X_test = X_test.drop(columns=redundant)

    # Random-Forest feature-importance ranking, fitted only on training data.
    selector_forest = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1,
        class_weight="balanced_subsample"
    )
    selector_forest.fit(X_train, y_train)
    selector = SelectFromModel(selector_forest, threshold="median", prefit=True)
    selected = X_train.columns[selector.get_support()].tolist()
    if not selected:
        selected = X_train.columns.tolist()
    X_train, X_test = X_train[selected], X_test[selected]

    # SMOTE is applied exclusively to the training partition.
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return (X_train, X_test, y_resampled, y_test, encoder, imputer,
            selected, redundant, X_resampled)


def build_models():
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, class_weight=None,
                                           random_state=RANDOM_STATE))
    ])
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale",
                           class_weight=None, cache_size=1000))
    ])
    rf = RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1,
        class_weight=None
    )
    return {"Logistic Regression": lr, "Support Vector Machine": svm,
            "Random Forest": rf}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Path to UNSW_NB15.csv")
    parser.add_argument("--test-size", type=float, default=0.30)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    (X_train, X_test, y_train, y_test, encoder, imputer,
     selected, redundant, X_resampled) = prepare_data(args.data, args.test_size)
    models = build_models()
    rows, reports, predictions = [], {}, {}
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_resampled, y_train)
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        pred = model.predict(X_test)
        predict_seconds = time.perf_counter() - start
        predictions[name] = pred
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="weighted", zero_division=0
        )
        rows.append({
            "model": name, "accuracy": accuracy_score(y_test, pred),
            "precision_weighted": precision, "recall_weighted": recall,
            "f1_weighted": f1, "training_seconds": train_seconds,
            "prediction_seconds": predict_seconds
        })
        reports[name] = classification_report(
            y_test, pred, labels=np.arange(len(encoder.classes_)),
            target_names=encoder.classes_, output_dict=True, zero_division=0
        )
        safe_name = name.lower().replace(" ", "_")
        pd.DataFrame(confusion_matrix(y_test, pred)).to_csv(
            OUT / f"{safe_name}_confusion_matrix.csv", index=False
        )
        print(f"{name}: accuracy={rows[-1]['accuracy']:.4f}, "
              f"weighted-F1={f1:.4f}, train={train_seconds:.1f}s", flush=True)

    # Hard majority voting over the three already-fitted classifiers. This is
    # equivalent to VotingClassifier(voting='hard') and avoids fitting clones.
    ordered_names = ["Logistic Regression", "Support Vector Machine", "Random Forest"]
    pred_matrix = np.vstack([predictions[name] for name in ordered_names])
    hybrid_pred = np.array([
        np.bincount(pred_matrix[:, i], minlength=len(encoder.classes_)).argmax()
        for i in range(pred_matrix.shape[1])
    ])
    start = time.perf_counter()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, hybrid_pred, average="weighted", zero_division=0
    )
    hybrid_row = {
        "model": "Hybrid Majority Voting", "accuracy": accuracy_score(y_test, hybrid_pred),
        "precision_weighted": precision, "recall_weighted": recall,
        "f1_weighted": f1, "training_seconds": float("nan"),
        "prediction_seconds": time.perf_counter() - start
    }
    rows.append(hybrid_row)
    reports["Hybrid Majority Voting"] = classification_report(
        y_test, hybrid_pred, labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_, output_dict=True, zero_division=0
    )
    pd.DataFrame(confusion_matrix(y_test, hybrid_pred)).to_csv(
        OUT / "hybrid_majority_voting_confusion_matrix.csv", index=False
    )

    pd.DataFrame(rows).to_csv(OUT / "performance_metrics.csv", index=False)
    with open(OUT / "classification_reports.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)
    with open(OUT / "preprocessing_metadata.json", "w", encoding="utf-8") as f:
        json.dump({"selected_features": selected,
                   "removed_correlated_features": redundant,
                   "classes": encoder.classes_.tolist(),
                   "random_state": RANDOM_STATE,
                   "test_size": args.test_size}, f, indent=2)
    print(f"\nClasses: {', '.join(encoder.classes_)}")
    print(f"Selected features ({len(selected)}): {', '.join(selected)}")
    print(f"Removed correlated features ({len(redundant)}): {', '.join(redundant)}")
    print(f"Training rows after SMOTE: {len(y_train)}; test rows: {len(y_test)}")
    print(f"Results saved to: {OUT}")


if __name__ == "__main__":
    main()
