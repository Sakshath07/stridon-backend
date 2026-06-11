"""
STRIDON Model Trainer
Trains the stream recommendation model from a dataset CSV.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODEL_DIR, DATA_DIR, DATASET_PATH, MODEL_PATH,
    PREPROCESSOR_PATH, LABEL_ENCODER_PATH,
    NUMERICAL_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES, TARGET,
)


def build_preprocessor() -> ColumnTransformer:
    """Build the feature preprocessing pipeline."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_model() -> VotingClassifier:
    """Build an ensemble classifier (RF + GBM for robustness on small datasets)."""
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )
    gbm = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    return VotingClassifier(
        estimators=[("rf", rf), ("gbm", gbm)],
        voting="soft",
        weights=[2, 1],       # RF gets higher weight — more interpretable
    )


def train(dataset_path: str = DATASET_PATH) -> dict:
    """
    Full training pipeline.

    Args:
        dataset_path: Path to CSV dataset with all required columns.

    Returns:
        dict with training metrics.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Load data ──────────────────────────────────────────
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Run: python ml/dataset_generator.py   to generate a synthetic dataset, "
            "or upload your own CSV with the required columns."
        )

    df = pd.read_csv(dataset_path)
    print(f"📂 Loaded dataset: {len(df)} rows, {df.shape[1]} columns")

    # Validate columns
    all_required = NUMERICAL_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing = [c for c in all_required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    X = df[NUMERICAL_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    # ── Label encode target ────────────────────────────────
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"🏷️  Classes: {list(le.classes_)}")
    print(f"📊 Distribution:\n{pd.Series(y).value_counts().to_string()}")

    # ── Build full pipeline ────────────────────────────────
    preprocessor = build_preprocessor()
    classifier = build_model()

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    # ── Cross-validation ───────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y_enc, cv=cv, scoring="accuracy")
    print(f"\n🔁 5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Train/test split ───────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print("\n📈 Test Set Results:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion Matrix:")
    print(cm)

    # ── Save artefacts ─────────────────────────────────────
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"\n✅ Model saved → {MODEL_PATH}")

    return {
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "test_accuracy": float(report["accuracy"]),
        "per_class_metrics": {
            cls: {
                "precision": float(report[cls]["precision"]),
                "recall": float(report[cls]["recall"]),
                "f1": float(report[cls]["f1-score"]),
            }
            for cls in le.classes_
        },
        "classes": list(le.classes_),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


if __name__ == "__main__":
    metrics = train()
    print("\n🎯 Training complete. Metrics:", metrics)
