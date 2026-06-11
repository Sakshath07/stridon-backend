"""
STRIDON Predictor
Loads the trained model, runs prediction, extracts SHAP values,
and calls the reasoning engine to produce a full student report.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import shap

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODEL_PATH, PREPROCESSOR_PATH, LABEL_ENCODER_PATH,
    NUMERICAL_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES,
)
from ml.reasoning_engine import build_reasoning

ALL_FEATURES = NUMERICAL_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


class STRIDONPredictor:
    """Singleton-style predictor. Load once, predict many times."""

    def __init__(self):
        self._pipeline = None
        self._label_encoder = None
        self._explainer = None
        self._feature_names: list[str] = []

    def _ensure_loaded(self):
        if self._pipeline is not None:
            return
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                "Model not found. Please train the model first by calling POST /api/v1/train "
                "or running: python ml/trainer.py"
            )
        self._pipeline = joblib.load(MODEL_PATH)
        self._label_encoder = joblib.load(LABEL_ENCODER_PATH)

        # Build SHAP explainer on the Random Forest inside the VotingClassifier
        # We use a small background dataset of zeros as reference
        rf_model = self._pipeline.named_steps["classifier"].estimators_[0]
        preprocessor = self._pipeline.named_steps["preprocessor"]

        # Background data for SHAP (use median-ish values)
        bg = pd.DataFrame(
            [[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, "ambivert", "visual"]],
            columns=ALL_FEATURES,
        )
        bg_transformed = preprocessor.transform(bg)
        self._explainer = shap.TreeExplainer(rf_model, data=bg_transformed, feature_perturbation="interventional")
        self._feature_names = ALL_FEATURES
        print("✅ STRIDON predictor loaded with SHAP explainer.")

    def _df_from_input(self, features: dict) -> pd.DataFrame:
        row = {f: features.get(f, 0) for f in ALL_FEATURES}
        return pd.DataFrame([row])

    def predict(self, features: dict) -> dict:
        """
        Run full prediction + SHAP reasoning for a student.

        Args:
            features: dict with all student feature values from the quiz.

        Returns:
            Full prediction result including recommendation, confidence,
            SHAP values, career paths, issues, and detailed reasoning.
        """
        self._ensure_loaded()

        df = self._df_from_input(features)

        # ── Model prediction ───────────────────────────────
        probs = self._pipeline.predict_proba(df)[0]
        classes = self._label_encoder.classes_
        predicted_idx = int(np.argmax(probs))
        predicted_stream = classes[predicted_idx]
        confidence = float(probs[predicted_idx])
        all_probs = {cls: float(p) for cls, p in zip(classes, probs)}

        # ── SHAP explanation ───────────────────────────────
        preprocessor = self._pipeline.named_steps["preprocessor"]
        rf_model = self._pipeline.named_steps["classifier"].estimators_[0]

        X_transformed = preprocessor.transform(df)
        shap_vals = self._explainer.shap_values(X_transformed, check_additivity=False)  # shape: [n_classes, 1, n_features]

        # Use SHAP values for the predicted class
        # shap_vals shape varies: list of arrays [n_classes][n_samples, n_features] or [n_samples, n_features, n_classes]
        if isinstance(shap_vals, list):
            # list of arrays: one per class, each shape (n_samples, n_features)
            class_shap = np.array(shap_vals[predicted_idx][0]).flatten()
        elif shap_vals.ndim == 3:
            # shape (n_samples, n_features, n_classes)
            class_shap = shap_vals[0, :, predicted_idx].flatten()
        else:
            # shape (n_samples, n_features)
            class_shap = shap_vals[0].flatten()

        # Build (feature, shap_value, actual_value) tuples
        shap_tuples = [
            (self._feature_names[i], float(class_shap[i]), features.get(self._feature_names[i], 0))
            for i in range(len(self._feature_names))
        ]

        # ── Reasoning ─────────────────────────────────────
        reasoning = build_reasoning(
            stream=predicted_stream,
            confidence=confidence,
            features=features,
            shap_values=shap_tuples,
            all_probabilities=all_probs,
        )

        # Top feature importances (for UI display)
        top_features = sorted(shap_tuples, key=lambda x: abs(x[1]), reverse=True)[:8]
        feature_importance_display = [
            {
                "feature": feat,
                "shap_value": round(sv, 4),
                "actual_value": av,
                "impact": "positive" if sv > 0 else "negative",
            }
            for feat, sv, av in top_features
        ]

        return {
            "recommended_stream": predicted_stream,
            "confidence": round(confidence, 4),
            "all_probabilities": {k: round(v, 4) for k, v in all_probs.items()},
            "reasoning": reasoning,
            "feature_importance": feature_importance_display,
        }


# Global singleton
_predictor_instance: STRIDONPredictor | None = None


def get_predictor() -> STRIDONPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = STRIDONPredictor()
    return _predictor_instance
