"""
STRIDON API Routes
"""

import os
import sys
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import (
    BehaviorAnalysisRequest,
    StudentInput, PredictionResponse, TrainRequest, TrainResponse, HealthResponse
)
from ml.predictor import get_predictor
from config import MODEL_PATH, DATASET_PATH, DATA_DIR

router = APIRouter()


# ── Health ─────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Check if the API and model are running."""
    return {
        "status": "healthy",
        "model_loaded": os.path.exists(MODEL_PATH),
        "version": "1.0.0",
    }


# ── Predict ────────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_stream(student: StudentInput):
    """
    Predict the best academic stream for a student and return detailed reasoning.
    
    This is the core STRIDON endpoint. Pass the student's quiz answers and receive:
    - Recommended stream (Science / Commerce / Arts)
    - Confidence score
    - Detailed reasoning (powered by SHAP, no external APIs)
    - Career path recommendations
    - Learning issues and areas to develop
    """
    try:
        predictor = get_predictor()
        features = student.model_dump(exclude={"student_name", "student_id", "class_grade"})
        result = predictor.predict(features)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ── Train ──────────────────────────────────────────────────────────────────────

@router.post("/train", response_model=TrainResponse, tags=["Training"])
def train_model(request: TrainRequest = TrainRequest()):
    """
    Train (or retrain) the STRIDON ML model.

    - If no dataset_path is provided, uses the default synthetic dataset.
    - If you've uploaded your own dataset via /upload-dataset, pass its path here.
    - Returns cross-validation accuracy, per-class metrics, and confusion matrix info.
    """
    from ml.trainer import train

    dataset_path = request.dataset_path or DATASET_PATH

    if not os.path.exists(dataset_path):
        # Auto-generate synthetic dataset if none exists
        from ml.dataset_generator import generate_dataset
        os.makedirs(DATA_DIR, exist_ok=True)
        df = generate_dataset(600)
        df.to_csv(DATASET_PATH, index=False)
        dataset_path = DATASET_PATH

    try:
        metrics = train(dataset_path)

        # Invalidate predictor cache so new model is loaded on next request
        import ml.predictor as pred_module
        pred_module._predictor_instance = None

        return {
            "status": "success",
            **metrics,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


# ── Upload Dataset ─────────────────────────────────────────────────────────────

@router.post("/upload-dataset", tags=["Training"])
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload your own CSV dataset to train the model.

    The CSV must have these columns:
    - personality, learning_style
    - analytical_score, creativity_score, leadership_score, communication_score, problem_solving_score
    - interest_science, interest_mathematics, interest_arts_culture, interest_business_economics,
      interest_social_humanities, interest_technology
    - extracurricular_sports, extracurricular_music_dance, extracurricular_debate_mun,
      extracurricular_coding_robotics, extracurricular_ngo_social, extracurricular_science_club
    - recommended_stream (Science / Commerce / Arts)
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    os.makedirs(DATA_DIR, exist_ok=True)
    upload_path = os.path.join(DATA_DIR, "user_uploaded_dataset.csv")

    content = await file.read()
    with open(upload_path, "wb") as f:
        f.write(content)

    try:
        df = pd.read_csv(upload_path)
        return {
            "status": "uploaded",
            "rows": len(df),
            "columns": list(df.columns),
            "dataset_path": upload_path,
            "next_step": "Call POST /api/v1/train with dataset_path set to this path to train the model.",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {str(e)}")


# ── Augment Dataset ────────────────────────────────────────────────────────────

@router.post("/augment-dataset", tags=["Training"])
async def augment_dataset(file: UploadFile = File(...), n_synthetic: int = 400):
    """
    Upload your small real dataset. STRIDON will augment it with synthetic data
    and save the combined dataset for training.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    from ml.dataset_generator import augment_with_real_data

    content = await file.read()
    import io
    real_df = pd.read_csv(io.BytesIO(content))

    combined = augment_with_real_data(real_df, n_synthetic=n_synthetic)

    os.makedirs(DATA_DIR, exist_ok=True)
    augmented_path = os.path.join(DATA_DIR, "augmented_dataset.csv")
    combined.to_csv(augmented_path, index=False)

    return {
        "status": "augmented",
        "real_rows": len(real_df),
        "synthetic_rows_added": n_synthetic,
        "total_rows": len(combined),
        "dataset_path": augmented_path,
        "next_step": f"Call POST /api/v1/train with dataset_path='{augmented_path}' to train on this combined data.",
    }


# ── Dataset Info ───────────────────────────────────────────────────────────────

@router.get("/dataset-schema", tags=["Training"])
def get_dataset_schema():
    """
    Returns the expected dataset schema with column names, types, and allowed values.
    Use this as reference when building your own dataset.
    """
    return {
        "columns": {
            "personality": {"type": "categorical", "values": ["introvert", "extrovert", "ambivert"]},
            "learning_style": {"type": "categorical", "values": ["visual", "auditory", "kinesthetic", "reading_writing"]},
            "analytical_score": {"type": "integer", "range": "1-5"},
            "creativity_score": {"type": "integer", "range": "1-5"},
            "leadership_score": {"type": "integer", "range": "1-5"},
            "communication_score": {"type": "integer", "range": "1-5"},
            "problem_solving_score": {"type": "integer", "range": "1-5"},
            "interest_science": {"type": "integer", "range": "1-5"},
            "interest_mathematics": {"type": "integer", "range": "1-5"},
            "interest_arts_culture": {"type": "integer", "range": "1-5"},
            "interest_business_economics": {"type": "integer", "range": "1-5"},
            "interest_social_humanities": {"type": "integer", "range": "1-5"},
            "interest_technology": {"type": "integer", "range": "1-5"},
            "extracurricular_sports": {"type": "integer", "range": "0 or 1"},
            "extracurricular_music_dance": {"type": "integer", "range": "0 or 1"},
            "extracurricular_debate_mun": {"type": "integer", "range": "0 or 1"},
            "extracurricular_coding_robotics": {"type": "integer", "range": "0 or 1"},
            "extracurricular_ngo_social": {"type": "integer", "range": "0 or 1"},
            "extracurricular_science_club": {"type": "integer", "range": "0 or 1"},
            "recommended_stream": {"type": "categorical", "values": ["Science", "Commerce", "Arts"]},
        },
        "notes": "recommended_stream is the target label. All other columns are features.",
    }


# ── Behavior Analysis ──────────────────────────────────────────────────────────

@router.post("/analyze-behavior", tags=["Behavior"])
def analyze_behavior(request: BehaviorAnalysisRequest):
    """
    Analyze HOW a student answered questions — not just whether they got them right.

    Captures: time per question, hesitation patterns, answer changes, review behavior.

    Returns:
      - engagement_score    (0–100, ML-driven, replaces rule-based score)
      - mastery_estimate    (0–100, topic mastery beyond raw accuracy)
      - confidence_level    (low / medium / high)
      - cognitive_load      (low / medium / high)
      - attention_stability (stable / moderate / fluctuating)
      - behavioral_insights (natural language observations)
      - adaptive_action     (what to do next)
      - question_analysis   (per-question breakdown)
    """
    try:
        from ml.behavior_model import get_behavior_analyzer
        analyzer = get_behavior_analyzer()
        q_data   = [q.model_dump() for q in request.questions]
        result   = analyzer.analyze(q_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Behavior analysis failed: {str(e)}")


@router.post("/train-behavior", tags=["Behavior"])
def train_behavior_model_endpoint(n_sessions: int = 2000):
    """Retrain the behavioral analysis model. Call after collecting real student data."""
    try:
        from ml.behavior_model import train_behavior_model, _analyzer
        import ml.behavior_model as bm
        metrics = train_behavior_model(n_sessions)
        bm._analyzer = None  # reset singleton
        return {"status": "success", **metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
