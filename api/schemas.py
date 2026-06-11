from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal


# ─── Input ────────────────────────────────────────────────────────────────────

class StudentInput(BaseModel):
    """
    Full student profile collected from the STRIDON quiz.
    All scores are on a 1–5 scale. Extracurriculars are 0 or 1.
    """

    # Personality & learning
    personality: Literal["introvert", "extrovert", "ambivert"] = Field(
        ..., description="Student's personality type"
    )
    learning_style: Literal["visual", "auditory", "kinesthetic", "reading_writing"] = Field(
        ..., description="Preferred learning style"
    )

    # Soft-skill scores (1–5)
    analytical_score: int = Field(..., ge=1, le=5, description="Analytical thinking score")
    creativity_score: int = Field(..., ge=1, le=5, description="Creativity score")
    leadership_score: int = Field(..., ge=1, le=5, description="Leadership score")
    communication_score: int = Field(..., ge=1, le=5, description="Communication score")
    problem_solving_score: int = Field(..., ge=1, le=5, description="Problem-solving score")

    # Interests (1–5)
    interest_science: int = Field(..., ge=1, le=5, description="Interest in Science")
    interest_mathematics: int = Field(..., ge=1, le=5, description="Interest in Mathematics")
    interest_arts_culture: int = Field(..., ge=1, le=5, description="Interest in Arts & Culture")
    interest_business_economics: int = Field(..., ge=1, le=5, description="Interest in Business/Economics")
    interest_social_humanities: int = Field(..., ge=1, le=5, description="Interest in Social Sciences/Humanities")
    interest_technology: int = Field(..., ge=1, le=5, description="Interest in Technology")

    # Extracurriculars (0 or 1)
    extracurricular_sports: int = Field(0, ge=0, le=1, description="Participates in sports")
    extracurricular_music_dance: int = Field(0, ge=0, le=1, description="Participates in music/dance")
    extracurricular_debate_mun: int = Field(0, ge=0, le=1, description="Participates in debate/MUN")
    extracurricular_coding_robotics: int = Field(0, ge=0, le=1, description="Does coding/robotics")
    extracurricular_ngo_social: int = Field(0, ge=0, le=1, description="Involved in NGO/social service")
    extracurricular_science_club: int = Field(0, ge=0, le=1, description="Member of science club")

    # Optional metadata
    student_name: Optional[str] = Field(None, description="Student's name (optional)")
    student_id: Optional[str] = Field(None, description="Firebase user ID (optional)")
    class_grade: Optional[int] = Field(None, ge=9, le=10, description="Class: 9 or 10")


# ─── Sub-response models ──────────────────────────────────────────────────────

class ReasoningOutput(BaseModel):
    summary: str
    stream_overview: str
    personality_insight: str
    key_strengths: list[str]
    why_this_stream: str
    career_paths: list[str]
    areas_to_develop: list[str]
    negative_factors: list[str]
    alternative_streams: str
    full_report: str


class FeatureImportanceItem(BaseModel):
    feature: str
    shap_value: float
    actual_value: float | int | str
    impact: Literal["positive", "negative"]


# ─── Main prediction response ─────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    recommended_stream: Literal["Science", "Commerce", "Arts"]
    confidence: float
    all_probabilities: dict[str, float]
    reasoning: ReasoningOutput
    feature_importance: list[FeatureImportanceItem]


# ─── Training ────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    dataset_path: Optional[str] = Field(
        None,
        description="Path to custom CSV dataset. Leave empty to use default dataset.",
    )


class PerClassMetrics(BaseModel):
    precision: float
    recall: float
    f1: float


class TrainResponse(BaseModel):
    status: str
    cv_accuracy_mean: float
    cv_accuracy_std: float
    test_accuracy: float
    per_class_metrics: dict[str, PerClassMetrics]
    classes: list[str]
    n_train: int
    n_test: int


# ─── Health check ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    version: str


# ─── Behavior Analysis ────────────────────────────────────────────────────────

class QuestionBehavior(BaseModel):
    """Behavioral signals for a single question."""
    time_spent:            float = Field(..., ge=0, description="Total seconds on this question")
    time_to_first_answer:  float = Field(..., ge=0, description="Seconds before first option selected")
    review_time:           float = Field(0.0, ge=0, description="Seconds after last change before submitting")
    n_changes:             int   = Field(0, ge=0, description="Number of answer changes")
    changed_back:          int   = Field(0, ge=0, le=1, description="Returned to a previously rejected option")
    is_correct:            Optional[int] = Field(None, description="Final answer was correct (0/1)")
    difficulty:            int   = Field(1, ge=0, le=2, description="0=easy, 1=medium, 2=hard")
    question_position:     int   = Field(1, ge=1, description="Position in test (1-indexed)")


class BehaviorAnalysisRequest(BaseModel):
    """Full session behavioral data to analyze."""
    questions:   list[QuestionBehavior] = Field(..., min_length=1)
    subject:     Optional[str] = Field(None, description="Subject being tested")
    student_id:  Optional[str] = Field(None, description="Firebase user ID")


class AdaptiveAction(BaseModel):
    next_difficulty: str
    action:          str
    icon:            str


class QuestionAnalysisItem(BaseModel):
    question_index:    int
    struggle_detected: bool
    rushed:            bool
    confidence:        str
    time_spent:        float
    n_changes:         int
    is_correct:        Optional[int]


class BehaviorAnalysisResponse(BaseModel):
    engagement_score:     int
    mastery_estimate:     int
    confidence_level:     str
    cognitive_load:       str
    attention_stability:  str
    behavioral_insights:  list[str]
    adaptive_action:      AdaptiveAction
    session_features:     dict[str, float]
    question_analysis:    list[QuestionAnalysisItem]
