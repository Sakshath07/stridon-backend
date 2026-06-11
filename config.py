import os
from dotenv import load_dotenv

load_dotenv()

MODEL_DIR = "models"
DATA_DIR = "data"
DATASET_PATH = os.path.join(DATA_DIR, "stridon_dataset.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "stridon_model.joblib")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")

FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase_credentials.json")

STREAMS = ["Science", "Commerce", "Arts"]

NUMERICAL_FEATURES = [
    "analytical_score", "creativity_score", "leadership_score",
    "communication_score", "problem_solving_score",
    "interest_science", "interest_mathematics", "interest_arts_culture",
    "interest_business_economics", "interest_social_humanities", "interest_technology",
]

BINARY_FEATURES = [
    "extracurricular_sports", "extracurricular_music_dance",
    "extracurricular_debate_mun", "extracurricular_coding_robotics",
    "extracurricular_ngo_social", "extracurricular_science_club",
]

CATEGORICAL_FEATURES = ["personality", "learning_style"]

ALL_FEATURES = NUMERICAL_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
TARGET = "recommended_stream"
