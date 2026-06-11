"""
STRIDON Dataset Generator
Generates synthetic student data for training the ML model.
You can replace / augment this with your real dataset.
"""

import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_PATH, DATA_DIR


def generate_student_profile(stream: str, rng: np.random.Generator) -> dict:
    """Generate a single student profile biased toward a given stream."""

    def score(low, high):
        return int(np.clip(rng.normal((low + high) / 2, 0.8), 1, 5))

    def prob(p):
        return int(rng.random() < p)

    personalities = ["introvert", "extrovert", "ambivert"]
    learning_styles = ["visual", "auditory", "kinesthetic", "reading_writing"]

    if stream == "Science":
        return {
            "personality": rng.choice(personalities, p=[0.40, 0.25, 0.35]),
            "learning_style": rng.choice(learning_styles, p=[0.30, 0.20, 0.30, 0.20]),
            "analytical_score": score(3, 5),
            "creativity_score": score(2, 4),
            "leadership_score": score(2, 4),
            "communication_score": score(2, 4),
            "problem_solving_score": score(3, 5),
            "interest_science": score(3, 5),
            "interest_mathematics": score(3, 5),
            "interest_arts_culture": score(1, 3),
            "interest_business_economics": score(1, 3),
            "interest_social_humanities": score(1, 3),
            "interest_technology": score(3, 5),
            "extracurricular_sports": prob(0.35),
            "extracurricular_music_dance": prob(0.15),
            "extracurricular_debate_mun": prob(0.20),
            "extracurricular_coding_robotics": prob(0.55),
            "extracurricular_ngo_social": prob(0.10),
            "extracurricular_science_club": prob(0.60),
            "recommended_stream": "Science",
        }

    elif stream == "Commerce":
        return {
            "personality": rng.choice(personalities, p=[0.20, 0.50, 0.30]),
            "learning_style": rng.choice(learning_styles, p=[0.25, 0.30, 0.20, 0.25]),
            "analytical_score": score(2, 4),
            "creativity_score": score(2, 4),
            "leadership_score": score(3, 5),
            "communication_score": score(3, 5),
            "problem_solving_score": score(2, 4),
            "interest_science": score(1, 3),
            "interest_mathematics": score(2, 4),
            "interest_arts_culture": score(1, 3),
            "interest_business_economics": score(3, 5),
            "interest_social_humanities": score(2, 4),
            "interest_technology": score(2, 4),
            "extracurricular_sports": prob(0.40),
            "extracurricular_music_dance": prob(0.20),
            "extracurricular_debate_mun": prob(0.60),
            "extracurricular_coding_robotics": prob(0.20),
            "extracurricular_ngo_social": prob(0.35),
            "extracurricular_science_club": prob(0.10),
            "recommended_stream": "Commerce",
        }

    else:  # Arts
        return {
            "personality": rng.choice(personalities, p=[0.30, 0.35, 0.35]),
            "learning_style": rng.choice(learning_styles, p=[0.35, 0.25, 0.25, 0.15]),
            "analytical_score": score(1, 3),
            "creativity_score": score(3, 5),
            "leadership_score": score(2, 4),
            "communication_score": score(3, 5),
            "problem_solving_score": score(1, 3),
            "interest_science": score(1, 2),
            "interest_mathematics": score(1, 2),
            "interest_arts_culture": score(3, 5),
            "interest_business_economics": score(1, 3),
            "interest_social_humanities": score(3, 5),
            "interest_technology": score(1, 3),
            "extracurricular_sports": prob(0.30),
            "extracurricular_music_dance": prob(0.65),
            "extracurricular_debate_mun": prob(0.30),
            "extracurricular_coding_robotics": prob(0.10),
            "extracurricular_ngo_social": prob(0.50),
            "extracurricular_science_club": prob(0.05),
            "recommended_stream": "Arts",
        }


def generate_dataset(n_samples: int = 600, seed: int = 42) -> pd.DataFrame:
    """Generate a balanced synthetic dataset."""
    rng = np.random.default_rng(seed)
    per_stream = n_samples // 3
    records = []

    for stream in ["Science", "Commerce", "Arts"]:
        for _ in range(per_stream):
            records.append(generate_student_profile(stream, rng))

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def augment_with_real_data(real_df: pd.DataFrame, n_synthetic: int = 400, seed: int = 99) -> pd.DataFrame:
    """
    Augment a small real dataset with synthetic samples.
    Use this when you have your own data but need more samples.

    Args:
        real_df: Your real dataset (must have all required columns)
        n_synthetic: Number of synthetic samples to add
        seed: Random seed

    Returns:
        Combined dataframe
    """
    synthetic = generate_dataset(n_synthetic, seed)
    combined = pd.concat([real_df, synthetic], ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)
    return combined


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    df = generate_dataset(600)
    df.to_csv(DATASET_PATH, index=False)
    print(f"✅ Dataset generated: {len(df)} records → {DATASET_PATH}")
    print(df["recommended_stream"].value_counts())
    print(df.head(3))
