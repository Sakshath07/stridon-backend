"""
STRIDON Behavioral Analysis Model
─────────────────────────────────────────────────────────────────────────────
Analyzes HOW a student answers questions, not just WHETHER they got it right.

Behavioral signals captured:
  • time_spent           – total seconds on a question
  • time_to_first_answer – seconds before first option selected (hesitation)
  • review_time          – seconds spent after last change (confidence check)
  • n_changes            – how many times the answer was changed
  • changed_back         – returned to a previously rejected option (0/1)
  • is_correct           – final answer was correct (0/1)
  • difficulty           – question difficulty (0=easy, 1=medium, 2=hard)
  • question_position    – position in test (fatigue / warm-up effects)

Model outputs per session:
  • engagement_score     – 0–100 (replaces simple rule-based score)
  • confidence_level     – "low" / "medium" / "high"
  • cognitive_load       – "low" / "medium" / "high"
  • attention_stability  – "stable" / "moderate" / "fluctuating"
  • mastery_estimate     – 0–100 (topic mastery beyond raw score)
  • behavioral_insights  – natural-language observations
  • adaptive_action      – next recommended study action
"""

import os, sys, joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_DIR           = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
BEHAVIOR_MODEL_PATH = os.path.join(MODEL_DIR, "behavior_model.joblib")
BEHAVIOR_ENCODERS_PATH = os.path.join(MODEL_DIR, "behavior_encoders.joblib")

# ─── Feature names ─────────────────────────────────────────────────────────────
SESSION_FEATURES = [
    # Per-question aggregates
    "avg_time",              "time_std",            "min_time",           "max_time",
    "avg_time_to_first",     "avg_review_time",
    "avg_changes",           "total_changes",
    "pct_changed",           "pct_changed_back",
    "pct_rush",              "pct_overthought",
    # Accuracy signals
    "accuracy",              "avg_difficulty",
    # Derived behavioral signals
    "hesitation_index",      "confidence_index",    "attention_consistency",
    "first_instinct_rate",   "metacognition_score",
]

# ─── Dataset generator ─────────────────────────────────────────────────────────

def _q(rng, pattern: str, position: int, n_questions: int) -> dict:
    """Generate one question's behavioral signals for a given student pattern."""
    fatigue = position / n_questions          # 0→1 as test progresses

    if pattern == "confident_expert":
        time_to_first  = rng.normal(10, 3)
        review_time    = rng.normal(4, 2)
        n_changes      = rng.choice([0, 1], p=[0.80, 0.20])
        changed_back   = 0
        is_correct     = int(rng.random() < 0.90 - fatigue * 0.05)
        difficulty     = rng.choice([0, 1, 2], p=[0.25, 0.45, 0.30])
        time_spent     = time_to_first + n_changes * rng.uniform(5, 15) + review_time

    elif pattern == "careful_learner":
        time_to_first  = rng.normal(20, 8)
        review_time    = rng.normal(10, 5)
        n_changes      = rng.choice([0, 1, 2], p=[0.40, 0.40, 0.20])
        changed_back   = int(rng.random() < 0.10 and n_changes >= 2)
        is_correct     = int(rng.random() < 0.70 - fatigue * 0.08)
        difficulty     = rng.choice([0, 1, 2], p=[0.30, 0.50, 0.20])
        time_spent     = time_to_first + n_changes * rng.uniform(10, 20) + review_time

    elif pattern == "struggling":
        time_to_first  = rng.normal(35, 15)
        review_time    = rng.normal(20, 10)
        n_changes      = rng.choice([1, 2, 3, 4], p=[0.20, 0.30, 0.30, 0.20])
        changed_back   = int(rng.random() < 0.35 and n_changes >= 2)
        is_correct     = int(rng.random() < 0.42 - fatigue * 0.10)
        difficulty     = rng.choice([0, 1, 2], p=[0.40, 0.40, 0.20])
        time_spent     = time_to_first + n_changes * rng.uniform(12, 25) + review_time

    elif pattern == "rushing":
        time_to_first  = rng.normal(5, 2)
        review_time    = rng.normal(1, 1)
        n_changes      = rng.choice([0, 1], p=[0.85, 0.15])
        changed_back   = 0
        is_correct     = int(rng.random() < 0.38)   # near-random
        difficulty     = rng.choice([0, 1, 2], p=[0.33, 0.34, 0.33])
        time_spent     = time_to_first + n_changes * rng.uniform(2, 8) + review_time

    elif pattern == "anxious":
        time_to_first  = rng.normal(18, 12)
        review_time    = rng.normal(15, 10)
        n_changes      = rng.choice([2, 3, 4, 5], p=[0.25, 0.30, 0.30, 0.15])
        changed_back   = int(rng.random() < 0.55 and n_changes >= 2)
        is_correct     = int(rng.random() < 0.55 - fatigue * 0.12)
        difficulty     = rng.choice([0, 1, 2], p=[0.33, 0.34, 0.33])
        time_spent     = time_to_first + n_changes * rng.uniform(8, 20) + review_time

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return {
        "time_spent":          max(3.0, float(time_spent)),
        "time_to_first_answer":max(1.0, float(time_to_first)),
        "review_time":         max(0.0, float(review_time)),
        "n_changes":           int(max(0, n_changes)),
        "changed_back":        int(changed_back),
        "is_correct":          int(is_correct),
        "difficulty":          int(difficulty),
        "question_position":   position,
    }


def _session_features(questions: list[dict]) -> dict:
    """Aggregate per-question signals into session-level features."""
    n = len(questions)
    times          = [q["time_spent"]           for q in questions]
    firsts         = [q["time_to_first_answer"]  for q in questions]
    reviews        = [q["review_time"]           for q in questions]
    changes        = [q["n_changes"]             for q in questions]
    correct        = [q["is_correct"]            for q in questions]
    difficulties   = [q["difficulty"]            for q in questions]

    avg_time       = np.mean(times)
    time_std       = np.std(times)

    # Rush = answered within 10 seconds total
    pct_rush       = sum(1 for t in times if t < 10) / n
    # Overthought = spent more than 90 seconds
    pct_overthought= sum(1 for t in times if t > 90) / n

    avg_changes    = np.mean(changes)
    pct_changed    = sum(1 for c in changes if c > 0) / n
    pct_changed_back = sum(q["changed_back"] for q in questions) / n

    # Hesitation index: average time before first answer relative to total
    hesitation_index = np.mean([f / t for f, t in zip(firsts, times) if t > 0])
    # Confidence index: how quickly they commit after last change
    confidence_index = 1.0 - np.mean([r / t for r, t in zip(reviews, times) if t > 0])
    # Attention consistency: stable timing = high attention
    attention_consistency = 1.0 / (1.0 + time_std / (avg_time + 1e-6))
    # First instinct: got it right with 0 changes
    first_instinct_rate = sum(1 for q in questions if q["n_changes"] == 0 and q["is_correct"]) / n
    # Metacognition: changed answer AND got it right (knows when wrong)
    metacognition_score = sum(
        1 for q in questions if q["n_changes"] > 0 and q["is_correct"]
    ) / max(1, sum(1 for q in questions if q["n_changes"] > 0))

    return {
        "avg_time":              avg_time,
        "time_std":              time_std,
        "min_time":              np.min(times),
        "max_time":              np.max(times),
        "avg_time_to_first":     np.mean(firsts),
        "avg_review_time":       np.mean(reviews),
        "avg_changes":           avg_changes,
        "total_changes":         sum(changes),
        "pct_changed":           pct_changed,
        "pct_changed_back":      pct_changed_back,
        "pct_rush":              pct_rush,
        "pct_overthought":       pct_overthought,
        "accuracy":              np.mean(correct),
        "avg_difficulty":        np.mean(difficulties),
        "hesitation_index":      float(np.clip(hesitation_index, 0, 1)),
        "confidence_index":      float(np.clip(confidence_index, 0, 1)),
        "attention_consistency": float(np.clip(attention_consistency, 0, 1)),
        "first_instinct_rate":   first_instinct_rate,
        "metacognition_score":   metacognition_score,
    }


def _labels_for_pattern(pattern: str, features: dict) -> dict:
    """Assign ground-truth labels based on pattern + computed features."""
    acc  = features["accuracy"]
    rush = features["pct_rush"]
    oth  = features["pct_overthought"]
    att  = features["attention_consistency"]
    chg  = features["pct_changed_back"]

    # Engagement score (0–100)
    if pattern == "confident_expert":
        eng = int(np.clip(80 + acc * 20 - rush * 10, 70, 100))
    elif pattern == "careful_learner":
        eng = int(np.clip(55 + acc * 30 - chg * 15, 45, 85))
    elif pattern == "struggling":
        eng = int(np.clip(30 + acc * 20 - oth * 10, 20, 60))
    elif pattern == "rushing":
        eng = int(np.clip(25 - rush * 15 + acc * 10, 10, 45))
    else:  # anxious
        eng = int(np.clip(40 + acc * 15 - chg * 20, 25, 70))

    # Confidence level
    if pattern in ("confident_expert",) or (acc > 0.75 and features["avg_changes"] < 1):
        conf = "high"
    elif pattern in ("struggling", "anxious") or (acc < 0.45 and features["pct_changed_back"] > 0.3):
        conf = "low"
    else:
        conf = "medium"

    # Cognitive load
    if pattern == "rushing" or (features["avg_time"] < 12 and features["avg_changes"] < 0.5):
        load = "low"
    elif pattern in ("struggling", "anxious") or features["avg_changes"] > 2 or features["pct_overthought"] > 0.3:
        load = "high"
    else:
        load = "medium"

    # Attention stability
    if att > 0.70:
        stability = "stable"
    elif att > 0.45:
        stability = "moderate"
    else:
        stability = "fluctuating"

    # Mastery estimate (0–100): performance + behavioral quality
    mastery = int(np.clip(
        acc * 60 + features["first_instinct_rate"] * 20 + att * 20, 0, 100
    ))

    return {
        "engagement_score": eng,
        "confidence_level": conf,
        "cognitive_load":   load,
        "attention_stability": stability,
        "mastery_estimate": mastery,
    }


def generate_behavior_dataset(n_sessions: int = 2000, n_questions: int = 10, seed: int = 42) -> pd.DataFrame:
    rng      = np.random.default_rng(seed)
    patterns = ["confident_expert", "careful_learner", "struggling", "rushing", "anxious"]
    rows     = []
    for i in range(n_sessions):
        pattern   = patterns[i % len(patterns)]
        questions = [_q(rng, pattern, pos + 1, n_questions) for pos in range(n_questions)]
        features  = _session_features(questions)
        labels    = _labels_for_pattern(pattern, features)
        rows.append({**features, **labels})
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


# ─── Training ──────────────────────────────────────────────────────────────────

def train_behavior_model(n_sessions: int = 2000) -> dict:
    os.makedirs(MODEL_DIR, exist_ok=True)
    df   = generate_behavior_dataset(n_sessions)

    X    = df[SESSION_FEATURES].values
    y_eng  = df["engagement_score"].values
    y_mas  = df["mastery_estimate"].values

    # Encode classification targets
    le_conf  = LabelEncoder(); y_conf  = le_conf.fit_transform(df["confidence_level"])
    le_load  = LabelEncoder(); y_load  = le_load.fit_transform(df["cognitive_load"])
    le_stab  = LabelEncoder(); y_stab  = le_stab.fit_transform(df["attention_stability"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train / test split
    X_tr, X_te, *ys = train_test_split(
        X_scaled,
        y_eng, y_mas, y_conf, y_load, y_stab,
        test_size=0.2, random_state=42
    )
    y_eng_tr,  y_eng_te  = ys[0],  ys[1]
    y_mas_tr,  y_mas_te  = ys[2],  ys[3]
    y_conf_tr, y_conf_te = ys[4],  ys[5]
    y_load_tr, y_load_te = ys[6],  ys[7]
    y_stab_tr, y_stab_te = ys[8],  ys[9]

    params_reg = dict(n_estimators=200, max_depth=5, learning_rate=0.06, subsample=0.8, random_state=42)
    params_clf = dict(n_estimators=200, max_depth=4, learning_rate=0.08, subsample=0.8, random_state=42)

    reg_eng  = GradientBoostingRegressor(**params_reg).fit(X_tr, y_eng_tr)
    reg_mas  = GradientBoostingRegressor(**params_reg).fit(X_tr, y_mas_tr)
    clf_conf = GradientBoostingClassifier(**params_clf).fit(X_tr, y_conf_tr)
    clf_load = GradientBoostingClassifier(**params_clf).fit(X_tr, y_load_tr)
    clf_stab = GradientBoostingClassifier(**params_clf).fit(X_tr, y_stab_tr)

    # Metrics
    from sklearn.metrics import mean_absolute_error, accuracy_score
    metrics = {
        "engagement_mae":        round(mean_absolute_error(y_eng_te,  reg_eng.predict(X_te)),  2),
        "mastery_mae":           round(mean_absolute_error(y_mas_te,  reg_mas.predict(X_te)),  2),
        "confidence_accuracy":   round(accuracy_score(y_conf_te, clf_conf.predict(X_te)),      4),
        "cognitive_load_accuracy": round(accuracy_score(y_load_te, clf_load.predict(X_te)),    4),
        "attention_accuracy":    round(accuracy_score(y_stab_te, clf_stab.predict(X_te)),      4),
    }

    bundle = {
        "scaler":    scaler,
        "reg_eng":   reg_eng,
        "reg_mas":   reg_mas,
        "clf_conf":  clf_conf,
        "clf_load":  clf_load,
        "clf_stab":  clf_stab,
        "le_conf":   le_conf,
        "le_load":   le_load,
        "le_stab":   le_stab,
        "features":  SESSION_FEATURES,
    }
    joblib.dump(bundle, BEHAVIOR_MODEL_PATH)
    print(f"✅ Behavior model saved → {BEHAVIOR_MODEL_PATH}")
    print(f"   Engagement MAE : {metrics['engagement_mae']} points")
    print(f"   Mastery MAE    : {metrics['mastery_mae']} points")
    print(f"   Confidence Acc : {metrics['confidence_accuracy']*100:.1f}%")
    print(f"   Cognitive Acc  : {metrics['cognitive_load_accuracy']*100:.1f}%")
    print(f"   Attention Acc  : {metrics['attention_accuracy']*100:.1f}%")
    return metrics


# ─── Insight generator ─────────────────────────────────────────────────────────

def _generate_insights(features: dict, confidence: str, load: str, stability: str, mastery: int) -> list[str]:
    insights = []

    # Time behaviour
    if features["pct_rush"] > 0.4:
        insights.append(f"You answered {int(features['pct_rush']*100)}% of questions in under 10 seconds — this suggests rushing or disengagement. Slowing down typically improves accuracy significantly.")
    elif features["pct_overthought"] > 0.3:
        insights.append(f"You spent over 90 seconds on {int(features['pct_overthought']*100)}% of questions. This indicates high cognitive effort — consider whether time management needs work.")
    elif 20 < features["avg_time"] < 55:
        insights.append(f"Your average question time of {features['avg_time']:.0f}s is well within the optimal 20–60 second range, showing balanced thinking.")

    # Answer changing
    if features["pct_changed"] > 0.6 and features["pct_changed_back"] > 0.3:
        insights.append("You frequently changed answers and often reverted to earlier choices — a classic sign of decision anxiety. Trust your first instinct more; it's right more often than it feels.")
    elif features["pct_changed"] > 0.5 and features["metacognition_score"] > 0.6:
        insights.append("When you changed answers, you usually got it right — excellent metacognitive awareness. You genuinely know when something doesn't feel right.")
    elif features["avg_changes"] < 0.3 and features["accuracy"] > 0.7:
        insights.append("You rarely changed answers and maintained strong accuracy — a hallmark of genuine topic mastery and confident recall.")

    # Attention
    if stability == "fluctuating":
        insights.append(f"Your response times varied significantly (std: {features['time_std']:.0f}s) — indicating attention fluctuations during the test. A consistent environment without distractions will improve this.")
    elif stability == "stable":
        insights.append("Your response times were highly consistent throughout the test, showing sustained attention and good focus control.")

    # First instinct
    if features["first_instinct_rate"] > 0.6:
        insights.append(f"You got {int(features['first_instinct_rate']*100)}% of questions right on the first try without any changes — strong topic foundation.")

    # Confidence calibration
    if confidence == "low" and features["accuracy"] > 0.6:
        insights.append("You're underconfident — you doubted yourself more than the results warranted. Your knowledge is stronger than you think.")
    elif confidence == "high" and features["accuracy"] < 0.5:
        insights.append("You showed high confidence but lower accuracy — a sign of overconfidence in this topic. More structured revision will recalibrate this.")

    return insights[:4]  # max 4 insights


def _adaptive_action(confidence: str, load: str, mastery: int, accuracy: float) -> dict:
    if mastery >= 75 and confidence == "high":
        return {
            "next_difficulty": "harder",
            "action":          "Challenge yourself with harder questions and past exam papers for this topic.",
            "icon":            "🚀",
        }
    elif mastery >= 50 and confidence == "medium":
        return {
            "next_difficulty": "same",
            "action":          "Consolidate understanding with varied practice. Focus on the types of questions where you changed answers.",
            "icon":            "📈",
        }
    elif load == "high" or confidence == "low":
        return {
            "next_difficulty": "easier",
            "action":          "Rebuild confidence with easier questions and ensure fundamentals are solid before progressing.",
            "icon":            "🔧",
        }
    elif accuracy < 0.45:
        return {
            "next_difficulty": "easier",
            "action":          "Review the core concepts of this topic with your mentor before attempting more tests.",
            "icon":            "📖",
        }
    else:
        return {
            "next_difficulty": "same",
            "action":          "Maintain consistent practice. Aim for 3 sessions per week on this topic.",
            "icon":            "✅",
        }


# ─── Predictor ─────────────────────────────────────────────────────────────────

class BehaviorAnalyzer:
    def __init__(self):
        self._bundle = None

    def _load(self):
        if self._bundle: return
        if not os.path.exists(BEHAVIOR_MODEL_PATH):
            print("⚙️  Behavior model not found — training now...")
            train_behavior_model()
        self._bundle = joblib.load(BEHAVIOR_MODEL_PATH)

    def analyze(self, question_data: list[dict]) -> dict:
        """
        Analyze behavioral signals from a completed test session.

        Args:
            question_data: list of dicts, one per question, each with keys:
                time_spent, time_to_first_answer, review_time, n_changes,
                changed_back, is_correct, difficulty, question_position

        Returns:
            Full behavioral analysis dict.
        """
        self._load()
        b = self._bundle

        # Aggregate features
        feat = _session_features(question_data)
        X    = np.array([[feat[f] for f in SESSION_FEATURES]])
        X_sc = b["scaler"].transform(X)

        eng_score  = int(np.clip(round(b["reg_eng"].predict(X_sc)[0]), 0, 100))
        mas_score  = int(np.clip(round(b["reg_mas"].predict(X_sc)[0]), 0, 100))
        confidence = b["le_conf"].inverse_transform(b["clf_conf"].predict(X_sc))[0]
        load       = b["le_load"].inverse_transform(b["clf_load"].predict(X_sc))[0]
        stability  = b["le_stab"].inverse_transform(b["clf_stab"].predict(X_sc))[0]

        insights = _generate_insights(feat, confidence, load, stability, mas_score)
        action   = _adaptive_action(confidence, load, mas_score, feat["accuracy"])

        # Per-question analysis
        q_analysis = []
        for i, q in enumerate(question_data):
            rush     = q["time_spent"] < 10
            struggle = q["time_spent"] > 90 or (q["n_changes"] >= 3 and q["changed_back"])
            q_analysis.append({
                "question_index":    i + 1,
                "struggle_detected": struggle,
                "rushed":            rush,
                "confidence":        "low" if q["n_changes"] >= 3 else "high" if q["n_changes"] == 0 else "medium",
                "time_spent":        q["time_spent"],
                "n_changes":         q["n_changes"],
                "is_correct":        q.get("is_correct", None),
            })

        return {
            "engagement_score":      eng_score,
            "mastery_estimate":      mas_score,
            "confidence_level":      confidence,
            "cognitive_load":        load,
            "attention_stability":   stability,
            "behavioral_insights":   insights,
            "adaptive_action":       action,
            "session_features":      {k: round(v, 4) for k, v in feat.items()},
            "question_analysis":     q_analysis,
        }


# Singleton
_analyzer: BehaviorAnalyzer | None = None

def get_behavior_analyzer() -> BehaviorAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = BehaviorAnalyzer()
    return _analyzer


if __name__ == "__main__":
    print("Training STRIDON Behavior Model...")
    metrics = train_behavior_model(2000)
    print("\nTesting inference...")
    analyzer = BehaviorAnalyzer()
    # Simulate a struggling student
    test_data = [
        {"time_spent": 75, "time_to_first_answer": 30, "review_time": 20,
         "n_changes": 4, "changed_back": 1, "is_correct": 0, "difficulty": 1, "question_position": i+1}
        for i in range(10)
    ]
    result = analyzer.analyze(test_data)
    print(f"\n Engagement: {result['engagement_score']}  | Mastery: {result['mastery_estimate']}")
    print(f" Confidence: {result['confidence_level']} | Load: {result['cognitive_load']} | Stability: {result['attention_stability']}")
    print(f"\n Insights:")
    for ins in result["behavioral_insights"]: print(f"  • {ins}")
    print(f"\n Action ({result['adaptive_action']['icon']}): {result['adaptive_action']['action']}")
