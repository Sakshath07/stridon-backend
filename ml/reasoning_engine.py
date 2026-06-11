"""
STRIDON Reasoning Engine
Generates detailed, personalized student reasoning from model outputs and SHAP values.
No external APIs used — all reasoning is derived from the ML model itself.
"""

from typing import Any


# ─────────────────────────────────────────────
# KNOWLEDGE BASE  (drives all natural language output)
# ─────────────────────────────────────────────

PERSONALITY_STREAM_INSIGHT = {
    "introvert": {
        "Science": (
            "Your introverted nature is a natural advantage in Science — deep focus, independent "
            "research, and analytical problem-solving thrive in quieter, self-directed environments."
        ),
        "Commerce": (
            "As an introvert in Commerce, you are well-suited for roles like accounting, financial "
            "analysis, and research-driven business strategy that reward careful, independent thinking."
        ),
        "Arts": (
            "Your introverted personality aligns beautifully with Arts — creative writing, visual "
            "arts, and reflective storytelling are spaces where introverts consistently excel."
        ),
    },
    "extrovert": {
        "Science": (
            "Your extroverted energy will shine in collaborative research, science communication, "
            "and team-based engineering projects within the Science stream."
        ),
        "Commerce": (
            "Commerce is an ideal home for your extroverted personality — networking, sales, "
            "leadership, and entrepreneurship all reward energy, confidence, and social fluency."
        ),
        "Arts": (
            "Your extroverted nature is a powerful asset in Arts — performing arts, journalism, "
            "public advocacy, and media communication all demand exactly your kind of energy."
        ),
    },
    "ambivert": {
        "Science": (
            "As an ambivert, you can seamlessly balance solitary research with collaborative "
            "lab work and presentations — making you a versatile Science student."
        ),
        "Commerce": (
            "Your balanced personality lets you adapt between independent financial analysis "
            "and team-based Commerce projects — a valuable trait in business environments."
        ),
        "Arts": (
            "Being an ambivert gives you the flexibility to move between deeply personal creative "
            "work and collaborative artistic projects in the Arts stream."
        ),
    },
}

LEARNING_STYLE_INSIGHT = {
    "visual": (
        "As a visual learner, you absorb knowledge best through diagrams, concept maps, "
        "infographics, and videos. Use colour-coded notes and mind maps in your studies."
    ),
    "auditory": (
        "As an auditory learner, lectures, podcasts, and group discussions are your strongest "
        "tools. Recording and replaying key concepts will accelerate your understanding."
    ),
    "kinesthetic": (
        "As a kinesthetic learner, hands-on experiments, project-based learning, and practical "
        "applications work best for you. Seek labs, internships, and workshop-style learning."
    ),
    "reading_writing": (
        "As a reading-writing learner, detailed notes, textbook study, essays, and written "
        "summaries are your most powerful tools. You thrive in research-heavy environments."
    ),
}

# Maps feature → (high value label, neutral label, low value label)
FEATURE_LABELS = {
    "analytical_score":           ("exceptional analytical thinking",        "good analytical thinking",        "developing analytical skills"),
    "creativity_score":           ("outstanding creative ability",            "solid creative thinking",          "developing creative skills"),
    "leadership_score":           ("strong natural leadership",               "growing leadership qualities",     "emerging leadership traits"),
    "communication_score":        ("excellent communication skills",          "competent communication",          "communication skills to develop"),
    "problem_solving_score":      ("exceptional problem-solving aptitude",    "good problem-solving ability",     "developing problem-solving skills"),
    "interest_science":           ("deep passion for Science",                "genuine interest in Science",      "limited interest in Science"),
    "interest_mathematics":       ("strong love for Mathematics",             "solid interest in Mathematics",    "limited interest in Mathematics"),
    "interest_arts_culture":      ("strong passion for Arts & Culture",       "genuine interest in Arts",         "limited interest in Arts"),
    "interest_business_economics":("strong business & economic acumen",       "interest in Business/Economics",   "limited interest in Business"),
    "interest_social_humanities": ("deep passion for Social Sciences",        "interest in Humanities",           "limited interest in Humanities"),
    "interest_technology":        ("deep passion for Technology",             "interest in Technology",           "limited interest in Technology"),
}

EXTRACURRICULAR_LABELS = {
    "extracurricular_sports":           "participation in sports (developing teamwork & discipline)",
    "extracurricular_music_dance":      "involvement in music/dance (nurturing artistic expression)",
    "extracurricular_debate_mun":       "experience in debate/MUN (building argumentation & critical thinking)",
    "extracurricular_coding_robotics":  "hands-on coding/robotics experience (bridging theory and technology)",
    "extracurricular_ngo_social":       "social service/NGO work (demonstrating empathy and social awareness)",
    "extracurricular_science_club":     "science club participation (applying scientific curiosity practically)",
}

# Career recommendations: stream → condition_key → list of careers
CAREER_MAP = {
    "Science": {
        "high_technology_coding":      ["Software Engineering", "AI/ML Engineering", "Data Science", "Cybersecurity"],
        "high_biology_science":        ["Medicine (MBBS)", "Biotechnology", "Pharmacy", "Environmental Science"],
        "high_analytical_math":        ["Engineering (IIT/NIT)", "Physics Research", "Architecture", "Actuarial Science"],
        "default":                     ["Engineering", "Medicine", "Research Scientist", "Data Science", "Pharmacy"],
    },
    "Commerce": {
        "high_analytical_leadership":  ["Chartered Accountancy (CA)", "Investment Banking", "Economics", "Finance"],
        "high_communication_social":   ["Marketing Management", "Human Resources", "Public Relations", "Consulting"],
        "high_leadership_debate":      ["Entrepreneurship", "Business Law (LLB)", "MBA", "Political Science + Commerce"],
        "default":                     ["Chartered Accountancy", "Business Administration (BBA/MBA)", "Economics", "Marketing", "Finance"],
    },
    "Arts": {
        "high_creativity_arts":        ["Fine Arts", "Graphic Design", "Film & Media Production", "Creative Writing"],
        "high_social_communication":   ["Journalism & Mass Communication", "Psychology", "Social Work", "Political Science"],
        "high_communication_debate":   ["Law (LLB)", "Teaching & Education", "Content Creation", "Philosophy"],
        "default":                     ["Psychology", "Journalism", "Fine Arts", "Law", "Literature", "Social Work"],
    },
}

STREAM_OVERVIEW = {
    "Science": (
        "The Science stream (Physics, Chemistry, Mathematics/Biology) is India's most competitive "
        "and opens doors to engineering, medicine, research, and technology careers. It demands "
        "strong logical reasoning, mathematical ability, and a curious, analytical mindset."
    ),
    "Commerce": (
        "The Commerce stream covers Economics, Accountancy, Business Studies, and Mathematics. "
        "It leads to careers in finance, business, law, and management. Commerce suits students "
        "who enjoy understanding how the world of money, markets, and organisations works."
    ),
    "Arts": (
        "The Arts/Humanities stream explores Literature, History, Geography, Psychology, Political "
        "Science, and Sociology. It is the gateway to creative, social, and liberal careers — "
        "journalism, design, law, psychology, civil services, and the performing arts."
    ),
}

CONFIDENCE_LABELS = {
    "very_high": "very strong",
    "high": "strong",
    "moderate": "moderate",
    "low": "preliminary",
}

# ─────────────────────────────────────────────
# ISSUE DETECTOR
# ─────────────────────────────────────────────

def detect_learning_issues(features: dict) -> list[str]:
    """
    Identify potential learning/development issues from the student's profile.
    Returns a list of human-readable issue descriptions.
    """
    issues = []

    score_fields = [
        "analytical_score", "creativity_score", "leadership_score",
        "communication_score", "problem_solving_score",
    ]
    low_scores = [f for f in score_fields if features.get(f, 3) <= 2]
    if len(low_scores) >= 3:
        issues.append(
            "Multiple core skills (analytical thinking, creativity, communication) appear to be in early "
            "development. A structured skill-building plan — including reading, group activities, and "
            "problem-solving exercises — will be essential before stream selection."
        )
    elif "analytical_score" in low_scores and "problem_solving_score" in low_scores:
        issues.append(
            "Analytical and problem-solving scores are currently low. Strengthening these through "
            "math puzzles, logic games, and structured reasoning exercises is recommended — especially "
            "if considering Science or Commerce."
        )
    elif "communication_score" in low_scores and "leadership_score" in low_scores:
        issues.append(
            "Communication and leadership skills are in early stages. Joining debate clubs, participating "
            "in group projects, and taking on small leadership roles (class rep, event coordinator) will "
            "develop these significantly."
        )
    elif "creativity_score" in low_scores:
        issues.append(
            "Creative thinking is still developing. Engaging with art, music, storytelling, and "
            "open-ended projects can help unlock creative potential — valuable in any stream."
        )

    extracurriculars = [
        "extracurricular_sports", "extracurricular_music_dance", "extracurricular_debate_mun",
        "extracurricular_coding_robotics", "extracurricular_ngo_social", "extracurricular_science_club",
    ]
    active = sum(features.get(e, 0) for e in extracurriculars)
    if active == 0:
        issues.append(
            "No extracurricular activities are currently listed. Participating in at least one "
            "activity — whether sports, arts, debate, or a science club — is strongly recommended "
            "for holistic development and future college applications."
        )

    interest_fields = [
        "interest_science", "interest_mathematics", "interest_arts_culture",
        "interest_business_economics", "interest_social_humanities", "interest_technology",
    ]
    high_interests = sum(1 for f in interest_fields if features.get(f, 0) >= 4)
    if high_interests >= 5:
        issues.append(
            "Your interests are very broad, spanning multiple domains. While curiosity is a strength, "
            "stream selection requires focusing on 2–3 core passions. Take time to explore which "
            "subjects genuinely excite you beyond surface level."
        )

    all_interests_low = all(features.get(f, 3) <= 2 for f in interest_fields)
    if all_interests_low:
        issues.append(
            "Interest levels appear low across most subject areas. This is common at this stage — "
            "exploring books, YouTube educational channels, and talking to older students or "
            "professionals in different fields can help ignite genuine passion."
        )

    return issues


# ─────────────────────────────────────────────
# CAREER RECOMMENDER
# ─────────────────────────────────────────────

def recommend_careers(stream: str, features: dict) -> list[str]:
    """Pick the most relevant careers based on the student's top features."""
    f = features

    if stream == "Science":
        if f.get("interest_technology", 0) >= 4 or f.get("extracurricular_coding_robotics", 0):
            return CAREER_MAP["Science"]["high_technology_coding"]
        if f.get("interest_science", 0) >= 4 and f.get("analytical_score", 0) <= 3:
            return CAREER_MAP["Science"]["high_biology_science"]
        if f.get("analytical_score", 0) >= 4 and f.get("interest_mathematics", 0) >= 4:
            return CAREER_MAP["Science"]["high_analytical_math"]
        return CAREER_MAP["Science"]["default"]

    elif stream == "Commerce":
        if f.get("analytical_score", 0) >= 4 and f.get("leadership_score", 0) >= 4:
            return CAREER_MAP["Commerce"]["high_analytical_leadership"]
        if f.get("communication_score", 0) >= 4 and f.get("interest_social_humanities", 0) >= 3:
            return CAREER_MAP["Commerce"]["high_communication_social"]
        if f.get("leadership_score", 0) >= 4 or f.get("extracurricular_debate_mun", 0):
            return CAREER_MAP["Commerce"]["high_leadership_debate"]
        return CAREER_MAP["Commerce"]["default"]

    else:  # Arts
        if f.get("creativity_score", 0) >= 4 and f.get("interest_arts_culture", 0) >= 4:
            return CAREER_MAP["Arts"]["high_creativity_arts"]
        if f.get("interest_social_humanities", 0) >= 4 or f.get("extracurricular_ngo_social", 0):
            return CAREER_MAP["Arts"]["high_social_communication"]
        if f.get("communication_score", 0) >= 4 or f.get("extracurricular_debate_mun", 0):
            return CAREER_MAP["Arts"]["high_communication_debate"]
        return CAREER_MAP["Arts"]["default"]


# ─────────────────────────────────────────────
# SHAP → NATURAL LANGUAGE
# ─────────────────────────────────────────────

def _get_feature_label(feature: str, value: Any) -> str:
    """Convert a feature + its value to a human-readable description."""
    if feature in FEATURE_LABELS:
        labels = FEATURE_LABELS[feature]
        val = float(value)
        if val >= 4:
            return labels[0]
        elif val >= 3:
            return labels[1]
        else:
            return labels[2]
    return feature.replace("_", " ")


def _shap_to_sentences(
    shap_values: list[tuple[str, float, Any]],
    stream: str,
    top_n: int = 5,
) -> tuple[list[str], list[str]]:
    """
    Convert SHAP values into positive and negative factor sentences.

    Args:
        shap_values: List of (feature_name, shap_value, actual_value) tuples
        stream: Predicted stream
        top_n: Number of top factors to narrate

    Returns:
        (positive_sentences, negative_sentences)
    """
    positives = []
    negatives = []

    sorted_by_impact = sorted(shap_values, key=lambda x: abs(x[1]), reverse=True)

    for feature, shap_val, actual_val in sorted_by_impact[:top_n]:
        if feature in EXTRACURRICULAR_LABELS:
            if actual_val == 1 and shap_val > 0:
                positives.append(f"Your {EXTRACURRICULAR_LABELS[feature]}")
            elif actual_val == 0 and shap_val < 0:
                negatives.append(
                    f"You haven't yet explored {feature.replace('extracurricular_', '').replace('_', '/')} — "
                    f"this could enrich your profile"
                )
        elif feature in FEATURE_LABELS:
            label = _get_feature_label(feature, actual_val)
            if shap_val > 0.05:
                # Only surface it as a strength if the score is actually solid
                if float(actual_val) >= 3:
                    positives.append(f"Your {label}")
            elif shap_val < -0.05:
                if float(actual_val) < 3:
                    negatives.append(f"Your {label} could be strengthened further")

    return positives, negatives


# ─────────────────────────────────────────────
# MAIN REASONING BUILDER
# ─────────────────────────────────────────────

def build_reasoning(
    stream: str,
    confidence: float,
    features: dict,
    shap_values: list[tuple[str, float, Any]],
    all_probabilities: dict[str, float],
) -> dict:
    """
    Generate the full, structured reasoning for a student's stream recommendation.

    Returns a dict with:
        - summary: one-sentence headline
        - stream_overview: what this stream is about
        - personality_insight: personality + learning style analysis
        - key_strengths: list of top driving factors (from SHAP)
        - why_this_stream: paragraph explaining the recommendation
        - career_paths: list of recommended careers
        - areas_to_develop: list of growth areas / issues
        - alternative_streams: brief note on runner-up streams
        - full_report: combined natural language report (string)
    """

    # Confidence tier
    if confidence >= 0.80:
        conf_label = "very_high"
    elif confidence >= 0.65:
        conf_label = "high"
    elif confidence >= 0.50:
        conf_label = "moderate"
    else:
        conf_label = "low"

    personality = features.get("personality", "ambivert")
    learning_style = features.get("learning_style", "visual")

    # SHAP-derived positives/negatives
    positives, negatives = _shap_to_sentences(shap_values, stream)

    # Issues
    issues = detect_learning_issues(features)

    # Careers
    careers = recommend_careers(stream, features)

    # Alternative streams
    sorted_streams = sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)
    alt_streams = [s for s, _ in sorted_streams if s != stream]

    # ── Build paragraphs ──────────────────────────────────────────

    # Para 1: Summary
    summary = (
        f"Based on your personality, interests, and activities, the {stream} stream is your "
        f"best-fit academic path with a {CONFIDENCE_LABELS[conf_label]} match "
        f"({confidence * 100:.0f}% confidence)."
    )

    # Para 2: Personality + Learning Style
    personality_insight = (
        f"{PERSONALITY_STREAM_INSIGHT.get(personality, {}).get(stream, '')} "
        f"{LEARNING_STYLE_INSIGHT.get(learning_style, '')}"
    ).strip()

    # Para 3: Key driving factors
    strengths_text = ""
    if positives:
        if len(positives) == 1:
            strengths_text = f"{positives[0]} is the primary factor driving this recommendation."
        else:
            strengths_text = (
                "The key factors driving this recommendation are: "
                + ", ".join(positives[:-1])
                + (f", and {positives[-1]}." if len(positives) > 1 else f". {positives[0]}.")
            )

    # Para 4: Why this stream (stream overview woven with student data)
    interest_highlights = []
    for field, label in [
        ("interest_science", "Science"), ("interest_mathematics", "Mathematics"),
        ("interest_technology", "Technology"), ("interest_arts_culture", "Arts & Culture"),
        ("interest_business_economics", "Business/Economics"),
        ("interest_social_humanities", "Social Sciences"),
    ]:
        if features.get(field, 0) >= 4:
            interest_highlights.append(label)

    interest_str = ""
    if interest_highlights:
        interest_str = (
            f" Your strong interest in {', '.join(interest_highlights[:3])} "
            f"directly aligns with the core subjects of this stream."
        )

    why_stream = f"{STREAM_OVERVIEW[stream]}{interest_str}"

    # Para 5: Alternative streams
    alt_text = ""
    if alt_streams:
        alt_probs = [f"{s} ({all_probabilities[s] * 100:.0f}%)" for s in alt_streams]
        alt_text = (
            f"Your profile also shows some affinity for {alt_probs[0]}. "
            f"If you feel uncertain, exploring subjects from both streams before finalising is advisable."
        )

    # ── Full report string ──────────────────────────────────────────
    sections = [
        f"🎯 RECOMMENDATION: {stream} Stream\n{summary}",
        f"\n📚 About the {stream} Stream\n{why_stream}",
        f"\n🧠 Your Personality & Learning Profile\n{personality_insight}",
    ]
    if strengths_text:
        sections.append(f"\n✅ Key Strengths Behind This Recommendation\n{strengths_text}")
    if negatives:
        neg_str = " ".join(negatives[:2]) + "." if negatives else ""
        sections.append(f"\n⚠️ Areas to Be Aware Of\n{neg_str}")
    sections.append(
        f"\n💼 Recommended Career Paths\n"
        + "\n".join(f"  • {c}" for c in careers[:5])
    )
    if issues:
        sections.append(
            f"\n🔧 Areas to Work On\n"
            + "\n".join(f"  • {i}" for i in issues)
        )
    if alt_text:
        sections.append(f"\n🔀 Alternative Consideration\n{alt_text}")

    full_report = "\n".join(sections)

    return {
        "summary": summary,
        "stream_overview": why_stream,
        "personality_insight": personality_insight,
        "key_strengths": positives,
        "why_this_stream": why_stream,
        "career_paths": careers[:5],
        "areas_to_develop": issues,
        "negative_factors": negatives,
        "alternative_streams": alt_text,
        "full_report": full_report,
    }
