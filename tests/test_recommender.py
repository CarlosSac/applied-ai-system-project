from src.recommender import (
    Song, UserProfile, Recommender,
    detect_genre_bias, generate_explanation,
    genre_diversity, score_spread, feature_balance,
)

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


def test_detect_genre_bias_flags_dominant_genre_share():
    breakdown = {
        "genre": 0.30,
        "mood": 0.00,
        "energy": 0.10,
        "acoustic": 0.10,
    }
    total = sum(breakdown.values())

    is_biased, share, warning = detect_genre_bias(breakdown, total)

    assert is_biased is True
    assert share > 0.50
    assert warning is not None
    assert "Bias warning" in warning


def test_generate_explanation_includes_bias_warning_in_fallback():
    explanation = generate_explanation(
        user_prefs={"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False},
        song={"title": "x", "genre": "pop", "mood": "happy", "energy": 0.8, "acousticness": 0.2},
        score=0.60,
        reasons=["genre match (+0.30)", "energy match (+0.20)", "acoustic fit (+0.10)"],
        breakdown={"genre": 0.30, "mood": 0.00, "energy": 0.20, "acoustic": 0.10},
        bias_warning="Bias warning: genre contributes 50% of total score.",
        use_ai=False,
    )

    assert "Bias warning" in explanation


def test_detect_genre_bias_no_flag_when_genre_share_is_low():
    breakdown = {
        "genre": 0.30,
        "mood": 0.25,
        "energy": 0.22,
        "acoustic": 0.18,
    }
    total = sum(breakdown.values())

    is_biased, share, warning = detect_genre_bias(breakdown, total)

    assert is_biased is False
    assert share <= 0.50
    assert warning is None


# --- Phase 4: Evaluation metrics ---

def make_recommendations(songs_data, scores, explanations=None):
    """Helper: build a fake recommendations list without running the full pipeline."""
    if explanations is None:
        explanations = [""] * len(songs_data)
    return list(zip(songs_data, scores, explanations))


def test_genre_diversity_all_different():
    songs = [
        {"genre": "pop", "energy": 0.8, "mood": "happy", "acousticness": 0.2},
        {"genre": "rock", "energy": 0.9, "mood": "intense", "acousticness": 0.1},
        {"genre": "jazz", "energy": 0.4, "mood": "relaxed", "acousticness": 0.6},
    ]
    recs = make_recommendations(songs, [0.8, 0.7, 0.6])
    assert genre_diversity(recs) == 1.0


def test_genre_diversity_all_same():
    songs = [
        {"genre": "pop", "energy": 0.8, "mood": "happy", "acousticness": 0.2},
        {"genre": "pop", "energy": 0.85, "mood": "happy", "acousticness": 0.15},
    ]
    recs = make_recommendations(songs, [0.9, 0.8])
    assert genre_diversity(recs) == 0.5


def test_score_spread_identical_scores():
    songs = [{"genre": "pop", "energy": 0.8, "mood": "happy", "acousticness": 0.2}] * 3
    recs = make_recommendations(songs, [0.7, 0.7, 0.7])
    assert score_spread(recs) < 1e-10


def test_score_spread_varying_scores():
    songs = [{"genre": "pop", "energy": 0.8, "mood": "happy", "acousticness": 0.2}] * 3
    recs = make_recommendations(songs, [1.0, 0.5, 0.0])
    spread = score_spread(recs)
    assert spread > 0.0


def test_feature_balance_sums_to_one():
    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False}
    songs = [
        {"genre": "pop", "mood": "happy", "energy": 0.8, "acousticness": 0.2},
        {"genre": "pop", "mood": "happy", "energy": 0.75, "acousticness": 0.1},
    ]
    recs = make_recommendations(songs, [0.9, 0.85])
    balance = feature_balance(user_prefs, recs)
    assert set(balance.keys()) == {"genre", "mood", "energy", "acoustic"}
    assert abs(sum(balance.values()) - 1.0) < 1e-6
