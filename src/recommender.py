import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_env_path)
except ImportError:
    # python-dotenv not installed: parse .env manually
    if os.path.isfile(_env_path):
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip())

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        user_prefs = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
        }

        scored_songs = []
        for song in self.songs:
            song_dict = {
                "id": song.id,
                "title": song.title,
                "artist": song.artist,
                "genre": song.genre,
                "mood": song.mood,
                "energy": song.energy,
                "tempo_bpm": song.tempo_bpm,
                "valence": song.valence,
                "danceability": song.danceability,
                "acousticness": song.acousticness,
            }
            score, _ = score_song(user_prefs, song_dict)
            scored_songs.append((song, score))

        ranked = sorted(scored_songs, key=lambda x: x[1], reverse=True)
        return [song for song, _ in ranked[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        parts = []

        if song.genre == user.favorite_genre:
            parts.append(f"genre matches your preference ({user.favorite_genre})")
        else:
            parts.append(f"genre differs from your preference ({user.favorite_genre})")

        if song.mood == user.favorite_mood:
            parts.append(f"mood matches your preference ({user.favorite_mood})")
        else:
            parts.append(f"mood differs from your preference ({user.favorite_mood})")

        energy_gap = abs(song.energy - user.target_energy)
        parts.append(f"energy is {song.energy:.2f} vs target {user.target_energy:.2f} (gap {energy_gap:.2f})")

        if user.likes_acoustic:
            parts.append(f"acousticness is {song.acousticness:.2f}, which helps for acoustic taste")
        else:
            non_acoustic_fit = 1 - song.acousticness
            parts.append(
                f"acousticness is {song.acousticness:.2f}, so non-acoustic fit is {non_acoustic_fit:.2f}"
            )

        return "; ".join(parts) + "."


def _score_breakdown(user_prefs: Dict, song: Dict) -> Dict[str, float]:
    """
    Returns weighted contribution per feature.
    """
    breakdown = {
        "genre": 0.0,
        "mood": 0.0,
        "energy": 0.0,
        "acoustic": 0.0,
    }

    if song["genre"] == user_prefs.get("genre"):
        breakdown["genre"] = 0.30

    if song["mood"] == user_prefs.get("mood"):
        breakdown["mood"] = 0.25

    if "energy" in user_prefs:
        breakdown["energy"] = (1 - abs(song["energy"] - user_prefs["energy"])) * 0.25

    if "likes_acoustic" in user_prefs:
        raw = song["acousticness"] if user_prefs["likes_acoustic"] else (1 - song["acousticness"])
        breakdown["acoustic"] = raw * 0.20

    return breakdown


def detect_genre_bias(
    breakdown: Dict[str, float],
    total_score: float,
    threshold: float = 0.50,
) -> Tuple[bool, float, Optional[str]]:
    """
    Flags when genre contribution dominates the final score.
    """
    if total_score <= 0:
        return False, 0.0, None

    genre_share = breakdown.get("genre", 0.0) / total_score
    if genre_share > threshold:
        warning = (
            f"Bias warning: genre contributes {genre_share:.0%} of total score "
            f"(threshold {threshold:.0%})."
        )
        return True, genre_share, warning

    return False, genre_share, None


def generate_explanation(
    user_prefs: Dict,
    song: Dict,
    score: float,
    reasons: List[str],
    breakdown: Dict[str, float],
    bias_warning: Optional[str] = None,
    use_ai: bool = True,
) -> str:
    """
    Generates an explanation with Gemini if configured.
    Falls back to deterministic rule-based text when unavailable.
    """
    fallback = f"score {score:.2f}: " + ", ".join(reasons)
    if bias_warning:
        fallback += f" | {bias_warning}"
    if not use_ai:
        return fallback

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return fallback

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        bias_instruction = (
            f"\n⚠️ REQUIRED: Your explanation MUST include this bias warning verbatim or paraphrased:\n"
            f'"{bias_warning}"\n'
        ) if bias_warning else ""

        prompt = (
            "You are explaining why a music recommender selected a song.\n"
            f"{bias_instruction}"
            "Write 2-3 sentences in plain English. Do not use markdown formatting.\n"
            "- Mention the strongest contributing features and the total score.\n"
            + ("- Explicitly state the bias warning above.\n" if bias_warning else "")
            + "Do not invent features.\n\n"
            f"User preferences: {user_prefs}\n"
            f"Song metadata: {song}\n"
            f"Weighted score breakdown: {breakdown}\n"
            f"Rule traces: {reasons}\n"
            f"Total score: {score:.2f}\n"
        )

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        llm_text = message.content[0].text
        if isinstance(llm_text, str) and llm_text.strip():
            import re
            cleaned = re.sub(r'\*\*(.+?)\*\*', r'\1', llm_text.strip())
            return cleaned
        return fallback
    except Exception as exc:
        print(f"[Claude] unavailable: {type(exc).__name__}: {exc}")
        return fallback

def check_claude_health() -> bool:
    """
    Sends a minimal prompt to verify the API key and quota are functional.
    Call once at startup before running recommendations.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("[Claude] ANTHROPIC_API_KEY not set — will use rule-based fallback.")
        return False
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        )
        text = message.content[0].text.strip().lower()
        if text:
            print(f"[Claude] health check passed (response: '{text}')")
            return True
        print("[Claude] health check returned empty response — will use fallback.")
        return False
    except Exception as exc:
        print(f"[Claude] health check failed: {type(exc).__name__}: {exc}")
        return False


def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    import csv

    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append({
                "id":           int(row["id"]),
                "title":        row["title"],
                "artist":       row["artist"],
                "genre":        row["genre"],
                "mood":         row["mood"],
                "spotify_id":   row.get("spotify_id", ""),
                "energy":       float(row["energy"]),
                "tempo_bpm":    float(row["tempo_bpm"]),
                "valence":      float(row["valence"]),
                "danceability": float(row["danceability"]),
                "acousticness": float(row["acousticness"]),
            })
    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against a user profile.
    Returns a total score (0.0 – 1.0) and a list of reasons explaining each contribution.
    """
    breakdown = _score_breakdown(user_prefs, song)
    score = sum(breakdown.values())
    reasons = []

    # Genre match: worth 0.30
    if breakdown["genre"] > 0:
        reasons.append(f"genre match (+0.30)")

    # Mood match: worth 0.25
    if breakdown["mood"] > 0:
        reasons.append(f"mood match (+0.25)")

    # Energy closeness: worth up to 0.25
    # The closer song.energy is to the target, the higher the contribution
    if "energy" in user_prefs:
        energy_contribution = breakdown["energy"]
        reasons.append(f"energy match (+{energy_contribution:.2f})")

    # Acoustic fit: worth up to 0.20
    # If user likes acoustic, reward high acousticness; otherwise reward low acousticness
    if "likes_acoustic" in user_prefs:
        acoustic_contribution = breakdown["acoustic"]
        reasons.append(f"acoustic fit (+{acoustic_contribution:.2f})")

    return score, reasons


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    """
    scored = []
    for song in songs:
        breakdown = _score_breakdown(user_prefs, song)
        score = sum(breakdown.values())
        _, reasons = score_song(user_prefs, song)
        _, _, bias_warning = detect_genre_bias(breakdown, score)
        scored.append((song, score, reasons, breakdown, bias_warning))

    ranked = sorted(scored, key=lambda x: x[1], reverse=True)
    results = []
    for song, score, reasons, breakdown, bias_warning in ranked[:k]:
        explanation = generate_explanation(
            user_prefs,
            song,
            score,
            reasons,
            breakdown,
            bias_warning=bias_warning,
        )
        results.append((song, score, explanation))
    return results


def genre_diversity(recommendations: List[Tuple[Dict, float, str]]) -> float:
    """Unique genres in top-k divided by k."""
    if not recommendations:
        return 0.0
    genres = {song["genre"] for song, _, _ in recommendations}
    return len(genres) / len(recommendations)


def score_spread(recommendations: List[Tuple[Dict, float, str]]) -> float:
    """Standard deviation of scores across top-k recommendations."""
    if len(recommendations) < 2:
        return 0.0
    scores = [score for _, score, _ in recommendations]
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    return variance ** 0.5


def feature_balance(
    user_prefs: Dict,
    recommendations: List[Tuple[Dict, float, str]],
) -> Dict[str, float]:
    """Average contribution share per feature across top-k recommendations."""
    if not recommendations:
        return {}
    totals: Dict[str, float] = {"genre": 0.0, "mood": 0.0, "energy": 0.0, "acoustic": 0.0}
    counted = 0
    for song, _, _ in recommendations:
        breakdown = _score_breakdown(user_prefs, song)
        total_score = sum(breakdown.values())
        if total_score > 0:
            for feature, value in breakdown.items():
                totals[feature] += value / total_score
            counted += 1
    if counted == 0:
        return totals
    return {f: v / counted for f, v in totals.items()}


def print_metrics(
    user_prefs: Dict,
    recommendations: List[Tuple[Dict, float, str]],
) -> None:
    """Prints a metrics summary for one profile run."""
    diversity = genre_diversity(recommendations)
    spread = score_spread(recommendations)
    balance = feature_balance(user_prefs, recommendations)
    balance_str = "  ".join(f"{f}: {v:.0%}" for f, v in balance.items())
    print(f"  [Metrics] Diversity: {diversity:.2f}  |  Spread: {spread:.3f}")
    print(f"  [Metrics] Balance  : {balance_str}")
