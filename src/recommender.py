from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

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
    score = 0.0
    reasons = []

    # Genre match: worth 0.30
    if song["genre"] == user_prefs.get("genre"):
        score += 0.30
        reasons.append(f"genre match (+0.30)")

    # Mood match: worth 0.25
    if song["mood"] == user_prefs.get("mood"):
        score += 0.25
        reasons.append(f"mood match (+0.25)")

    # Energy closeness: worth up to 0.25
    # The closer song.energy is to the target, the higher the contribution
    if "energy" in user_prefs:
        energy_contribution = (1 - abs(song["energy"] - user_prefs["energy"])) * 0.25
        score += energy_contribution
        reasons.append(f"energy match (+{energy_contribution:.2f})")

    # Acoustic fit: worth up to 0.20
    # If user likes acoustic, reward high acousticness; otherwise reward low acousticness
    if "likes_acoustic" in user_prefs:
        raw = song["acousticness"] if user_prefs["likes_acoustic"] else (1 - song["acousticness"])
        acoustic_contribution = raw * 0.20
        score += acoustic_contribution
        reasons.append(f"acoustic fit (+{acoustic_contribution:.2f})")

    return score, reasons


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    """
    scored = [
        (song, *score_song(user_prefs, song))
        for song in songs
    ]
    ranked = sorted(scored, key=lambda x: x[1], reverse=True)
    return [(song, score, ", ".join(reasons)) for song, score, reasons in ranked[:k]]
