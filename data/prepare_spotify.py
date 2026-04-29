"""
Converts a Kaggle Spotify audio-features CSV into the format expected by this project.

Tested against the most common Kaggle Spotify datasets, e.g.:
  - "Spotify Tracks Dataset" (maharshipandya/spotify-tracks-dataset)
  - "Spotify Dataset 1921-2020" (yamaerenay/spotify-dataset-19212020-600k-tracks)

Usage:
    python data/prepare_spotify.py --input data/dataset.csv --output data/songs.csv --limit 200
"""

import argparse
import csv
import sys


# ---------------------------------------------------------------------------
# Mood derivation from valence + energy
# ---------------------------------------------------------------------------
def derive_mood(valence: float, energy: float) -> str:
    if valence >= 0.6 and energy >= 0.6:
        return "happy"
    if valence >= 0.6 and energy < 0.6:
        return "relaxed"
    if valence < 0.4 and energy >= 0.6:
        return "intense"
    if valence < 0.4 and energy < 0.4:
        return "sad"
    return "chill"


# ---------------------------------------------------------------------------
# Column name aliases — covers the most common Kaggle Spotify schemas
# ---------------------------------------------------------------------------
ALIASES = {
    "title":        ["track_name", "name", "title", "song_name"],
    "artist":       ["artists", "artist_name", "artist", "performer"],
    "genre":        ["track_genre", "playlist_genre", "genre", "Genre"],
    "spotify_id":   ["track_id", "spotify_id", "id"],
    "energy":       ["energy"],
    "tempo_bpm":    ["tempo"],
    "valence":      ["valence"],
    "danceability": ["danceability"],
    "acousticness": ["acousticness"],
}


def find_col(header: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in header:
            return c
    return None


def clean_genre(raw: str) -> str:
    return raw.strip().lower().replace("-", " ").replace("_", " ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Kaggle Spotify CSV to project format.")
    parser.add_argument("--input",     required=True, help="Path to the downloaded Kaggle CSV")
    parser.add_argument("--output",    default="data/songs.csv", help="Output path (default: data/songs.csv)")
    parser.add_argument("--limit",     type=int, default=500, help="Total songs to export (default: 500)")
    parser.add_argument("--per-genre", type=int, default=None, help="Cap songs per genre (default: limit / genres)")
    args = parser.parse_args()

    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []

        cols = {field: find_col(header, candidates) for field, candidates in ALIASES.items()}
        required = [f for f in ("energy", "tempo_bpm", "valence", "danceability", "acousticness")
                    if cols.get(f) is None]
        if required:
            print(f"ERROR: could not find columns for: {required}", file=sys.stderr)
            print(f"Available columns: {header}", file=sys.stderr)
            sys.exit(1)

        # Collect all valid rows grouped by genre
        by_genre: dict[str, list] = {}
        for raw in reader:
            try:
                energy       = float(raw[cols["energy"]])
                tempo_bpm    = float(raw[cols["tempo_bpm"]])
                valence      = float(raw[cols["valence"]])
                danceability = float(raw[cols["danceability"]])
                acousticness = float(raw[cols["acousticness"]])
            except (ValueError, TypeError):
                continue

            title      = raw[cols["title"]].strip()      if cols["title"]      else "Unknown"
            artist     = raw[cols["artist"]].strip()     if cols["artist"]     else "Unknown"
            genre      = clean_genre(raw[cols["genre"]]) if cols["genre"]      else "unknown"
            spotify_id = raw[cols["spotify_id"]].strip() if cols["spotify_id"] else ""
            mood       = derive_mood(valence, energy)

            by_genre.setdefault(genre, []).append({
                "title":        title,
                "artist":       artist,
                "genre":        genre,
                "mood":         mood,
                "spotify_id":   spotify_id,
                "energy":       round(energy, 4),
                "tempo_bpm":    round(tempo_bpm, 1),
                "valence":      round(valence, 4),
                "danceability": round(danceability, 4),
                "acousticness": round(acousticness, 4),
            })

        # Sample evenly across genres
        n_genres   = len(by_genre) or 1
        per_genre  = args.per_genre or max(1, args.limit // n_genres)
        rows: list = []
        for genre_rows in by_genre.values():
            rows.extend(genre_rows[:per_genre])
            if len(rows) >= args.limit:
                break
        rows = rows[:args.limit]
        print(f"Sampled {len(rows)} songs across {n_genres} genres ({per_genre} per genre max).")

    if not rows:
        print("ERROR: no valid rows found. Check your input file.", file=sys.stderr)
        sys.exit(1)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "title", "artist", "genre", "mood",
                                                "spotify_id", "energy", "tempo_bpm", "valence",
                                                "danceability", "acousticness"])
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            writer.writerow({"id": i, **row})

    print(f"Done. Wrote {len(rows)} songs to {args.output}")


if __name__ == "__main__":
    main()
