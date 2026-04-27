from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st

try:
    from src.recommender import (
        _score_breakdown,
        check_claude_health,
        detect_genre_bias,
        feature_balance,
        generate_explanation,
        genre_diversity,
        load_songs,
        score_song,
        score_spread,
    )
except ModuleNotFoundError:
    from recommender import (
        _score_breakdown,
        check_claude_health,
        detect_genre_bias,
        feature_balance,
        generate_explanation,
        genre_diversity,
        load_songs,
        score_song,
        score_spread,
    )


def build_recommendations(
    user_prefs: Dict,
    songs: List[Dict],
    k: int,
    use_ai: bool,
    bias_threshold: float,
) -> List[Dict]:
    scored: List[Dict] = []

    for song in songs:
        breakdown = _score_breakdown(user_prefs, song)
        score, reasons = score_song(user_prefs, song)
        _, genre_share, bias_warning = detect_genre_bias(
            breakdown, score, threshold=bias_threshold
        )
        scored.append(
            {
                "song": song,
                "score": score,
                "reasons": reasons,
                "breakdown": breakdown,
                "genre_share": genre_share,
                "bias_warning": bias_warning,
            }
        )

    scored.sort(key=lambda row: row["score"], reverse=True)
    top_rows = scored[:k]
    for row in top_rows:
        row["explanation"] = generate_explanation(
            user_prefs=user_prefs,
            song=row["song"],
            score=row["score"],
            reasons=row["reasons"],
            breakdown=row["breakdown"],
            bias_warning=row["bias_warning"],
            use_ai=use_ai,
        )
    return top_rows


def make_metrics_payload(recommendations: List[Dict]) -> Tuple[float, float, Dict[str, float]]:
    as_tuples = [
        (row["song"], row["score"], row["explanation"]) for row in recommendations
    ]
    if not as_tuples:
        return 0.0, 0.0, {}

    first_user_prefs = st.session_state.get("user_prefs", {})
    diversity = genre_diversity(as_tuples)
    spread = score_spread(as_tuples)
    balance = feature_balance(first_user_prefs, as_tuples)
    return diversity, spread, balance


def app() -> None:
    st.set_page_config(
        page_title="Music Recommender Studio",
        page_icon="music",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Fraunces:opsz,wght@9..144,600&display=swap');

        :root {
            --app-bg-top: #faf7f2;
            --app-bg-bottom: #fffdf9;
            --text-main: #2b2b2b;
            --heading: #2f1f1a;
            --card-bg: rgba(255, 255, 255, 0.86);
            --card-border: #efe2d0;
            --shadow: rgba(38, 28, 24, 0.08);
            --pill-bg: #fff4d6;
            --pill-border: #f4c96a;
            --pill-text: #8c5a00;
        }

        html[data-theme='light'] {
            --app-bg-top: #faf7f2;
            --app-bg-bottom: #fffdf9;
            --text-main: #2b2b2b;
            --heading: #2f1f1a;
            --card-bg: rgba(255, 255, 255, 0.86);
            --card-border: #efe2d0;
            --shadow: rgba(38, 28, 24, 0.08);
            --pill-bg: #fff4d6;
            --pill-border: #f4c96a;
            --pill-text: #8c5a00;
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --app-bg-top: #171717;
                --app-bg-bottom: #0f1115;
                --text-main: #ececec;
                --heading: #f6efe7;
                --card-bg: rgba(30, 33, 40, 0.84);
                --card-border: #3a3f49;
                --shadow: rgba(0, 0, 0, 0.3);
                --pill-bg: #3a2f15;
                --pill-border: #7d6321;
                --pill-text: #ffd98e;
            }
        }

        html[data-theme='dark'] {
            --app-bg-top: #171717;
            --app-bg-bottom: #0f1115;
            --text-main: #ececec;
            --heading: #f6efe7;
            --card-bg: rgba(30, 33, 40, 0.84);
            --card-border: #3a3f49;
            --shadow: rgba(0, 0, 0, 0.3);
            --pill-bg: #3a2f15;
            --pill-border: #7d6321;
            --pill-text: #ffd98e;
        }

        .stApp {
            background:
                radial-gradient(1100px 500px at 90% -5%, rgba(246, 173, 85, 0.18), transparent 60%),
                radial-gradient(900px 420px at -10% 20%, rgba(45, 212, 191, 0.16), transparent 55%),
                linear-gradient(180deg, var(--app-bg-top) 0%, var(--app-bg-bottom) 100%);
            color: var(--text-main);
        }

        [data-testid='stAppViewContainer'],
        [data-testid='stHeader'] {
            background:
                radial-gradient(1100px 500px at 90% -5%, rgba(246, 173, 85, 0.18), transparent 60%),
                radial-gradient(900px 420px at -10% 20%, rgba(45, 212, 191, 0.16), transparent 55%),
                linear-gradient(180deg, var(--app-bg-top) 0%, var(--app-bg-bottom) 100%);
            color: var(--text-main);
        }

        h1, h2, h3 {
            font-family: 'Fraunces', serif;
            color: var(--heading);
            letter-spacing: 0.2px;
        }

        p, li, label, .stMarkdown, .stCaption {
            font-family: 'Space Grotesk', sans-serif;
            color: var(--text-main);
        }

        .hero-card {
            border: 1px solid var(--card-border);
            background: var(--card-bg);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 28px var(--shadow);
        }

        .song-card {
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.7rem;
            background: var(--card-bg);
            box-shadow: 0 8px 24px var(--shadow);
        }

        .bias-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 0.15rem 0.6rem;
            background: var(--pill-bg);
            border: 1px solid var(--pill-border);
            color: var(--pill-text);
            font-size: 0.8rem;
            font-weight: 700;
            margin-left: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero-card">
            <h1 style="margin-bottom:0.2rem;">Music Recommender Studio</h1>
            <p style="margin-top:0;">
                Explore profile-driven recommendations with AI explanations, genre-bias checks,
                and Phase 4 evaluation metrics in one dashboard.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    data_path = Path(__file__).resolve().parent.parent / "data" / "songs.csv"
    songs = load_songs(str(data_path))

    genres = sorted({song["genre"] for song in songs})
    moods = sorted({song["mood"] for song in songs})

    with st.sidebar:
        with st.form("controls_form"):
            st.header("Profile")
            selected_genre = st.selectbox("Preferred genre", options=genres, index=0)
            selected_mood = st.selectbox("Preferred mood", options=moods, index=0)
            selected_energy = st.slider("Target energy", 0.0, 1.0, 0.75, 0.01)
            likes_acoustic = st.toggle("Likes acoustic songs", value=False)

            st.divider()
            st.header("Run Settings")
            k = st.slider("Top-k recommendations", 1, min(10, len(songs)), 3)
            bias_threshold = st.slider("Bias threshold (genre share)", 0.30, 0.80, 0.50, 0.01)
            use_ai = st.toggle("Use Claude for explanations", value=True)
            run_clicked = st.form_submit_button("Generate recommendations", use_container_width=True)

        if st.button("Check Claude health", use_container_width=True):
            healthy = check_claude_health()
            if healthy:
                st.success("Claude API is reachable.")
            else:
                st.warning("Claude unavailable. Falling back to rule-based explanations.")

    user_prefs = {
        "genre": selected_genre,
        "mood": selected_mood,
        "energy": selected_energy,
        "likes_acoustic": likes_acoustic,
    }
    st.session_state["user_prefs"] = user_prefs

    if run_clicked:
        st.session_state["recommendations"] = build_recommendations(
            user_prefs=user_prefs,
            songs=songs,
            k=k,
            use_ai=use_ai,
            bias_threshold=bias_threshold,
        )
    recommendations = st.session_state.get("recommendations", [])
    if not recommendations:
        st.info("Set your profile and click 'Generate recommendations'.")

    diversity, spread, balance = make_metrics_payload(recommendations)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Genre Diversity", f"{diversity:.2f}")
    metric_col2.metric("Score Spread", f"{spread:.3f}")
    top_feature = max(balance, key=balance.get) if balance else "n/a"
    metric_col3.metric("Top Feature Share", f"{top_feature}: {balance.get(top_feature, 0.0):.0%}")

    st.subheader("Feature Balance")
    if balance:
        pretty_balance = {
            "Genre": balance.get("genre", 0.0),
            "Mood": balance.get("mood", 0.0),
            "Energy": balance.get("energy", 0.0),
            "Acoustic": balance.get("acoustic", 0.0),
        }
        st.bar_chart(pretty_balance)
    else:
        st.caption("No recommendations yet.")

    st.subheader("Recommendations")
    for rank, row in enumerate(recommendations, start=1):
        song = row["song"]
        warning_html = (
            '<span class="bias-pill">Genre Bias Flag</span>' if row["bias_warning"] else ""
        )

        st.markdown(
            f"""
            <div class="song-card">
                <h3 style="margin:0;">#{rank} {song['title']} - {song['artist']} {warning_html}</h3>
                <p style="margin:0.35rem 0 0 0;"><b>Score:</b> {row['score']:.2f}</p>
                <p style="margin:0.35rem 0 0 0;"><b>Genre share:</b> {row['genre_share']:.0%}</p>
                <p style="margin:0.35rem 0 0 0;"><b>Why:</b> {row['explanation']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if row["bias_warning"]:
            st.warning(row["bias_warning"])


if __name__ == "__main__":
    app()
