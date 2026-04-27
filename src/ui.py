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


def make_metrics_payload(user_prefs: Dict, recommendations: List[Dict]) -> Tuple[float, float, Dict[str, float]]:
    as_tuples = [
        (row["song"], row["score"], row["explanation"]) for row in recommendations
    ]
    if not as_tuples:
        return 0.0, 0.0, {}

    diversity = genre_diversity(as_tuples)
    spread = score_spread(as_tuples)
    balance = feature_balance(user_prefs, as_tuples)
    return diversity, spread, balance


def app() -> None:
    st.set_page_config(
        page_title="Music Recommender Studio",
        page_icon="🎵",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Fraunces:opsz,wght@9..144,600&display=swap');

        :root {
            --app-bg-top: var(--background-color);
            --app-bg-bottom: var(--background-color);
            --text-main: var(--text-color);
            --heading: var(--text-color);
            --card-bg: var(--secondary-background-color);
            --card-border: rgba(127, 127, 127, 0.35);
            --shadow: rgba(38, 28, 24, 0.08);
            --pill-bg: #fff4d6;
            --pill-border: #f4c96a;
            --pill-text: #8c5a00;
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
                and evaluation metrics in one dashboard.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    @st.cache_data
    def _load_songs(path: str) -> list:
        return load_songs(path)

    data_path = Path(__file__).resolve().parent.parent / "data" / "songs.csv"
    songs = _load_songs(str(data_path))

    genres = sorted({song["genre"] for song in songs})
    moods = sorted({song["mood"] for song in songs})

    with st.sidebar:
        with st.form("controls_form"):
            st.header("Profile")
            selected_genre = st.selectbox("Preferred genre", options=genres, index=0)
            selected_mood = st.selectbox("Preferred mood", options=moods, index=0)
            selected_energy = st.slider("Target energy", 0.0, 1.0, 0.75, 0.01)
            likes_acoustic = st.toggle("Likes acoustic songs", value=False)

            with st.expander("Advanced settings"):
                k = st.slider("Top-k recommendations", 1, min(10, len(songs)), 3)
                bias_threshold = st.slider("Bias threshold (genre share)", 0.30, 0.80, 0.50, 0.01)
                use_ai = st.toggle("Use Claude for explanations", value=True)

            st.divider()
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
        with st.spinner("Generating recommendations..."):
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

    st.subheader("Recommendations")
    for rank, row in enumerate(recommendations, start=1):
        song = row["song"]
        warning_html = (
            '<span class="bias-pill">Genre Bias Flag</span>' if row["bias_warning"] else ""
        )

        breakdown = row["breakdown"]
        total = sum(breakdown.values()) or 1

        def bar(label: str, value: float, color: str) -> str:
            pct = value / total * 100
            return (
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
                f'<span style="width:58px;font-size:0.75rem;opacity:0.75;">{label}</span>'
                f'<div style="flex:1;background:rgba(127,127,127,0.15);border-radius:4px;height:12px;">'
                f'<div style="width:{pct:.1f}%;background:{color};border-radius:4px;height:100%;"></div>'
                f'</div>'
                f'<span style="width:32px;font-size:0.75rem;text-align:right;">{pct:.0f}%</span>'
                f'</div>'
            )

        breakdown_html = (
            bar("Genre",    breakdown["genre"],    "#f472b6") +
            bar("Mood",     breakdown["mood"],     "#a78bfa") +
            bar("Energy",   breakdown["energy"],   "#f6ad55") +
            bar("Acoustic", breakdown["acoustic"], "#2dd4bf")
        )

        st.markdown(
            f"""
            <div class="song-card" style="display:flex;gap:0;align-items:stretch;">
                <div style="flex:3;padding-right:1.25rem;">
                    <h3 style="margin:0;">#{rank} {song['title']} - {song['artist']} {warning_html}</h3>
                    <p style="margin:0.35rem 0 0 0;"><b>Score:</b> {row['score']:.2f}</p>
                    <p style="margin:0.35rem 0 0 0;"><b>Why:</b> {row['explanation']}</p>
                </div>
                <div style="width:1px;background:rgba(127,127,127,0.15);flex-shrink:0;"></div>
                <div style="flex:2;min-width:180px;align-self:center;padding-left:1.25rem;">
                    <p style="margin:0 0 0.4rem 0;font-size:0.75rem;opacity:0.6;font-weight:600;letter-spacing:0.5px;">SCORE BREAKDOWN</p>
                    {breakdown_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if row["bias_warning"]:
            st.warning(row["bias_warning"])

    if recommendations:
        st.divider()
        st.subheader("Evaluation Metrics")
        diversity, spread, balance = make_metrics_payload(user_prefs, recommendations)

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Genre Diversity", f"{diversity:.2f}")
        metric_col2.metric("Score Spread", f"{spread:.3f}")
        top_feature = max(balance, key=lambda f: balance[f]) if balance else "n/a"
        metric_col3.metric("Top Feature Share", f"{top_feature}: {balance.get(top_feature, 0.0):.0%}")

        st.subheader("Feature Balance")
        import plotly.graph_objects as go
        features = ["Genre", "Mood", "Energy", "Acoustic"]
        bal_values = [
            balance.get("genre", 0.0),
            balance.get("mood", 0.0),
            balance.get("energy", 0.0),
            balance.get("acoustic", 0.0),
        ]
        bal_colors = ["#f472b6", "#a78bfa", "#f6ad55", "#2dd4bf"]
        bal_fig = go.Figure(go.Bar(
            x=features,
            y=bal_values,
            marker_color=bal_colors,
            text=[f"{v:.0%}" for v in bal_values],
            textposition="outside",
        ))
        bal_fig.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=8, b=0),
            yaxis=dict(tickformat=".0%", range=[0, max(bal_values) * 1.25]),
            xaxis=dict(tickfont=dict(size=13)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(bal_fig, use_container_width=True, config={"displayModeBar": False})


if __name__ == "__main__":
    app()
