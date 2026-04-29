# Model Card

## 1. Model Name

**Music Recommender Studio**

---

## 2. Intended Use

VibeMatch suggests songs from a real Spotify catalog based on a user's stated taste profile: preferred genre, mood, target energy, and acoustic preference. It generates a plain-English explanation for each recommendation using Claude, detects when the scoring is over-relying on a single feature, and reports evaluation metrics after every run. It is intended as an educational demonstration of a transparent, explainable AI recommendation pipeline.

---

## 3. How the Model Works

Every song in the catalog receives a weighted score between 0 and 1:

| Feature          | Weight     |
| ---------------- | ---------- |
| Genre match      | 0.30       |
| Mood match       | 0.25       |
| Energy proximity | up to 0.25 |
| Acoustic fit     | up to 0.20 |

Songs are ranked by total score. The top-k results are passed individually to Claude Haiku, which receives the score breakdown, rule traces, and any bias warning as structured context and generates a 2-3 sentence plain-English explanation. If the Claude API is unavailable, a deterministic rule-based explanation is produced instead. After all results are shown, three evaluation metrics are computed: genre diversity, score spread, and feature balance.

---

## 4. Data

The catalog is sourced from the Spotify Tracks Dataset on Kaggle (`maharshipandya/spotify-tracks-dataset`), processed through `data/prepare_spotify.py`. The script samples songs evenly across genres to ensure diversity. Mood is derived from Spotify's `valence` and `energy` audio features using fixed thresholds since Spotify does not provide mood labels directly. The catalog is not redistributed in this repository; users generate it locally using the provided script.

---

## 5. Strengths

The system is fully transparent: every recommendation comes with a weighted score breakdown showing exactly how much each feature contributed. The bias detection layer flags recommendations where a single feature dominates, making over-reliance visible rather than hidden. The Claude explanation grounds its output in the retrieved score data, so users can verify the reasoning against the numbers shown. The fallback ensures the system always produces output regardless of API availability.

---

## 6. Limitations and Bias

Genre is applied as a binary match and carries the single largest weight. A user whose preferred genre is absent from the catalog will never earn that 0.30, structurally capping their maximum possible score compared to users whose genre is well-represented. Mood is approximate: it is derived from valence and energy thresholds rather than human labels, so edge cases exist where the computed mood does not match how a listener would describe the song. The weights are fixed for all users, meaning someone who cares far more about energy than genre has no way to express that. The system also scores each recommendation independently, so the top-k results can include multiple songs by the same artist.

---

## 7. Evaluation

The system is validated through 10 automated tests covering scoring logic, bias detection (flagged and non-flagged cases), fallback explanation generation, and all three evaluation metric functions. Four profiles were used during development to stress-test behavior:

**Ghost Genre (genre=metal):** Metal is not in the catalog. The genre weight is permanently zero, capping the maximum score at 0.70. The bias detection correctly flags these results since energy and acoustic contributions dominate.

**Sad but Hype (genre=pop, mood=sad):** No pop/sad song exists in the catalog, so the mood weight is always wasted. The system returns pop songs ranked by energy and acousticness alone, which the evaluation metrics capture as low genre diversity.

**Acoustic Chaos (genre=folk, energy=0.95):** Folk songs are inherently low energy, so the system recommends a genre/mood match that directly contradicts the energy preference. The feature balance metric shows energy contributing minimally, which surfaces the tension.

**Chill Lofi:** The most consistent profile. Genre, mood, energy, and acousticness all align well with available songs, producing the highest diversity scores and tightest score spread across runs.

---

## 8. Future Work

**Replace the CSV catalog with a live music API.**
The biggest structural limitation is the static catalog. If a genre is missing from the CSV, no song of that genre will ever appear. Integrating the Spotify Web API or Last.fm would allow the system to query a fresh set of candidate tracks for each user profile, score them through the existing pipeline, and return results from millions of real songs. This eliminates the catalog coverage problem entirely and removes the need for users to download and prepare a dataset manually.

**Make weights adjustable per user.**
Fixed weights treat every user identically. Exposing weight sliders in the sidebar would allow users to prioritize energy over genre or mood over acousticness, making the scoring formula genuinely personalized rather than just preference-filtered.

**Add artist and tempo diversity rules.**
The current top-k results can include two songs by the same artist or songs at nearly identical tempos. A post-ranking diversity filter would make recommendations feel less narrow without changing the scoring logic.

**Add collaborative signals.**
Storing anonymized preference profiles over time would allow a lightweight collaborative filtering layer: users with similar genre and mood preferences could receive songs that performed well for similar profiles, adding a behavioral dimension to the currently feature-only scoring.

---

## 9. Reflection and Ethics

**What are the limitations or biases in your system?**

Genre dominance is the clearest structural bias. Because genre is a binary match worth 0.30, it creates a tiered system where users whose preferred genre is well-stocked in the catalog consistently receive better recommendations than users whose genre is rare or absent. This is not a flaw in the algorithm, it is a consequence of how the catalog is populated. The mood derivation adds a second layer of approximation: songs near the valence and energy thresholds can be assigned a mood that does not match listener perception, and that error propagates directly into the scoring.

**Could your AI be misused, and how would you prevent that?**

The explanation module sends song metadata to Claude as part of a prompt. If song titles or artist names in the catalog contained adversarial text designed to manipulate the model's output, the explanation could produce harmful or misleading content. The current mitigations are `html.escape()` on all user-facing text before rendering and a prompt structure that clearly labels song data as context rather than instructions. A stronger mitigation would be validating catalog entries at load time and rejecting rows with unusual formatting or injected syntax.

**What surprised you while testing your AI's reliability?**

The most unexpected finding was how many failures were invisible before diagnostic output was added. The original implementation caught all exceptions silently and returned the rule-based fallback with no indication that Claude had never been called. The app appeared to work correctly across multiple runs. Only after adding `print(f"[Claude] unavailable: {exc}")` did it become clear that the Gemini API key had a quota of zero, and later that the `google-genai` package was not installed in the virtual environment at all. The fallback was indistinguishable from a live response, which made the system appear reliable when it was not. Building systems that fail visibly turned out to be as important as building systems that work correctly.

**Collaboration with AI during this project.**

Claude was used throughout development as a coding collaborator. One instance where the suggestion was genuinely helpful was the progressive rendering approach for the UI. The original implementation froze the interface silently for 10 to 20 seconds while all Claude API calls finished. Claude suggested converting the recommendation builder into a Python generator that yields one result at a time, paired with `st.status` to show live progress. Each card appears on screen as soon as its explanation is ready, which made the system feel responsive without changing any underlying logic.

One instance where the suggestion was flawed was the initial attempt to embed a Plotly chart inside a card using `fig.to_html()` injected into `st.markdown`. The approach generated valid HTML, but Streamlit's markdown renderer strips JavaScript for security reasons, leaving a blank space where the chart should appear. The fix was switching to pure CSS progress bars, which have no JavaScript dependency. The lesson was that a technically correct approach can still fail silently when the execution environment imposes constraints that are not visible from the API surface.
