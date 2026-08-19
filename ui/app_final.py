"""Streamlit UI for MovieScout AI."""
import html
import time
import streamlit as st
from retrieval.controller_retrieval import AdaptiveSearchPipeline

st.set_page_config(page_title="MovieScout AI", page_icon="🎬", layout="wide")
st.markdown("""<style>
.stApp{background:#0B0F1A}.title{text-align:center;color:#22D3EE}
.movie{background:#111827;border:1px solid #4c1d95;border-radius:14px;padding:18px;margin:12px 0}
.meta{color:#94A3B8}.badge{background:#7C3AED;color:white;padding:4px 10px;border-radius:12px}
</style>""", unsafe_allow_html=True)

def parse_movie_info(raw_text):
    director, cast, plot = "Unknown", "Unknown", raw_text
    if "Director:" in raw_text and ". Cast:" in raw_text:
        director = raw_text.split("Director:", 1)[1].split(". Cast:", 1)[0].strip()
    if "Cast:" in raw_text and ". Genres:" in raw_text:
        cast = raw_text.split("Cast:", 1)[1].split(". Genres:", 1)[0].strip()
    if "Overview:" in raw_text:
        plot = raw_text.split("Overview:", 1)[1].strip()
    return director, cast, plot

@st.cache_resource(show_spinner="Starting the search engine…")
def init_engine():
    return AdaptiveSearchPipeline()

st.markdown("<h1 class='title'>🎬 MovieScout AI</h1>", unsafe_allow_html=True)
st.caption("Hybrid semantic movie search — describe a plot and find the title.")

genres = ["All", "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery", "Romance", "Science Fiction", "Thriller", "War", "Western"]
with st.form("search_form"):
    left, right = st.columns(2)
    genre = left.selectbox("Genre", genres)
    year = right.text_input("Release year", placeholder="2014 or 2000-2020")
    query = st.text_input("Movie description", placeholder="A man trapped inside a video game…")
    submitted = st.form_submit_button("Search", use_container_width=True)

if submitted:
    if not query.strip():
        st.warning("Enter a movie description first.")
        st.stop()
    filters = {"genre": genre, "year": year}
    try:
        engine = init_engine()
        started = time.perf_counter()
        with st.spinner("Searching…"):
            results, route, hypothetical = engine.search(query, top_n=10, user_filters=filters)
        st.success(f"Found {len(results)} movies in {time.perf_counter()-started:.2f}s · route: {route}")
        if route == "HARD" and hypothetical:
            with st.expander("HyDE query expansion"):
                st.write(hypothetical)
        for rank, movie in enumerate(results, 1):
            text = movie.get("document", {}).get("text", "")
            director, cast, plot = parse_movie_info(text)
            title = html.escape(str(movie.get("title", "Unknown")))
            st.markdown(
                f"<div class='movie'><h3>#{rank} {title} ({html.escape(str(movie.get('release_year') or str(movie.get('release_date',''))[:4]))})</h3>"
                f"<span class='badge'>Relevance {movie.get('final_score',0):.4f}</span> "
                f"<span class='meta'>TMDB rating: {movie.get('vote_average',0)}/10</span>"
                f"<p><b>Director:</b> {html.escape(director)}<br><b>Cast:</b> {html.escape(cast)}<br>"
                f"<b>Overview:</b> {html.escape(plot)}</p></div>", unsafe_allow_html=True)
    except ValueError as exc:
        st.warning(str(exc))
    except Exception:
        st.error("The search service is temporarily unavailable. Check Qdrant/Groq configuration and try again.")
