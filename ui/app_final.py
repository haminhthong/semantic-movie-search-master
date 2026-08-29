"""Giao diện Streamlit của MovieScout AI."""

import html
import logging

import streamlit as st

from retrieval.service import SearchService

logger = logging.getLogger(__name__)

st.set_page_config(page_title="MovieScout AI", page_icon="🎬", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background: #0B0F1A}
    .title {text-align: center; color: #22D3EE}
    .movie {background: #111827; border: 1px solid #4c1d95; border-radius: 14px;
            padding: 18px; margin: 12px 0}
    .meta {color: #94A3B8}
    .badge {background: #7C3AED; color: white; padding: 4px 10px; border-radius: 12px}
    </style>
    """,
    unsafe_allow_html=True,
)


def parse_movie_info(raw_text: str):
    """Tách đạo diễn, diễn viên và overview từ tài liệu đã ghép."""
    text = raw_text if isinstance(raw_text, str) else ""
    director, cast, plot = "Unknown", "Unknown", text
    if "Director:" in text and ". Cast:" in text:
        director = text.split("Director:", 1)[1].split(". Cast:", 1)[0].strip()
    if "Cast:" in text and ". Genres:" in text:
        cast = text.split("Cast:", 1)[1].split(". Genres:", 1)[0].strip()
    if "Overview:" in text:
        plot = text.split("Overview:", 1)[1].strip()
    return director, cast, plot


def escaped(value) -> str:
    """Chuyển giá trị sang chuỗi an toàn trước khi chèn vào HTML."""
    return html.escape(str(value), quote=True)


@st.cache_resource(show_spinner="Starting the search engine…")
def init_engine() -> SearchService:
    """Nạp model một lần cho mỗi tiến trình Streamlit."""
    return SearchService()


def render_movie(rank: int, movie: dict) -> None:
    """Hiển thị một kết quả tìm kiếm."""
    director, cast, plot = parse_movie_info(movie.get("document", {}).get("text", ""))
    year = movie.get("release_year") or str(movie.get("release_date", ""))[:4] or "N/A"
    st.markdown(
        f"<div class='movie'><h3>#{rank} {escaped(movie.get('title', 'Unknown'))} "
        f"({escaped(year)})</h3>"
        f"<span class='badge'>Relevance {float(movie.get('final_score', 0.0)):.4f}</span> "
        f"<span class='meta'>TMDB rating: {escaped(movie.get('vote_average', 0))}/10</span>"
        f"<p><b>Director:</b> {escaped(director)}<br>"
        f"<b>Cast:</b> {escaped(cast)}<br>"
        f"<b>Overview:</b> {escaped(plot)}</p></div>",
        unsafe_allow_html=True,
    )


st.markdown("<h1 class='title'>🎬 MovieScout AI</h1>", unsafe_allow_html=True)
st.caption("Hybrid semantic movie search — describe a plot and find the title.")

genres = [
    "All", "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
    "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery",
    "Romance", "Science Fiction", "Thriller", "War", "Western",
]
with st.form("search_form"):
    left, right = st.columns(2)
    genre = left.selectbox("Genre", genres)
    year = right.text_input("Release year", placeholder="2014 or 2000-2020")
    query = st.text_input(
        "Movie description",
        placeholder="A man trapped inside a video game…",
    )
    submitted = st.form_submit_button("Search", use_container_width=True)

if submitted:
    if not query.strip():
        st.warning("Enter a movie description first.")
        st.stop()
    try:
        engine = init_engine()
        with st.spinner("Searching…"):
            response = engine.search(
                query,
                top_n=10,
                genre=genre,
                year=year,
            )
        results = response["movies"]
        route = response["route"]
        hypothetical = response["hyde"]
        if not results:
            st.info("No matching movies were found.")
        else:
            st.success(f"Found {len(results)} movies in {response['latency_ms']:.0f}ms · route: {route}")
            if route == "HARD" and hypothetical:
                with st.expander("HyDE query expansion"):
                    st.write(hypothetical)
            for rank, movie in enumerate(results, start=1):
                render_movie(rank, movie)
    except ValueError as exc:
        st.warning(str(exc))
    except Exception:
        logger.exception("Tìm kiếm gặp lỗi")
        st.error("The search service is unavailable. Check the service configuration and try again.")
