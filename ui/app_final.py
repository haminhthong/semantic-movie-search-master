"""Giao diện web tương tác (Streamlit Application) của MovieScout AI.

Cung cấp trải nghiệm tìm kiếm phim hiện đại với phong cách thiết kế Dark Glassmorphism,
hiển thị trực quan chỉ số phân tuyến thích ứng (Adaptive Route Badge), độ trễ phản hồi (Latency),
thanh điểm tương quan (Relevance Progress Bar) và chi tiết metadata của bộ phim.
"""

import html
import logging
from typing import Any

import streamlit as st

from retrieval.service import SearchService

logger = logging.getLogger(__name__)

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="MovieScout AI — Hybrid Semantic Search Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS cho phong cách Modern Dark Theme & Glassmorphism
st.markdown(
    """
    <style>
    /* Nền chính ứng dụng */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Tiêu đề ứng dụng */
    .main-title {
        font-size: 2.75rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Thẻ hiển thị bộ phim (Glassmorphism Card) */
    .movie-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .movie-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.25);
        border-color: rgba(129, 140, 248, 0.3);
    }

    .movie-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 8px;
    }

    /* Đánh dấu badge tuyến xử lý */
    .badge-easy {
        background: linear-gradient(135deg, #059669, #10b981);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .badge-hard {
        background: linear-gradient(135deg, #7c3aed, #a855f7);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .badge-score {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .meta-text {
        color: #94a3b8;
        font-size: 0.92rem;
        line-height: 1.6;
    }

    .meta-highlight {
        color: #e2e8f0;
        font-weight: 600;
    }

    /* Thanh điểm tương quan Progress Bar custom */
    .score-bar-bg {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        height: 8px;
        width: 100%;
        margin-top: 8px;
        margin-bottom: 12px;
        overflow: hidden;
    }

    .score-bar-fill {
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        height: 100%;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def parse_movie_info(raw_text: str) -> tuple[str, str, str]:
    """Trích xuất đạo diễn, diễn viên và tóm tắt cốt truyện từ tài liệu combined_text.

    Args:
        raw_text: Chuỗi combined_text thô ghép từ metadata.

    Returns:
        Tuple[str, str, str]: (Đạo diễn, Diễn viên, Overview cốt truyện).
    """
    text = raw_text if isinstance(raw_text, str) else ""
    director, cast, plot = "Chưa rõ", "Chưa rõ", text

    if "Director:" in text and ". Cast:" in text:
        director = text.split("Director:", 1)[1].split(". Cast:", 1)[0].strip()
    if "Cast:" in text and ". Genres:" in text:
        cast = text.split("Cast:", 1)[1].split(". Genres:", 1)[0].strip()
    if "Overview:" in text:
        plot = text.split("Overview:", 1)[1].strip()

    return director, cast, plot


def safe_escape(value: Any) -> str:
    """Mã hóa chuỗi an toàn trước khi chèn vào HTML để tránh lỗi XSS injection."""
    return html.escape(str(value), quote=True)


@st.cache_resource(show_spinner="Đang khởi tạo các mô hình AI và kết nối Qdrant...")
def init_engine() -> SearchService:
    """Tải và khởi tạo đệm cho dịch vụ MovieSearch duy nhất trong một tiến trình Streamlit."""
    return SearchService()


def render_movie(rank: int, movie: dict[str, Any]) -> None:
    """Hiển thị một thẻ thông tin kết quả phim sang trọng."""
    director, cast, plot = parse_movie_info(movie.get("document", {}).get("text", ""))
    year = movie.get("release_year") or str(movie.get("release_date", ""))[:4] or "N/A"
    final_score = float(movie.get("final_score", 0.0))
    score_pct = int(final_score * 100)
    title = movie.get("title", "Không rõ tên")
    vote_avg = movie.get("vote_average", 0)
    poster_path = movie.get("poster_path", "")

    # Tạo cột hiển thị ảnh poster (nếu có) và thông tin chi tiết
    col_poster, col_info = st.columns([1, 4]) if poster_path else (None, None)

    card_html = f"""
    <div class="movie-card">
        <div class="movie-title">
            #{rank}. {safe_escape(title)} <span style="color: #94a3b8; font-size: 1rem; font-weight: normal;">({safe_escape(year)})</span>
        </div>
        <div style="margin-bottom: 10px;">
            <span class="badge-score">Relevance: {final_score:.4f}</span>
            <span style="color: #cbd5e1; font-size: 0.88rem; margin-left: 12px;">⭐ TMDB Rating: <b style="color: #facc15;">{safe_escape(vote_avg)}/10</b></span>
            <span style="color: #94a3b8; font-size: 0.88rem; margin-left: 12px;">🎭 {safe_escape(movie.get("genres", ""))}</span>
        </div>
        <div class="score-bar-bg">
            <div class="score-bar-fill" style="width: {score_pct}%;"></div>
        </div>
        <div class="meta-text">
            <p style="margin-bottom: 6px;"><span class="meta-highlight">🎬 Đạo diễn:</span> {safe_escape(director)}</p>
            <p style="margin-bottom: 6px;"><span class="meta-highlight">👥 Diễn viên:</span> {safe_escape(cast)}</p>
            <p style="margin-bottom: 0;"><span class="meta-highlight">📖 Cốt truyện:</span> {safe_escape(plot)}</p>
        </div>
    </div>
    """

    if poster_path and col_poster and col_info:
        with col_poster:
            poster_url = f"https://image.tmdb.org/t500{poster_path}"
            st.image(poster_url, use_container_width=True)
        with col_info:
            st.markdown(card_html, unsafe_allow_html=True)
    else:
        st.markdown(card_html, unsafe_allow_html=True)


# Header ứng dụng
st.markdown("<div class='main-title'>🎬 MovieScout AI</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Hệ thống tìm kiếm phim ngữ nghĩa thích ứng — BM25 + Dense Retrieval + RRF + HyDE + Reranking</div>",
    unsafe_allow_html=True,
)

# Thư viện thể loại phim
genres_list = [
    "All",
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "History",
    "Horror",
    "Music",
    "Mystery",
    "Romance",
    "Science Fiction",
    "Thriller",
    "War",
    "Western",
]

# Sidebar thông tin dự án
with st.sidebar:
    st.header("⚙️ Cấu Hình & Thông Tin")
    st.info(
        "**MovieScout AI** tự động điều hướng truy vấn:\n"
        "- 🟢 **EASY Route**: Độ tự tin cao -> Trả kết quả ngay (Latencies thấp).\n"
        "- 🟣 **HARD Route**: Truy vấn phức tạp -> Mở rộng HyDE LLM & Rerank."
    )
    top_n_slider = st.slider("Số kết quả hiển thị (top_n)", min_value=1, max_value=20, value=10)
    st.markdown("---")
    st.markdown("<b>Công nghệ sử dụng:</b>", unsafe_allow_html=True)
    st.caption("• Vector DB: Qdrant Cloud / Local")
    st.caption("• Dense Model: sentence-transformers/all-MiniLM-L6-v2")
    st.caption("• Sparse Model: FastEmbed Qdrant/bm25")
    st.caption("• Query Expansion: Groq Llama-3.1-8b HyDE")
    st.caption("• Reranker: ms-marco-MiniLM-L-6-v2")

# Form tìm kiếm
with st.form("search_form"):
    col_genre, col_year = st.columns(2)
    genre_selected = col_genre.selectbox("Thể loại phim", genres_list)
    year_input = col_year.text_input("Năm sản xuất", placeholder="Ví dụ: 2014 hoặc 2000-2020")

    query_input = st.text_input(
        "Mô tả nội dung cốt truyện bộ phim cần tìm",
        placeholder="Mô tả ý tưởng, ví dụ: 'A team of explorers travels through a wormhole in space to save humanity...'",
    )
    submit_button = st.form_submit_button("🔍 Tìm Kiếm Phim", use_container_width=True)

# Xử lý sự kiện Submit
if submit_button:
    if not query_input.strip():
        st.warning("⚠️ Vui lòng nhập mô tả cốt truyện phim trước khi bấm Tìm Kiếm.")
        st.stop()

    try:
        engine = init_engine()
        with st.spinner("🚀 Đang truy hồi vector và tính toán độ tương quan ngữ nghĩa..."):
            response = engine.search(
                query=query_input,
                top_n=top_n_slider,
                genre=genre_selected,
                year=year_input,
            )

        movies = response["movies"]
        route = response["route"]
        hyde_text = response["hyde"]
        latency = response["latency_ms"]

        if not movies:
            st.info("💡 Không tìm thấy bộ phim nào phù hợp với điều kiện lọc và mô tả của bạn.")
        else:
            # Hiển thị các Metrics
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Số kết quả tìm thấy", f"{len(movies)} phim")
            m_col2.metric("Thời gian phản hồi", f"{latency:.2f} ms")
            m_col3.metric("Tuyến xử lý (Route)", f"{route} Route")

            # Hiển thị badge tuyến
            if route == "EASY":
                st.markdown(
                    "<span class='badge-easy'>🟢 EASY ROUTE: Kết quả có độ tự tin cao từ RRF vòng 1</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<span class='badge-hard'>🟣 HARD ROUTE: Kích hoạt HyDE LLM & Cross-Encoder Reranking</span>",
                    unsafe_allow_html=True,
                )

            st.write("")

            # Nếu là HARD Route và có HyDE text, cho phép mở rộng xem
            if route == "HARD" and hyde_text:
                with st.expander("📄 Xem văn bản tóm tắt giả định được sinh bởi HyDE (LLM Query Expansion)"):
                    st.write(f"_{hyde_text}_")

            st.write("### 🍿 Danh Sách Phim Tương Quan Nhất:")
            for rank, movie in enumerate(movies, start=1):
                render_movie(rank, movie)

    except ValueError as exc:
        st.warning(f"⚠️ Tham số không hợp lệ: {exc}")
    except Exception:
        logger.exception("Tìm kiếm gặp sự cố hệ thống")
        st.error(
            "❌ Dịch vụ tìm kiếm tạm thời không sẵn sàng. Vui lòng kiểm tra lại kết nối Qdrant/API Key và thử lại."
        )
