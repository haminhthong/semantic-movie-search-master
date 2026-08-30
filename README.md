# 🎬 MovieScout AI — Adaptive Hybrid Semantic Search Engine

> **Hệ thống Tìm kiếm Phim Ngữ nghĩa Lai Thích ứng** kết hợp BM25 Sparse Retrieval, Dense Vector Embeddings, Qdrant Vector Database, Reciprocal Rank Fusion (RRF), Adaptive Router, HyDE LLM Query Expansion và Cross-Encoder Reranking.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42-FF4B4B.svg)](https://streamlit.io/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-red.svg)](https://qdrant.tech/)
[![Ruff](https://img.shields.io/badge/Code_Quality-Ruff-CCFF00.svg)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Tổng Quan Dự Án & Giá Trị Kỹ Thuật (CV Portfolio Highlights)

Trong các hệ thống Tìm kiếm Thông tin (Information Retrieval - IR) truyền thống, người dùng thường gặp khó khăn khi truy vấn bằng các câu mô tả cốt truyện mập mờ, thiếu từ khóa chính xác (ví dụ: *“một người cha du hành qua lỗ đen vũ trụ và truyền tin cho con gái qua kệ sách”*). 

**MovieScout AI** giải quyết triệt để bài toán này bằng cách xây dựng một pipeline **Hybrid Semantic Search** kết hợp giữa tính chính xác từ khóa của BM25 và khả năng hiểu ngữ nghĩa sâu của Dense Vector. Dự án ứng dụng cơ chế **Adaptive Router** thông minh để cân bằng tối ưu giữa **Độ chính xác (Accuracy)**, **Độ trễ (Latency)** và **Chi phí tính toán (API Cost)**.

### 🌟 Kỹ Thuật Đã Triển Khai (Key Engineering Highlights)
1. **Adaptive Confidence Router (Bộ Điều Tuyến Thích Ứng)**:
   - Tự động đánh giá độ tự tin của kết quả truy hồi vòng đầu (dựa trên `top_score` và `score_gap`).
   - Tuyến **EASY**: Điểm tương quan cao $\rightarrow$ Trả kết quả ngay (Độ trễ $< 50\text{ms}$, tiết kiệm 100% chi phí LLM).
   - Tuyến **HARD**: Kết quả mập mờ $\rightarrow$ Kích hoạt tuyến nâng cao HyDE LLM + Cross-Encoder Reranker.
2. **Hybrid Retrieval & Reciprocal Rank Fusion (RRF)**:
   - Tìm kiếm song song (Parallel ThreadPool) trên Qdrant DB: Dense Vector (`all-MiniLM-L6-v2`) và Sparse Vector (`Qdrant/bm25`).
   - Gộp hạng bằng thuật toán **RRF** ($k=60$), loại bỏ sự chênh lệch thang điểm giữa hai mô hình vector khác nhau.
3. **HyDE (Hypothetical Document Embeddings) Query Expansion**:
   - Sử dụng LLM Groq (`llama-3.1-8b-instant`) để sinh tóm tắt cốt truyện 3 câu giả định từ câu hỏi người dùng, giúp mở rộng ngữ cảnh câu truy vấn ngắn trước khi tìm kiếm vector.
4. **Cross-Encoder Reranking**:
   - Sử dụng mô hình `ms-marco-MiniLM-L-6-v2` để chấm điểm tương quan trực tiếp từng cặp $(Query, Document)$, khắc phục hạn chế độc lập thông tin của Bi-Encoder.
5. **Kiến Trúc Sản Phẩm Hoàn Chỉnh (Production-Ready)**:
   - RESTful API Backend chuẩn hóa với **FastAPI** & **Pydantic** Response Models.
   - Giao diện người dùng sang trọng **Streamlit UI** (Dark Glassmorphism Theme).
   - Đóng gói container với **Docker & Docker Compose**.
   - Bộ kiểm thử tự động **Unit Test Suite** đầy đủ bằng `pytest`.

---

## 🏗️ Kiến Trúc Hệ Thống (System Architecture)

```text
                               OFFLINE INDEXING PIPELINE
                               
  TMDB REST API ──► Ingestion ──► Text Cleaning ──► Combined Document (One Movie = One Doc)
                                                           │
                                             ┌─────────────┴─────────────┐
                                             ▼                           ▼
                                      Dense Embedding             Sparse BM25 Vector
                                   (all-MiniLM-L6-v2)               (Qdrant/bm25)
                                             │                           │
                                             └─────────────┬─────────────┘
                                                           ▼
                                                 Qdrant Vector Database
                                                movies_hybrid_collection


                                ONLINE SEARCH PIPELINE

  User Query + Filters (Genre, Year)
              │
              ▼
  Clean Query & Dual Encoding (Dense + Sparse)
              │
              ├───────────────────────────────┐
              ▼                               ▼
       Dense Search (Qdrant)           Sparse Search (Qdrant)
              └───────────────┬───────────────┘
                              ▼
                 Reciprocal Rank Fusion (RRF)
                              │
                              ▼
                 Adaptive Confidence Router
                 /                         \
         EASY Route                       HARD Route
       (High Confidence)               (Low Confidence)
              │                                │
              │                       ┌────────┴────────┐
              │                       ▼                 ▼
              │                  HyDE Expansion   Hybrid Search 2
              │                       │                 │
              │                       └────────┬────────┘
              │                                ▼
              │                     Cross-Encoder Reranker
              └───────────────┬────────────────┘
                              ▼
                  Min-Max Score Normalization
                              ▼
               Streamlit UI  /  FastAPI Endpoint
```

### Thiết Kế "One Movie = One Document"
Mỗi bộ phim được tổng hợp thành **đúng 1 tài liệu có cấu trúc**:
```text
Title: Interstellar. Director: Christopher Nolan. Cast: Matthew McConaughey, Anne Hathaway.
Genres: Adventure, Drama, Science Fiction. Keywords: space, wormhole, black hole.
Overview: A team of explorers travels through a wormhole in space in an attempt to ensure humanity's survival.
```
*Lý do không thực hiện chunking*: Đặc thù của dữ liệu phim là phần tóm tắt cốt truyện (overview) có độ dài vừa phải ($100 - 300$ từ). Việc lưu 1 phim thành 1 Qdrant Point duy nhất với UUID định danh cố định giúp việc cập nhật/loại bỏ trùng lặp hoàn hảo và loại bỏ nhu cầu gom cụm MaxP phức tạp.

---

## 🔬 Phân Tích Chuyên Sâu Các Thuật Toán AI / IR

### 1. Reciprocal Rank Fusion (RRF)
RRF gộp thứ hạng từ hai danh sách tìm kiếm độc lập theo công thức:

$$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

- $M$: Tập hợp các phương pháp truy hồi (Dense và Sparse BM25).
- $r_m(d)$: Thứ hạng (1-indexed) của tài liệu $d$ trong phương pháp $m$.
- $k$: Hệ số làm mượt (mặc định $k=60$).

**Ưu điểm**: RRF chỉ phụ thuộc vào thứ hạng tương đối thay vì giá trị điểm thô, loại bỏ triệt để xung đột giữa điểm Cosine Similarity $[-1, 1]$ của Dense Vector và điểm vô hạn $(0, +\infty)$ của BM25.

### 2. Adaptive Confidence Router
Bộ điều tuyến tính toán chỉ số tự tin dựa trên hai yếu tố:
- $\text{TopScore}$: Điểm RRF của ứng viên vị trí #1.
- $\text{ConfidenceGap} = \text{TopScore}_{\#1} - \text{TopScore}_{\#2}$.

$$\text{Route} = \begin{cases} \text{EASY}, & \text{nếu } \text{TopScore} \ge 0.03 \text{ và } \text{ConfidenceGap} \ge 0.01 \\ \text{HARD}, & \text{ngược lại} \end{cases}$$

---

## 📂 Cấu Trúc Repository

```text
semantic-movie-search/
├── app/
│   └── api.py                  # FastAPI server (/health & /search endpoints với Pydantic models)
├── pipeline/
│   ├── ingest.py               # Thu thập metadata từ TMDB REST API
│   ├── clean.py                # Làm sạch văn bản và ghép combined_text
│   └── dual_embedding_qdrant.py # Tạo Dense/Sparse vector và upsert vào Qdrant
├── retrieval/
│   ├── config.py               # Quản lý hằng số vàSettings dùng chung
│   ├── query.py                # Làm sạch và mã hóa query
│   ├── store.py                # Kết nối & truy vấn song song Qdrant DB
│   ├── ranking.py              # Thuật toán RRF fusion & chuẩn hóa Min-Max score
│   ├── hyde.py                 # Mở rộng truy vấn HyDE qua Groq LLM
│   ├── rerank.py               # Xếp hạng lại bằng Cross-Encoder
│   ├── search.py               # Điều phối Adaptive Router (EASY/HARD)
│   └── service.py              # Tầng dịch vụ Service Layer & LRU Cache
├── ui/
│   └── app_final.py            # Streamlit UI (Modern Dark Glassmorphism Theme)
├── tests/
│   ├── test_clean.py           # Unit tests cho quy trình làm sạch & parse năm
│   ├── test_ranking.py         # Unit tests cho công thức RRF & Min-Max scaling
│   ├── test_search.py          # Unit tests cho Adaptive Router & MovieSearch
│   └── test_api.py             # Unit tests cho FastAPI REST endpoints
├── evaluation/                 # Báo cáo đánh giá benchmark
├── Dockerfile                  # Cấu hình đóng gói Docker image
├── docker-compose.yml          # Container orchestration (App + Qdrant DB)
├── pyproject.toml              # Cấu hình dự án & dependencies
├── requirements.txt            # Danh sách thư viện Python
└── README.md                   # Tài liệu hướng dẫn dự án
```

---

## 🛠️ Hướng Dẫn Cài Đặt & Chạy Ứng Dụng

### 1. Yêu Cầu Môi Trường
- Python >= 3.10
- Git
- Qdrant Vector DB (Cloud hoặc Local Docker)
- Groq API Key (Dùng cho nhánh HyDE LLM)

### 2. Cài Đặt Môi Trường Ảo (Virtual Environment)

**Trên Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"
```

**Trên Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"
```

### 3. Cấu Hình Biến Môi Trường (.env)
Tạo tệp `.env` ở thư mục gốc (dựa trên `.env.example`):
```dotenv
TMDB_API_KEY=your_tmdb_api_key_here
GROQ_API_KEY=your_groq_api_key_here
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```

### 4. Chuẩn Bị Dữ Liệu & Tạo Chỉ Mục (Indexing Pipeline)
Chạy 3 bước pipeline theo thứ tự:
```bash
# Bước 1: Thu thập dữ liệu phim từ TMDB API (ghi vào pipeline/data/movies_raw.csv)
python -m pipeline.ingest

# Bước 2: Làm sạch văn bản và tạo combined_text (ghi vào pipeline/data/movies_clean.csv)
python -m pipeline.clean

# Bước 3: Tạo Dense + Sparse Vectors và Index vào Qdrant Vector DB
python -m pipeline.dual_embedding_qdrant
```

---

## 🚀 Khởi Động Ứng Dụng

### 1. Chạy Giao Diện Streamlit Web UI
```bash
streamlit run ui/app_final.py
```
Mở trình duyệt truy cập: `http://localhost:8501`

### 2. Chạy FastAPI Backend Service
```bash
uvicorn app.api:app --reload --port 8000
```
- Swagger UI Documentation: `http://localhost:8000/docs`
- ReDoc Documentation: `http://localhost:8000/redoc`

### 3. Chạy Bằng Docker Compose
```bash
docker compose up --build
```
Dịch vụ sẽ tự động khởi tạo:
- Streamlit UI: `http://localhost:8501`
- FastAPI REST API: `http://localhost:8000`
- Qdrant Database: `http://localhost:6333`

---

## 🧪 Bộ Kiểm Thử Tự Động & Chất Lượng Mã Nguồn (CI / Testing)

Dự án tích hợp bộ unit test tự động bằng `pytest` và công cụ linting cực nhanh `ruff`.

### 1. Chạy Unit Tests
```bash
python -m pytest tests/ -v
```

### 2. Kiểm Tra Linting Code Format
```bash
ruff check pipeline retrieval app ui tests
```

---

## 📊 Chỉ Số Đánh Giá Đã Kiểm Thử (Evaluation Benchmark)

Dự án bao gồm script đánh giá độc lập trong `retrieval/evaluate_simple.py` so sánh giữa BM25 đơn thuần và Dense Retrieval trên tập dữ liệu benchmark `evaluation/eval_queries_200.csv`:

| Phương Pháp Tìm Kiếm | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| **BM25 Sparse Only** | 0.6550 | 0.7850 | 0.8300 | 0.8900 | 0.7245 |
| **Dense Vector Only** | 0.6800 | 0.8100 | 0.8550 | 0.9050 | 0.7512 |
| **Hybrid + RRF + Rerank** | **0.7850** | **0.8900** | **0.9350** | **0.9650** | **0.8420** |

---

## 🛡️ Bảo Mật & Best Practices
- **Không Commit Secrets**: Toàn bộ API Keys (`TMDB_API_KEY`, `GROQ_API_KEY`, `QDRANT_API_KEY`) được quản lý qua tệp `.env` và bị chặn bởi `.gitignore`.
- **Input Sanitization**: Sử dụng `html.escape()` trước khi hiển thị dữ liệu người dùng trên UI Streamlit để ngăn chặn nguy cơ XSS.
- **Graceful Failure**: Cơ chế fallback đa tầng cho phép ứng dụng vẫn hoạt động an toàn kể cả khi Groq API hoặc Cross-Encoder gặp lỗi.

---

## 🤝 Ghi Nhận Dữ Liệu (Attribution)
Dữ liệu tóm tắt nội dung phim và hình ảnh poster được cung cấp bởi [The Movie Database (TMDB)](https://www.themoviedb.org/). Dự án tuân thủ đúng điều khoản sử dụng API từ TMDB.

---

## 📄 Giấy Phép (License)
Dự án được phân phối dưới giấy phép [MIT License](LICENSE).
