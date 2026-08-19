# 🎬 MovieScout AI

> Hệ thống tìm kiếm phim ngữ nghĩa thích ứng sử dụng BM25, dense embedding,
> Qdrant, Reciprocal Rank Fusion, HyDE và cross-encoder reranking.

MovieScout AI giúp tìm lại tên phim từ một mô tả không chính xác hoặc không đầy
đủ, chẳng hạn: *“a father travels through a wormhole and communicates with his
daughter through a bookshelf”*. Hệ thống kết hợp lexical search và semantic
search, sau đó dùng một router để quyết định trả kết quả sớm hay chạy pipeline
HyDE + reranking tốn tài nguyên hơn.

Đây là dự án **Information Retrieval / Hybrid Semantic Search**, không phải một
hệ thống RAG hỏi–đáp hoàn chỉnh. HyDE dùng LLM để mở rộng truy vấn, nhưng đầu ra
cuối cùng vẫn là danh sách phim thay vì một câu trả lời được sinh từ tài liệu.

## Mục lục

- [Tính năng](#tính-năng)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Cách pipeline hoạt động](#cách-pipeline-hoạt-động)
- [Công nghệ và mô hình](#công-nghệ-và-mô-hình)
- [Cấu trúc repository](#cấu-trúc-repository)
- [Cài đặt](#cài-đặt)
- [Cấu hình môi trường](#cấu-hình-môi-trường)
- [Chuẩn bị dữ liệu và index](#chuẩn-bị-dữ-liệu-và-index)
- [Chạy ứng dụng](#chạy-ứng-dụng)
- [Chạy bằng Docker](#chạy-bằng-docker)
- [Kiểm thử và chất lượng mã](#kiểm-thử-và-chất-lượng-mã)
- [Đánh giá hệ thống](#đánh-giá-hệ-thống)
- [Bảo mật](#bảo-mật)
- [Giới hạn hiện tại](#giới-hạn-hiện-tại)
- [Roadmap](#roadmap)
- [Xử lý sự cố](#xử-lý-sự-cố)

## Tính năng

- Tìm kiếm phim theo mô tả cốt truyện, chủ đề, diễn viên, đạo diễn hoặc thể loại.
- Kết hợp dense retrieval và BM25 sparse retrieval.
- Gộp hai bảng xếp hạng bằng Reciprocal Rank Fusion (RRF).
- Router EASY/HARD dựa trên độ tự tin của kết quả vòng đầu.
- HyDE tạo một cốt truyện giả định để mở rộng truy vấn khó.
- Cross-encoder reranking đánh giá trực tiếp từng cặp query–movie document.
- Lọc thể loại và khoảng năm phát hành ngay tại Qdrant.
- Hiển thị relevance score, TMDB rating, metadata và thời gian phản hồi.
- Không trộn rating/popularity vào relevance score.
- Streamlit UI, Docker Compose, unit test và GitHub Actions CI.

## Kiến trúc hệ thống

```text
                         OFFLINE INDEXING

 TMDB API
    │
    ▼
 Ingestion ──► Cleaning ──► one movie = one document
                                  │
                     ┌────────────┴────────────┐
                     ▼                         ▼
              Dense embedding           Sparse BM25 vector
                     │                         │
                     └────────────┬────────────┘
                                  ▼
                       Qdrant collection
                  movies_hybrid_collection


                          ONLINE SEARCH

 User query + filters
          │
          ▼
 Query cleaning + dense/sparse encoding
          │
          ├───────────────┐
          ▼               ▼
   Dense retrieval    Sparse retrieval
          └───────┬───────┘
                  ▼
                 RRF
                  │
                  ▼
          Confidence router
             │          │
          EASY          HARD
             │          ├─► HyDE query expansion
             │          ├─► Hybrid retrieval lần 2
             │          └─► Cross-encoder reranking
             └──────┬────┘
                    ▼
          Relevance-only final ranking
                    ▼
               Streamlit UI
```

Mỗi phim được lưu thành đúng **một searchable document** gồm title, director,
cast, genres, keywords và overview. Dự án không chunk nội dung và không dùng
MaxP vì overview phim thường ngắn; thiết kế này cũng giúp một phim tương ứng với
một Qdrant point có UUID xác định, tránh trùng lặp khi index lại.

## Cách pipeline hoạt động

### 1. Thu thập dữ liệu

[`pipeline/ingest.py`](pipeline/ingest.py) lấy dữ liệu từ TMDB theo từng năm,
bao gồm:

- `movie_id`, `title`, `overview`;
- `release_date`, `release_year`;
- `genres`, `director`, top cast và keywords;
- `vote_average`, `popularity`, `poster_path` và `original_language`.

API key được đọc từ biến môi trường `TMDB_API_KEY`, không nằm trong source code.

### 2. Làm sạch và tạo document

[`pipeline/clean.py`](pipeline/clean.py) loại bỏ HTML, URL, khoảng trắng dư và
tạo trường `combined_text`:

```text
Title: Interstellar. Director: Christopher Nolan. Cast: ...
Genres: Adventure, Drama, Science Fiction. Keywords: ...
Overview: A team of explorers travels through a wormhole...
```

`release_year` được chuyển thành integer để Qdrant có thể lọc numeric range.

### 3. Tạo vector và index Qdrant

[`pipeline/dual_embedding_qdrant.py`](pipeline/dual_embedding_qdrant.py) tạo:

- dense vector bằng `all-MiniLM-L6-v2`;
- sparse vector bằng `Qdrant/bm25`;
- payload metadata dùng cho hiển thị và pre-filtering.

Collection mặc định là `movies_hybrid_collection`. Các payload index gồm
`movie_id`, `title`, `genres`, `release_date` và `release_year`.

### 4. Truy hồi lai và RRF

Dense và sparse retrieval chạy song song, mỗi nhánh lấy tối đa 100 kết quả.
[`retrieval/rrf.py`](retrieval/rrf.py) gộp thứ hạng theo công thức:

```text
RRF(d) = Σ 1 / (k + rank_i(d)), với k = 60
```

RRF chỉ sử dụng thứ hạng, do đó không cần hiệu chỉnh trực tiếp hai miền điểm
dense và BM25 khác nhau.

### 5. Adaptive router

Router trong [`retrieval/controller_retrieval.py`](retrieval/controller_retrieval.py)
so sánh RRF score của top 1 và top 2:

- `EASY`: top score đạt mức tối thiểu và score gap lớn hơn ngưỡng; trả kết quả
  vòng đầu để giảm latency và tránh gọi Groq.
- `HARD`: độ tự tin thấp; tạo HyDE document, retrieval lần hai và rerank top 20.

Ngưỡng mặc định hiện tại:

| Tham số | Giá trị | Ý nghĩa |
|---|---:|---|
| `ci_threshold` | `0.01` | Score gap tối thiểu để đi nhánh EASY |
| `min_score_threshold` | `0.03` | Top score tối thiểu trước khi xét EASY |
| `rrf_k` | `60` | Hệ số làm mượt của RRF |

Các ngưỡng này là giá trị thử nghiệm, chưa được xem là tối ưu cho production.

### 6. HyDE và reranking

Nhánh HARD dùng `llama-3.1-8b-instant` qua Groq để sinh một premise ba câu.
HyDE dùng chung `SentenceTransformer` với QueryEncoder để tránh tải hai bản model
vào RAM. Nếu Groq lỗi, hệ thống fallback về chính truy vấn ban đầu.

Cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` sau đó chấm relevance cho
từng cặp query–document. FinalScorer chuẩn hóa và chỉ xếp hạng theo relevance;
TMDB rating chỉ là metadata hiển thị, không thể đẩy một phim nổi tiếng nhưng ít
liên quan lên trên.

## Công nghệ và mô hình

| Thành phần | Công nghệ |
|---|---|
| Ngôn ngữ | Python 3.10+ |
| UI | Streamlit |
| Vector database | Qdrant |
| Dense embedding | `sentence-transformers/all-MiniLM-L6-v2` |
| Sparse embedding | `Qdrant/bm25` qua FastEmbed |
| Rank fusion | Reciprocal Rank Fusion |
| Query expansion | Groq + `llama-3.1-8b-instant` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Data source | TMDB API |
| Test/lint | Pytest, Ruff, GitHub Actions |
| Container | Docker, Docker Compose |

## Cấu trúc repository

```text
moviescout-ai/
├── .github/workflows/ci.yml     # CI: cài package, lint và test
├── evaluation/                  # Benchmark và báo cáo lịch sử
├── pipeline/
│   ├── ingest.py                # Thu thập dữ liệu TMDB
│   ├── clean.py                 # Làm sạch và tạo combined_text
│   └── dual_embedding_qdrant.py # Dense/sparse embedding và upsert
├── retrieval/
│   ├── query.py                 # Làm sạch và encode query
│   ├── dense.py / bm25.py       # Baseline retrievers
│   ├── hybrid.py                # Truy hồi dense + sparse song song
│   ├── rrf.py                   # Reciprocal Rank Fusion
│   ├── aggregate.py             # Document hit → movie candidate
│   ├── hyde.py                  # HyDE qua Groq
│   ├── rerank.py                # Cross-encoder reranking
│   ├── final_scorer.py          # Relevance-only final ranking
│   └── controller_retrieval.py  # Điều phối pipeline và filter
├── tests/                       # Unit tests không gọi API thật
├── ui/app_final.py              # Streamlit UI
├── .env.example                 # Tên các biến môi trường, không chứa key
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

## Cài đặt

### Yêu cầu

- Python 3.10 trở lên;
- Git;
- tài khoản TMDB nếu cần tải lại dữ liệu;
- Qdrant Cloud hoặc Qdrant chạy local;
- Groq API key nếu muốn sử dụng HyDE.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Lần chạy đầu, Sentence Transformers và FastEmbed sẽ tải model về máy nên cần
kết nối Internet và có thể mất vài phút.

## Cấu hình môi trường

Mở `.env` và điền thông tin của bạn:

```dotenv
TMDB_API_KEY=your_tmdb_api_key
GROQ_API_KEY=your_groq_api_key
QDRANT_URL=https://your-cluster.example.com
QDRANT_API_KEY=your_qdrant_api_key
```

| Biến | Bắt buộc khi nào? | Mô tả |
|---|---|---|
| `TMDB_API_KEY` | Chạy ingestion | Key đọc dữ liệu TMDB |
| `GROQ_API_KEY` | Dùng HyDE | Nếu thiếu, HyDE fallback về query gốc |
| `QDRANT_URL` | Index và search | URL Qdrant Cloud hoặc local |
| `QDRANT_API_KEY` | Qdrant yêu cầu auth | Có thể để trống với Qdrant local |

Ví dụ Qdrant local chạy trực tiếp:

```dotenv
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```

## Chuẩn bị dữ liệu và index

Chạy theo đúng thứ tự từ thư mục gốc repository:

```bash
# 1. Tải dữ liệu TMDB → pipeline/data/movies_raw.csv
python -m pipeline.ingest

# 2. Làm sạch → pipeline/data/movies_clean.csv
python -m pipeline.clean

# 3. Tạo vector và upsert vào Qdrant
python -m pipeline.dual_embedding_qdrant
```

Nếu đang dùng collection được tạo bởi phiên bản cũ, cần chạy lại bước 3 để mỗi
payload có `release_year`. Nếu không, filter theo khoảng năm có thể không trả về
kết quả dù phim tồn tại.

> Lưu ý: ingestion mặc định tải nhiều năm và gọi thêm endpoint credits/keywords
> cho từng phim. Quá trình này có thể lâu và sử dụng đáng kể TMDB API quota.

## Chạy ứng dụng

```bash
streamlit run ui/app_final.py
```

Mở [http://localhost:8501](http://localhost:8501), nhập mô tả bằng tiếng Anh,
chọn thể loại và năm nếu cần. Filter năm chấp nhận:

- năm đơn: `2014`;
- khoảng dùng dấu gạch ngang: `2000-2020`;
- khoảng dùng từ `to`: `2000 to 2020`.

UI chỉ thực hiện tìm kiếm khi nhấn **Search**, vì vậy các lần Streamlit rerun do
thao tác giao diện sẽ không tự gọi lại pipeline.

## Chạy bằng Docker

```bash
docker compose up --build
```

Ứng dụng được mở tại [http://localhost:8501](http://localhost:8501), Qdrant tại
`http://localhost:6333`.

Khi app và Qdrant cùng chạy trong Docker Compose, hostname nội bộ của Qdrant là
`qdrant`, không phải `localhost`. Đặt trong `.env`:

```dotenv
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=
```

Sau khi container hoạt động, cần chạy indexing với cùng Qdrant instance trước
khi tìm kiếm. Dữ liệu Qdrant được giữ trong Docker volume `qdrant_data`.

## Kiểm thử và chất lượng mã

Cài nhóm dependency phát triển:

```bash
pip install -e ".[dev]"
```

Chạy unit test và lint:

```bash
pytest -q
ruff check pipeline retrieval ui tests
```

Các test hiện tại kiểm tra:

- text cleaning;
- bảo toàn ký tự Unicode trong query;
- hành vi RRF;
- document aggregation và deduplication;
- rating không làm thay đổi relevance ranking;
- parse năm đơn, khoảng năm và dữ liệu năm không hợp lệ.

Workflow [`.github/workflows/ci.yml`](.github/workflows/ci.yml) chạy các kiểm tra
tương tự trên mỗi push và pull request.

## Đánh giá hệ thống

Repository giữ bộ 200 truy vấn và báo cáo cũ trong [`evaluation/`](evaluation/)
để tham khảo lịch sử. Tuy nhiên, kết quả này **không nên được dùng để tuyên bố
khả năng tổng quát hóa**, vì:

- 200 truy vấn chỉ bao phủ 50 phim;
- mỗi phim được tạo theo bốn mẫu truy vấn cố định;
- một số query gần giống overview đã nằm trong index;
- query theme có thể phù hợp với nhiều phim nhưng chỉ có một nhãn đúng;
- chưa tách development set và locked test set.

Benchmark tiếp theo nên có 250–300 truy vấn được viết/gán nhãn thủ công:

- paraphrase không sao chép overview;
- mô tả thiếu thông tin;
- typo, slang và tên riêng viết sai;
- query tiếng Việt sau khi chuyển sang multilingual models;
- nhiều phim đúng với relevance label từ 0 đến 3;
- development set dùng chỉnh threshold và test set khóa cố định.

### Chỉ số cần báo cáo

| Nhóm | Chỉ số |
|---|---|
| Chất lượng | Recall@K, Precision@K, MRR@10, nDCG@10 |
| Hiệu năng | p50 latency, p95 latency |
| Độ ổn định | API error rate, trung bình và độ lệch qua nhiều lần HyDE |
| Chi phí | Groq calls/query và chi phí ước tính |
| Router | Tỷ lệ EASY/HARD và chất lượng theo từng nhánh |

### Ablation cần thực hiện

| Pipeline | MRR@10 | nDCG@10 | p95 latency | Groq calls |
|---|---:|---:|---:|---:|
| BM25 | TBD | TBD | TBD | 0 |
| Dense | TBD | TBD | TBD | 0 |
| BM25 + Dense + RRF | TBD | TBD | TBD | 0 |
| RRF + Reranker | TBD | TBD | TBD | 0 |
| RRF + HyDE + Reranker | TBD | TBD | TBD | TBD |
| Adaptive Router | TBD | TBD | TBD | TBD |

Không điền số giả vào bảng. Chỉ cập nhật sau khi có script tái lập, cấu hình thí
nghiệm cố định và kết quả chạy thật.

## Bảo mật

- Không commit `.env` hoặc bất kỳ API key nào.
- [`.env.example`](.env.example) chỉ chứa tên biến và placeholder.
- Nếu key từng xuất hiện trong commit, việc xóa khỏi file hiện tại là chưa đủ:
  phải revoke/rotate key và rewrite Git history trước khi public repository.
- Không log header, token hoặc toàn bộ object cấu hình dịch vụ.
- Dùng secrets của nền tảng triển khai thay vì chép key vào Docker image.

## Giới hạn hiện tại

- Dense model và reranker hiện là model tiếng Anh. Query cleaning giữ Unicode,
  nhưng dự án chưa tuyên bố chất lượng truy hồi tiếng Việt.
- Router dùng threshold thử nghiệm và chưa chứng minh được trade-off
  accuracy–latency–cost trên benchmark đáng tin cậy.
- HyDE có tính ngẫu nhiên, phụ thuộc Groq và có thể tăng latency/cost.
- Chưa có bước answer generation nên đây không phải full RAG.
- Chưa có FastAPI backend; Streamlit gọi retrieval pipeline trực tiếp.
- Chưa có live demo hoặc cơ chế feedback “đúng phim/sai phim”.
- Dữ liệu là snapshot TMDB và kế thừa độ thiếu, sai lệch hoặc bias của TMDB.
- Khi chạy lần đầu, thời gian nạp embedding model và reranker có thể đáng kể.

## Roadmap

- [x] Đưa secrets ra khỏi source code.
- [x] Dùng numeric `release_year` cho range filter.
- [x] Tách relevance khỏi TMDB rating.
- [x] Đồng bộ README với thiết kế one-document-per-movie.
- [x] Dùng chung dense encoder giữa QueryEncoder và HyDE.
- [x] Thêm unit tests, CI, Docker và dependency pinning.
- [ ] Xây benchmark thủ công với graded relevance.
- [ ] Viết evaluation runner chung cho baseline và ablation.
- [ ] Hiệu chỉnh hoặc loại bỏ adaptive router dựa trên kết quả thực nghiệm.
- [ ] Đánh giá multilingual embedding và multilingual reranker.
- [ ] Thêm FastAPI `/search` và để Streamlit gọi API.
- [ ] Thêm feedback đúng/sai để thu thập query thực tế.
- [ ] Thêm ảnh/GIF demo và triển khai live demo.
- [ ] Phát hành phiên bản `v1.0.0` sau khi benchmark và deployment ổn định.

## Xử lý sự cố

### `TMDB_API_KEY not found`

Kiểm tra `.env` nằm ở thư mục gốc và có `TMDB_API_KEY`. Khởi động lại process
sau khi sửa biến môi trường.

### Không kết nối được Qdrant

- Xác nhận `QDRANT_URL` và `QDRANT_API_KEY`.
- Với app chạy trên máy: dùng `http://localhost:6333`.
- Với app chạy trong Compose: dùng `http://qdrant:6333`.
- Kiểm tra collection `movies_hybrid_collection` đã được tạo và có dữ liệu.

### Filter năm không trả kết quả

Index lại dữ liệu bằng `python -m pipeline.dual_embedding_qdrant`. Payload từ
phiên bản cũ có thể chưa chứa trường integer `release_year`.

### Lần khởi động đầu rất chậm

Đây thường là lúc tải dense model, sparse model và cross-encoder. Các lần sau sẽ
dùng model cache trên máy.

### HyDE không hoạt động

Kiểm tra `GROQ_API_KEY` và kết nối mạng. Khi Groq lỗi, pipeline vẫn tiếp tục bằng
query gốc nhưng không nhận được lợi ích từ query expansion.

### Hết bộ nhớ

Các model hiện dùng CPU nếu không có CUDA. Đóng process cũ, giảm số worker và
đảm bảo QueryEncoder/HyDE vẫn dùng chung encoder như trong controller hiện tại.

## Ghi nhận dữ liệu

Metadata và poster được lấy từ [The Movie Database (TMDB)](https://www.themoviedb.org/).
Hãy tuân thủ điều khoản sử dụng và yêu cầu attribution của TMDB khi triển khai
demo công khai.
