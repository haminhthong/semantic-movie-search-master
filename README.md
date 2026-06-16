# 🎬 MovieScout AI: Hệ thống Tìm kiếm Phim Ngữ nghĩa (Advanced RAG)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20Database-red)
![Groq](https://img.shields.io/badge/Groq-Llama--3.1-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B)

**MovieScout AI** là một hệ thống tìm kiếm thông tin điện ảnh ứng dụng kiến trúc **Retrieval-Augmented Generation (RAG) Nâng cao**. Thay vì tìm kiếm theo từ khóa (Keyword matching) truyền thống, hệ thống cho phép người dùng tìm kiếm phim dựa trên **ngữ nghĩa, bối cảnh, hoặc những miêu tả cốt truyện mơ hồ** (Ví dụ: *"phim khoa học viễn tưởng về không gian có cảnh bố để lại đồng hồ cho con gái"*).

Dự án được xây dựng với mục tiêu giải quyết bài toán "bất đồng ranh giới từ vựng" (Vocabulary Mismatch) trong Information Retrieval (IR), tối ưu hóa tốc độ tìm kiếm dưới 2 giây với độ chính xác cao nhờ kiến trúc đa tầng thích ứng.

---

## 🛠️ Cấu trúc thư mục & Giải thích các file Code

Dưới đây là chi tiết chức năng của từng tệp tin trong mã nguồn. Bạn có thể nhấn trực tiếp vào liên kết của từng file để xem nội dung:

### 1. Cấu hình hệ thống (Root)
*   [requirements.txt](file:///d:/hoc/semantic-movie-search-master/requirements.txt): Định nghĩa toàn bộ các thư viện Python cần thiết cho dự án (được cập nhật chuẩn hóa các thư viện thực tế sử dụng như `fastembed`, `qdrant-client`, `groq`, `sentence-transformers`, `torch`, `tqdm`).
*   [.env](file:///d:/hoc/semantic-movie-search-master/.env): Lưu trữ thông tin cấu hình và khóa bí mật API (Qdrant Cloud, Groq Llama, TMDB API, v.v.).
*   [.gitignore](file:///d:/hoc/semantic-movie-search-master/.gitignore): Quy định các thư mục/file không được đẩy lên Git như môi trường ảo, cache Python (`__pycache__`), các file CSV dữ liệu tải về, và tệp cấu hình bảo mật `.env`.

### 2. Offline Pipeline (`pipeline/`): Thu thập và Tiền xử lý dữ liệu
*   [pipeline/ingest.py](file:///d:/hoc/semantic-movie-search-master/pipeline/ingest.py): Thu thập dữ liệu phim từ TMDB API (lọc phim từ năm 1990-2026 có rating > 6.5 và vote count >= 100). Sau đó truy vấn chi tiết phim để lấy thông tin đạo diễn, top 5 diễn viên, từ khóa (keywords) và lưu thành file thô `pipeline/data/movies_raw.csv`.
*   [pipeline/clean.py](file:///d:/hoc/semantic-movie-search-master/pipeline/clean.py): Làm sạch dữ liệu văn bản (loại bỏ thẻ HTML, liên kết URL, chuẩn hóa khoảng trắng, giữ nguyên từ dừng - stopwords) và ghép các trường dữ liệu thành một chuỗi duy nhất `combined_text` định dạng: `Title: ... Director: ... Cast: ... Genres: ... Keywords: ... Overview: ...`. Lưu kết quả vào `pipeline/data/movies_clean.csv`.
*   [pipeline/dual_embedding_qdrant.py](file:///d:/hoc/semantic-movie-search-master/pipeline/dual_embedding_qdrant.py): Nhúng văn bản phim thành hai dạng: **Dense Vector** (sử dụng model `all-MiniLM-L6-v2` của `sentence-transformers`) và **Sparse Vector** (sử dụng BM25 qua thư viện `fastembed`). Sau đó đẩy đồng thời (Hybrid Upsert) lên Vector Database **Qdrant Cloud** với ID dạng UUID v5 chống trùng lặp dữ liệu.
*   [pipeline/batch_update.py](file:///d:/hoc/semantic-movie-search-master/pipeline/batch_update.py): File trống (placeholder) dùng để mở rộng cập nhật dữ liệu theo lô trong tương lai.

### 3. Online Retrieval Pipeline (`retrieval/`): Công cụ Truy hồi & Định tuyến
*   [retrieval/query.py](file:///d:/hoc/semantic-movie-search-master/retrieval/query.py): Tiếp nhận câu truy vấn thô của người dùng, tiền xử lý làm sạch và mã hóa câu hỏi thành cả hai dạng Dense Vector và Sparse Vector.
*   [retrieval/bm25.py](file:///d:/hoc/semantic-movie-search-master/retrieval/bm25.py): Thực hiện tìm kiếm từ khóa thuần túy (Lexical Search) sử dụng Sparse Vector trên cơ sở dữ liệu Qdrant.
*   [retrieval/dense.py](file:///d:/hoc/semantic-movie-search-master/retrieval/dense.py): Thực hiện tìm kiếm ngữ nghĩa thuần túy (Semantic Search) sử dụng Dense Vector trên cơ sở dữ liệu Qdrant.
*   [retrieval/hybrid.py](file:///d:/hoc/semantic-movie-search-master/retrieval/hybrid.py): Thực hiện tìm kiếm lai (Hybrid Search) chạy song song (Parallel execution qua `ThreadPoolExecutor`) cả Dense và Sparse Search để tối ưu tốc độ phản hồi.
*   [retrieval/rrf.py](file:///d:/hoc/semantic-movie-search-master/retrieval/rrf.py): Thuật toán **Reciprocal Rank Fusion (RRF)** ghép điểm và sắp xếp lại thứ hạng của các phân đoạn phim từ hai luồng tìm kiếm lai với tham số chuẩn hóa $k = 60$.
*   [retrieval/aggregate.py](file:///d:/hoc/semantic-movie-search-master/retrieval/aggregate.py): Sử dụng thuật toán **MaxP (Maximum Passage Pooling)** để gộp các đoạn văn bản khớp về mức độ phim và áp dụng bộ nhân logarithm (Logarithmic Boost) giúp tăng điểm cho các phim có nhiều đoạn khớp.
*   [retrieval/hyde.py](file:///d:/hoc/semantic-movie-search-master/retrieval/hyde.py): Sử dụng mô hình Llama-3.1-8b (via Groq Cloud) sinh ra một "cốt truyện giả định" (Hypothetical Document) từ mô tả ngắn của người dùng để mở rộng từ vựng truy vấn. Có tích hợp cache nội bộ giúp giảm tần suất gọi API.
*   [retrieval/rerank.py](file:///d:/hoc/semantic-movie-search-master/retrieval/rerank.py): Sử dụng mô hình Cross-Encoder chuyên sâu `ms-marco-MiniLM-L-6-v2` để tính điểm tương quan chéo giữa câu hỏi và nội dung đầy đủ của phim, tăng độ chính xác xếp hạng.
*   [retrieval/final_scorer.py](file:///d:/hoc/semantic-movie-search-master/retrieval/final_scorer.py): Tính điểm số chung cuộc bằng cách kết hợp điểm ngữ nghĩa từ mô hình AI (trọng số 80%) và điểm chất lượng phim của TMDB (vote average, trọng số 20%) để lọc ra các phim xuất sắc nhất.
*   [retrieval/controller_retrieval.py](file:///d:/hoc/semantic-movie-search-master/retrieval/controller_retrieval.py): Bộ điều khiển trung tâm **AdaptiveSearchPipeline**. Nó điều phối toàn bộ luồng tìm kiếm:
    *   **Vòng 1:** Thực hiện tìm kiếm lai (Hybrid Search) đầu tiên và áp dụng bộ lọc Pre-filtering (Thể loại, Năm phát hành) trực tiếp trên Qdrant.
    *   **Difficulty Router (Định tuyến độ khó):** Đo khoảng cách điểm (Score Gap) giữa phim top 1 và top 2.
        *   Nếu khoảng cách $\ge$ ngưỡng cài đặt (`ci_threshold`): Xác định là truy vấn dễ (**EASY**), lập tức dừng và trả kết quả (**Early Exit**).
        *   Nếu khoảng cách bé hoặc điểm số quá thấp: Xác định là truy vấn khó (**HARD**), kích hoạt luồng **HyDE (LLM)** để mở rộng câu hỏi $\rightarrow$ Tìm kiếm lai vòng 2 $\rightarrow$ Reranker (Cross-Encoder) $\rightarrow$ Trả kết quả.
*   [retrieval/evaluate_simple.py](file:///d:/hoc/semantic-movie-search-master/retrieval/evaluate_simple.py): Script đánh giá, so sánh hiệu năng tìm kiếm độc lập giữa BM25 và Dense Retrieval trên 200 câu test thực tế để kiểm chứng độ chính xác.

### 4. Giao diện người dùng (`ui/`)
*   [ui/app_final.py](file:///d:/hoc/semantic-movie-search-master/ui/app_final.py): Giao diện web được xây dựng bằng Streamlit với thiết kế tối giản, hiện đại (Ultra Premium Dark Theme), hiển thị thời gian phản hồi thực tế, cơ chế định tuyến (EASY/HARD), biểu diễn cốt truyện ảo do HyDE sinh ra, và danh sách phim kèm poster chất lượng cao.

### 5. Thống kê & Đánh giá (`evaluation/`)
*   Các file báo cáo chi tiết như [evaluation/EVALUATION_SUMMARY.md](file:///d:/hoc/semantic-movie-search-master/evaluation/EVALUATION_SUMMARY.md) và kết quả chạy thử nghiệm lưu trữ trong thư mục này.

---

## 🚀 Hướng dẫn thiết lập và Chạy ứng dụng

### 1. Yêu cầu hệ thống
*   Cài đặt Python phiên bản 3.10 trở lên.
*   Kết nối mạng ổn định (để giao tiếp với API Qdrant Cloud và Groq).

### 2. Thiết lập Môi trường & Thư viện
Mở Terminal/PowerShell tại thư mục gốc của dự án và chạy lệnh:

```bash
# Cài đặt toàn bộ thư viện yêu cầu
pip install -r requirements.txt
```

### 3. Cấu hình Khóa API
Mở tệp [.env](file:///d:/hoc/semantic-movie-search-master/.env) ở thư mục gốc và cung cấp các khóa API cần thiết:
```env
TMDB_API_KEY="your_tmdb_api_key_here"
GROQ_API_KEY="your_groq_api_key_here"
QDRANT_URL="https://your-qdrant-cluster-url.io"
QDRANT_API_KEY="your_qdrant_api_key_here"
```
*(Hiện tại dự án đã cấu hình sẵn một bộ key dùng thử trực tuyến kết nối với database Qdrant của lớp học).*

---

## 🏃 Quy trình thực thi các Pipeline

### Bước 1: Thu thập dữ liệu (Offline - Crawling)
Thu thập danh sách phim từ TMDB về máy tính cá nhân:
```bash
python pipeline/ingest.py
```
*Kết quả:* Dữ liệu thô được ghi nhận tại `pipeline/data/movies_raw.csv`.

### Bước 2: Làm sạch & Đóng gói dữ liệu (Offline - Document Cleaning)
Làm sạch văn bản thô, tạo chuỗi text tích hợp cho mỗi bộ phim:
```bash
python pipeline/clean.py
```
*Kết quả:* File dữ liệu sạch được tạo ra tại `pipeline/data/movies_clean.csv`.

### Bước 3: Tạo Vector & Tải lên database (Offline - Indexing)
Nhúng dữ liệu thành Vector Dense & Sparse rồi đẩy lên Qdrant Cloud:
```bash
python pipeline/dual_embedding_qdrant.py
```
*Kết quả:* Collection `movies_hybrid_collection` được tạo lập và lập chỉ mục (index) sẵn sàng trên đám mây Qdrant.

### Bước 4: Chạy Đánh giá thử nghiệm (Optional - Evaluation)
Đánh giá so sánh hiệu năng truy hồi của hệ thống:
```bash
python retrieval/evaluate_simple.py
```

### Bước 5: Chạy Giao diện tìm kiếm (Online - Streamlit Web App)
Khởi chạy ứng dụng web tìm kiếm phim trực quan:
```bash
streamlit run ui/app_final.py
```
Sau khi khởi chạy thành công, trình duyệt sẽ tự động mở trang web tại địa chỉ: `http://localhost:8501`. Bạn có thể nhập mô tả phim của mình để kiểm tra độ chính xác của lõi tìm kiếm Adaptive RAG!