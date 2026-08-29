# Dữ liệu

Thư mục này chứa các tệp được tạo bởi pipeline và không lưu CSV đầy đủ trong Git.

- `movies_raw.csv`: metadata lấy từ TMDB bằng `pipeline.ingest`.
- `movies_clean.csv`: dữ liệu đã làm sạch và trường `combined_text` dùng để index.

Nguồn dữ liệu: The Movie Database (TMDB). Pipeline mặc định lấy phim giai đoạn
1990-2026, `vote_count >= 100`, `vote_average >= 6.5`, tối đa 25 trang mỗi năm.
Thời điểm thu thập phụ thuộc lần chạy và cần được ghi lại khi công bố benchmark.

Không dùng các tệp này làm nhãn đánh giá. Bộ truy vấn đánh giá được quản lý riêng
trong `evaluation/` để tránh trộn dữ liệu index với dữ liệu benchmark.
