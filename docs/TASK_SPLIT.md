# Phân công công việc — Day 8 RAG Pipeline

**Đề tài:** Dịch vụ đại học — Học bổng & Hỗ trợ tài chính (VinUniversity)
**Nhóm trưởng:** Nguyễn Long Khánh

---

## Dữ liệu

**Task 1, 2, 3 đã hoàn thành** (xong bởi Nguyễn Long Khánh, đã push lên fork):

| Yêu cầu | Trạng thái | Chi tiết |
|---|---|---|
| Task 1: ≥3 tài liệu **PDF/DOCX** chính sách | ✅ Xong (3 file, `data/landing/legal/`) | Tải trực tiếp từ nguồn chính thức VinUni: `vingroup-scholarship-call-for-applications-2023-2025.pdf`, `conditions-to-retain-scholarship-2021.pdf`, `guideline-student-financial-support-request-2025.pdf` (đều từ `vinuni.edu.vn`/`policy.vinuni.edu.vn`, có text layer đọc được — 1 file khác bị loại vì là bản scan không có text). |
| Task 2: ≥5 bài **news** | ✅ Xong (6 file JSON, `data/landing/news/`) | Cào thủ công qua trình duyệt thật (site chính có WAF chặn request tự động) từ `vinuni.edu.vn/news-events/` và `scholarships.vinuni.edu.vn/news/`: tin học bổng SEED 2026, câu chuyện SV nhận học bổng, thông báo học bổng PhD CS, cảnh báo lừa đảo mạo danh học bổng, lễ trao học bổng 2023, học bổng ĐH Sydney. |
| Task 3: convert sang markdown chuẩn | ✅ Xong (`python -m src.task3_convert_markdown`) | `data/standardized/legal/` có 11 file (3 từ PDF mới convert bằng MarkItDown + 8 file học bổng cào từ Day 7 để làm giàu corpus), `data/standardized/news/` có 6 file. Toàn bộ 3 test liên quan trong `tests/test_acceptance.py` đã pass. |

Người phụ trách Task 4 (Indexing) có thể bắt đầu ngay trên `data/standardized/` — không cần chờ gì thêm.

---

## Phân công 4 người

| # | Phụ trách | Module | Điểm rubric | Việc cụ thể |
|---|---|---|---:|---|
| 1 | **Data Collection** | Task 1, 2, 3 | 10đ | Tìm ≥3 PDF/DOCX chính sách thật, crawl ≥5 bài news, convert sang markdown chuẩn |
| 2 | **Indexing & Dense Retrieval** | Task 4, 5 | 10 + phần dense trong 20đ | Chunk + embed + index vào ChromaDB, semantic search (dense) |
| 3 | **Lexical, Fusion & Pipeline** | Task 6, 7, 8, 9 | phần BM25/RRF trong 20 + 10đ | BM25, RRF, PageIndex fallback (optional), ráp retrieval pipeline + hiệu chỉnh threshold |
| 4 | **Generation, Chatbot & Evaluation** | Task 10, `app.py`, Evaluation | 15 + 10 + 10 = 35đ | Generation kèm citation, chatbot Streamlit, 15 câu golden Q&A + RAGAS 4 metric + A/B, điền `RESULT.md` |

Phần README + individual reports (5đ) là việc chung — nhóm trưởng tổng hợp và review tích hợp giữa 4 phần.

**Việc phân bổ không đều tuyệt đối** (Người 4 nặng nhất ~35đ, Người 1 nhẹ nhất ~10đ). Hai cách cân bằng lại:
- Người 1 (Data) phối hợp viết 15 câu golden Q&A cùng Người 4 — vì Người 1 hiểu corpus nhất (bớt việc cho Người 4).
- Người 2 và Người 3, sau khi xong Task 5–9, phụ Người 4 nối `retrieve()`/`generate_with_citation()` vào `app.py` thay vì để một mình Người 4 làm toàn bộ Streamlit UI.

**Thứ tự phụ thuộc:** dữ liệu (Task 1-3) đã xong nên Người 2/3 có thể bắt đầu ngay trên `data/standardized/` thật, không cần chunk mẫu/mock nữa. Người 4 vẫn có thể code Task 10/`app.py` song song dựa trên schema `SearchResult` cố định sẵn trong `src/contracts.py` (mock `retrieve()` tạm nếu Task 9 chưa xong) mà không cần đợi Người 2/3 hoàn thiện hẳn.

---

## Ràng buộc kỹ thuật chung (đọc trước khi code)

- Mọi module trả đúng schema trong `src/contracts.py` (`Document`, `SearchResult`, `GenerationResult`)
- `id` phải ổn định, không trùng; chạy lại pipeline không tạo dữ liệu trùng
- Task 4 và Task 5 phải dùng chung hàm `embed_texts()`
- Fallback so sánh với **cosine score gốc của dense search**, không dùng RRF score
- Không commit `.env` hoặc API key thật
- Chi tiết đầy đủ: `docs/MODULE_CONTRACTS.md`, `docs/STEP_BY_STEP.md`, `docs/GRADING_RUBRIC.md`
