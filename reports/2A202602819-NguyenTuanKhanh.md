# Individual contribution report

Mỗi thành viên copy template này thành:

```text
reports/<student-id>-<short-name>.md
```

Giới hạn khuyến nghị: 1 trang, không chép lại README hoặc mô tả lý thuyết chung. Báo cáo không phải một bài pipeline cá nhân; mục đích là ghi nhận ownership và bằng chứng đóng góp trong sản phẩm nhóm.

---

## Thông tin

- Họ và tên: Nguyễn Tuấn Khanh
- Mã học viên: 2A202602819
- Nhóm: K4-L3A-RAG-Pipeline
- Repository/branch: Workspace hiện tại, commit `f1eb6c8`

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Evaluation harness | Thêm retry/backoff cho 429, batch evaluation theo `EVAL_START_INDEX`, merge kết quả theo từng metric | [run_ab_eval.py](../evaluation/run_ab_eval.py) | Done |
| Offline evaluation | Tính proxy metrics từ checkpoint khi provider hết quota và ghi failure analysis | [RESULT.md](../evaluation/RESULT.md), [ab_eval_generations_checkpoint.json](../evaluation/ab_eval_generations_checkpoint.json) | Done |

Chỉ kê khai công việc có thể đối chiếu bằng file, commit, pull request, test hoặc kết quả evaluation.

## Quyết định kỹ thuật quan trọng

Mô tả tối đa hai quyết định mà bạn trực tiếp tham gia:

1. **Quyết định:** Dùng checkpoint đã hoàn tất thay vì gọi lại generator.
   **Lý do/evidence:** Checkpoint có 18 câu hỏi cho cả dense-only và hybrid+RRF, khớp hoàn toàn với golden dataset; provider trả 429/code 1300.
   **Trade-off:** Không phát sinh thêm quota nhưng không đo lại được latency generation.

2. **Quyết định:** Chạy generator từng câu hỏi một cho mỗi cấu hình để tạo checkpoint.
   **Lý do/evidence:** `EVAL_START_INDEX` và `EVAL_LIMIT=1` giới hạn mỗi batch generation còn một câu hỏi; kết quả được lưu dần vào `ab_eval_generations_checkpoint.json`, còn exponential backoff xử lý lỗi 429.
   **Trade-off:** Generation mất nhiều thời gian hơn nhưng giảm burst request và có thể tiếp tục từ checkpoint thay vì chạy lại toàn bộ 18 câu hỏi.

## Kiểm thử và kết quả

- Test hoặc query tôi đã dùng: `py_compile group_project/evaluation/run_ab_eval.py`; chạy batch `EVAL_START_INDEX=0`, `EVAL_LIMIT=1`; kiểm tra checkpoint có 18/18 case.
- Kết quả trước/sau nếu có: Config B cao hơn proxy offline ở cả bốn metric; average 0.4689 so với 0.3843 của Config A.
- Lỗi đã phát hiện và cách xử lý: Mistral trả HTTP 429/code 1300; thêm exponential backoff, sau đó dùng proxy offline từ checkpoint khi quota cạn.

## Điều còn hạn chế

- Một hạn chế cụ thể của phần tôi làm: Proxy token-overlap không thay thế đánh giá ngữ nghĩa của Ragas LLM-judge; chưa có số đo latency/cost đầy đủ.
- Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện: Chạy lại cùng checkpoint với Ragas khi quota hoạt động và đối chiếu các delta proxy với điểm LLM-judge.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 2026-09-21
- Tên thành viên: Nguyễn Tuấn Khanh
