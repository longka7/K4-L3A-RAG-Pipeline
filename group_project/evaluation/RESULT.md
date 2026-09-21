# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-21 |
| Framework and version              | Offline Python proxy; Ragas 0.4.3 quota-limited |
| Evaluator model                    | None; deterministic lexical proxy |
| Generator model                    | `ministral-3b-2512` |
| Embedding model                    | `BAAI/bge-m3` |
| Corpus version/commit              | `f1eb6c8` |
| Golden dataset size                | 18 cases |
| `top_k`                            | 5 |
| Fallback threshold and calibration | 0.3 default; no new calibration run |

## Configurations
- **Config A — dense-only:** semantic retrieval only.
- **Config B — hybrid + RRF:** semantic and lexical retrieval fused with reciprocal rank fusion.
Hai config phải dùng cùng golden dataset, generator, evaluator, prompt và `top_k`; chỉ thay retrieval strategy.

## Overall scores

| Metric            | Config A | Config B | Delta B−A |
| ----------------- | -------: | -------: | --------: |
| Faithfulness      | 0.4762 | 0.5842 | +0.1079 |
| Answer relevance  | 0.2458 | 0.3473 | +0.1015 |
| Context recall    | 0.7143 | 0.8146 | +0.1003 |
| Context precision | 0.1007 | 0.1295 | +0.0288 |
| **Average**       | 0.3843 | 0.4689 | +0.0846 |

Các điểm trên là **proxy offline**, không phải điểm Ragas LLM-judge. Cách tính được cố định và có thể tái lập từ checkpoint: faithfulness là tỷ lệ token câu trả lời xuất hiện trong context; answer relevance là F1 token giữa câu trả lời và expected answer; context recall là tỷ lệ token expected context xuất hiện trong context đã truy xuất; context precision là tỷ lệ token truy xuất thuộc expected context. Stopwords và token ngắn được loại bỏ.

## A/B comparison
**Cấu hình tốt hơn:** Config B theo proxy offline trên cả bốn metric.

**Evidence:** B tăng faithfulness +0.1079, relevance +0.1015, recall +0.1003 và precision +0.0288; average tăng +0.0846.

**Trade-off:** B cần thêm BM25 và bước RRF, nên có chi phí CPU/latency retrieval cao hơn A; checkpoint không lưu timing nên chưa định lượng được.

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage             | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------------------- | ---------- |
|   1 | What does the Vingroup Science & Technology Scholarship Program for Overseas Study fully cover for recipients? | B | 0.3521 | 0.0435 | 0.1786 | 0.0758 | retrieval + generation | Context B không lấy được đoạn coverage chi tiết; câu trả lời từ chối và không chứa các khoản chi phí trong expected answer. |
|   2 | What research areas were the 2023 VinUni-affiliated University of Sydney PhD scholars working on? | B | 0.6250 | 0.0000 | 0.2083 | 0.0549 | retrieval + generation | Context có health/robotics và faculty nhưng thiếu ba chủ đề cụ thể; câu trả lời cũng không nêu đúng các chủ đề cần tìm. |
|   3 | What percentage of tuition does the VinUniversity MBA scholarship cover? | B | 0.3415 | 0.0727 | 0.5714 | 0.0331 | retrieval + generation | Context lấy thông tin học bổng chung nhưng thiếu tài liệu MBA; câu trả lời từ chối nên không khớp expected answer 50%-100%. |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
|        1 | Tăng trọng số/kiểm tra lexical retrieval cho câu hỏi coverage và nguồn chuyên biệt | B vẫn có recall thấp ở các case Vingroup/Sydney; expected evidence không xuất hiện trong top-5 | Tăng recall và giảm câu trả lời từ chối | Chạy lại top-k 5/8 với cùng checkpoint mới và kiểm tra source_file của expected_context |
|        2 | Bổ sung chunk liên quan vào corpus và kiểm tra mapping nguồn | MBA và scholarship overseas có tài liệu corpus nhưng không được lấy vào context | Tăng context recall/precision ở câu hỏi nguồn dễ nhầm | Truy vấn trực tiếp từng source, rồi chạy lại A/B trên ba case worst |
|        3 | Giữ prompt từ chối nhưng yêu cầu trả lời đủ các evidence đã có | Một số câu trả lời đúng hướng nhưng bỏ sót số liệu/chủ đề trong context | Tăng faithfulness và answer relevance mà không khuyến khích bịa | Kiểm tra thủ công citation và chạy lại proxy cùng bộ tokenization |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| Advanced reranker (Jina/self-hosted) | Chưa thực hiện; baseline là dense + BM25 + RRF | Chưa có số đo | Chưa đo; không thêm dependency hay provider cost | Cài một reranker, giữ nguyên dataset/generator/prompt/top_k, rồi so sánh trực tiếp với baseline RRF |
