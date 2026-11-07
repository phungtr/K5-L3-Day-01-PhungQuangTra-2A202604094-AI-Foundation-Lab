# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202604094
- **Họ và tên:** Phùng Quang Trà
- **SEED:** 38

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

Bộ kiểm tra trong notebook: 15/15 đạt.

## 2. Kết quả chính

Phần 4 dùng Groq: model nhỏ `openai/gpt-oss-20b`, model lớn `openai/gpt-oss-120b`.

| Phương án | Precision | Recall | F1 test | F1 SUB 40 | p95 ms/tin | tin/s (cả lô) | USD/1k tin |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.783 | 0.988 | 0.886 | 0.850 | 0.05 | 53,614 | 0 |
| 2. BoW + MLP (deep learning) | 0.915 | 0.970 | 0.943 | 0.950 | 0.14 | 70,222 | 0 |
| 3a. Encoder (MiniLM) + kNN | 0.953 | 0.856 | 0.904 | 0.950 | 35.21 | 24 | 0 |
| 3b. Encoder (MiniLM) + LogReg | 0.953 | 0.944 | 0.948 | 0.975 | 30.39 | 50 | 0 |
| 3c. Encoder (MiniLM) + MLP head | 0.953 | 0.971 | 0.962 | 0.950 | 29.02 | 52 | 0 |
| 4a. LLM nhỏ zero-shot | 1.000 | 0.870 | — | 0.935 | 2357 | 1.31 | 0* |
| 4b. LLM lớn zero-shot | 1.000 | 1.000 | — | 1.000 | 2396 | 1.27 | 0* |
| 4c. LLM nhỏ few-shot | 1.000 | 0.952 | — | 0.976 | 2411 | 1.21 | 0* |

\* Groq gói miễn phí nên chi phí đo được là $0. Precision/recall hàng 4 tính trên SUB 40 tin.

- Đường Pareto (theo F1 SUB): BoW + LogReg, BoW + MLP, 3b và 4b.
- Đạt SLA p95 < 50 ms: 1, 2, 3a, 3b, 3c. Không có LLM nào đạt.

## 3. Dự đoán và thực tế

1. **`ml_f1`** — Em đoán F1 ≈ 0.85 và precision > recall. Thực tế F1 = 0.886 nhưng recall 0.988 > precision 0.783, ngược với dự đoán. Em nghĩ dữ liệu ít spam thì mô hình sẽ bỏ sót spam, nhưng các từ như "call", "free", "txt" có trọng số rất mạnh nên mô hình chặn quá tay cả tin ham.
2. **`dl_curve`** — Em đoán val loss giảm mãi và MLP kém hơn LogReg. Thực tế val loss thấp nhất ở epoch 6 rồi tăng lại (overfit), còn MLP đạt F1 0.943 > 0.886. Em chưa tính tới việc mạng 128,129 tham số học thuộc tập train nhỏ.
3. **`emb_tricky`** — Em đoán Encoder+LogReg đúng khoảng 5/6, BoW khoảng 2/6. Thực tế BoW 4/6, Encoder 6/6. BoW không sai ở các tin ham có từ khoá ("free for lunch", "won the match") như em nghĩ, mà chỉ sai ở 2 tin spam né từ khoá ("at no cost", "account rewards expire").

## 4. Phân tích 50 case khó

| Model | F1 SUB | F1 50 case khó | Acc 50 case khó |
| --- | --- | --- | --- |
| 1. BoW + LogReg | 0.850 | 0.472 | 0.509 |
| 2. BoW + MLP | 0.950 | 0.520 | 0.545 |
| 3a. Emb + kNN | 0.950 | 0.710 | 0.673 |
| 3b. Emb + LogReg | 0.975 | 0.661 | 0.655 |
| 3c. Emb + MLP head | 0.950 | 0.702 | 0.691 |
| 4a. LLM nhỏ zero-shot | 0.935 | 0.894 | 0.873 |
| 4d. LLM nhỏ + prompt phòng thủ | 0.935 | 0.907 | 0.891 |

- Tụt nhiều nhất là **BoW + MLP** (F1 0.950 → 0.520). Tập test thường cùng phân phối với train, không có case cố tình đánh lừa, nên điểm quá lạc quan.
- **BoW** yếu nhất ở nhóm né từ khoá (0–17%) và spam giọng tự nhiên, vì chỉ nhìn token rời rạc. Ví dụ case #29 `"F R E E  e n t r y into our weekly draw…"`: chữ bị tách ra nên "free", "entry" không còn khớp từ trong vocab.
- **Embedding** yếu nhất ở spam giọng tự nhiên (20–50%), vì vector nghĩa của câu nằm gần tin bình thường.
- **LLM** yếu nhất ở nhóm tưởng spam nhưng không phải (~70%), vì đọc như người nên nghi ngờ quá mức. Ví dụ case #2 `"FREE pizza in the office kitchen right now…"` là ham nhưng 4d đoán spam vì có "FREE" và giọng gấp gáp.
- Prompt phòng thủ (4d) nâng nhóm prompt injection từ 83% lên 100%, nhưng làm spam giọng tự nhiên giảm từ 100% xuống 90%.
- Với 5 case tự viết, 4d sai 1/5: tin spam giả làm thông báo phòng gym kèm link `bit.ly`, giọng lịch sự và không có từ khoá spam.

## 5. Thử thách cá nhân

**Đề:**

```
    "ML · Đổi ngưỡng quyết định 0.5 → 0.2 và → 0.8 cho mô hình Phần 1. Báo precision/recall/F1 từng ngưỡng. Một ngân hàng sợ chặn nhầm tin OTP thì nên chọn ngưỡng nào? Vì sao?",
    "ML · Giảm vocab còn 100 từ và tăng lên 5000 từ (build_vocab(..., max_size)). Train lại, báo F1 test. Giải thích xu hướng.",
    "DL · Đổi lớp ẩn MLP thành 4 và 512 nơ-ron. Báo số tham số, F1 test, thời gian train. Khi nào nên chọn mạng nhỏ?",
    "Embedding · Chạy knn_predict với k = 1, 5, 25. Báo F1. Vì sao k quá nhỏ hoặc quá lớn đều có hại?",
    "Embedding · Viết 4 tin nhắn của riêng bạn (2 spam né từ khoá, 2 ham chứa 'free'/'win'). So dự đoán BoW vs Embedding+LogReg và giải thích từng tin.",
    "LLM · Sửa build_classification_prompt để LLM trả JSON {\"label\": ..., \"reason\": ...}, viết parse cho JSON, chạy lại trên 10 tin SUB. So token/chi phí với bản 1 từ. Đáng không?",
```

**Cách làm:** Em chưa làm được ạ.


## 6. Phương án cho sản phẩm thật

Em chọn **3b. Encoder (MiniLM) + LogReg**:

- Đạt SLA: p95 30.4 ms < 50 ms.
- Ít lỗi nhất trong các cách đạt SLA trên SUB (F1 0.975); F1 test 0.948.
- Không tốn tiền API.
- Throughput đo được khoảng 50 tin/s mỗi máy, nên để xử lý ~116 tin/s (10 triệu SMS/ngày) cần khoảng 3 máy hoặc chạy GPU.

LLM chính xác nhất (4b, F1 SUB = 1.000) nhưng p95 khoảng 2.4 giây, vượt SLA gần 48 lần, và ước tính khoảng $3,400/tháng nếu trả theo giá niêm yết.

Đánh đổi em chấp nhận: 3b vẫn sai khoảng 5% trên test thường và chỉ đạt F1 0.661 trên 50 case khó, yếu nhất với spam giọng tự nhiên.

Hướng cải thiện là dùng router: 3b xử lý mọi tin, chỉ tin lưỡng lự (p ∈ [0.3, 0.7]) mới gửi LLM xử lý bất đồng bộ. Cách này gần nhanh và rẻ như 3b mà chính xác gần bằng LLM. Thêm bước chuẩn hoá văn bản trước model để chống né từ khoá.

## 7. AI log và điều đã học

| Phần | Hỏi AI gì | AI trả lời/viết gì | Em sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| 1 – ML | Giải thích dòng cập nhật w, vì sao BoW sai; câu 1.B; Bắt lỗi 1; Checkpoint A | Gợi ý câu trả lời ngắn | Đối chiếu với output tin bị đoán sai và biểu đồ loss |
| 2 – DL | Đếm tham số MLP, câu 2.A, Giải thích 2 (5 dòng TODO, zero_grad) | Cách tính 128,129 tham số, giải thích overfit và vòng lặp train | Kiểm tra số tham số và epoch 6 với output notebook |
| 3 – Embedding | Câu hỏi 3, Checkpoint B | Gợi ý trả lời | AI ghi nhầm ví dụ "free for lunch", em sửa theo kết quả thật (BoW 4/6, Enc 6/6) |
| 4 – LLM | Lấy API key, lỗi NameError và Connection refused với Ollama | Hướng dẫn chuyển sang Groq, chạy ô cấu hình trước | Chuyển sang Groq, chạy lại thành công |
| 4 – LLM | Câu hỏi 4, Bắt lỗi 3, dự đoán llm_models / llm_fewshot | Gợi ý trả lời | Đối chiếu F1 few-shot 0.976 trong output |
| 5 – Tổng hợp | Câu hỏi 5, 6, Bonus 5 case, Phản ánh cuối bài | Gợi ý trả lời, viết 5 case khó | Đối chiếu biểu đồ Pareto và bảng 50 case khó; chạy lại để xem model sai case nào |


Điều em hiểu ra hôm nay: [điền 2–3 câu]
nên tự làm  thử  thách  bản thân  tránh  dùng  ai việc  sử  dụng  gợi  ý  từ ai chỉ nên dùng  ở 1,2,3  khi chưa  hiểu rõ và  thực  sự  có  nhu cầu  cần  giải
hiểu  được  bản  chất  của bài  học, ví  dụ như các thế hệ ai:Từ ML (người chọn feature), sang DL (máy tự học feature), sang embedding (mượn feature đã pre-train), rồi đến LLM (không cần train)
một  model dat diem cao  thì  chưa  chắc đã đúng  do có  thể  gặp dữ  liệu  trùng lặp,..