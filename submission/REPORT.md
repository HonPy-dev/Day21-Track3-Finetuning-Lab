# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Hồng Phi  
**MSSV**: 2A202602750  
**Ngày chạy**: 2026-10-07  
**Tier**: `T4`  **Base model**: `HuggingFaceTB/SmolLM2-135M-Instruct`  **GPU thực tế**: Tesla T4, 14.6 GiB khả dụng

> Các số liệu lấy từ `results/` sau lần chạy đầy đủ trên Colab. Kết quả FAILED được giữ nguyên; không sửa prompt/eval để làm đẹp số liệu.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt, phân loại thành JSON triage |
| Train / val | 225 / 25 (seed 42); eval target 50; eval regression 15 |
| `max_length` | 1024 theo tier T4; p95 đo được 162 token (p99 173, max 175; đề xuất 256) |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 optimizer steps cho cả bốn run |

**Lý do lựa chọn.** Chọn SmolLM2-135M-Instruct theo yêu cầu thử nghiệm model rất nhỏ; T4 có đủ bộ nhớ để so sánh LoRA 16-bit và QLoRA trong cùng tier. Dataset 250 ticket CSKH tiếng Việt được chọn vì bài toán đầu ra JSON triage phù hợp để kiểm tra cả nội dung phân loại lẫn ràng buộc schema; eval target và regression được giữ riêng, không dùng để train.

**Template có giữ khối `<think>` không?** Có. `results/template_check.json` xác nhận template render cả `<think>…</think>` và NB1 kết luận reasoning được giữ lại. Không có biến đổi thủ công nào lên template.

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.313 (41/131 token trong ví dụ proof) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn được tính loss trong proof:

```text
<|im_start|>assistant
{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Mask proof xác nhận phần câu hỏi bị mask, còn JSON trả lời được supervised. Thống kê này là proof trên ví dụ NB1, không phải tỷ lệ tính trung bình toàn bộ corpus.

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0.000 | 0.0667 | 0.000 | 1685.2 |
| (b) base + optimized prompt | 0.065 | 0.0667 | 0.280 | 1658.9 |
| (c) LoRA fine-tune | 0.000 | 0.0667 | 0.000 | 2145.5 |

**(b) có thật sự mạnh hơn (a) không?** Có: target tăng từ 0.000 lên 0.065, format từ 0.000 lên 0.280. Tôi không sửa `OPTIMIZED_PROMPT`; hash trong baseline đóng băng khớp prompt gốc. Đây là cải thiện tương đối so với prompt ngây thơ, nhưng điểm tuyệt đối vẫn thấp: model nhỏ thường lặp lại ticket hoặc không sinh được JSON hợp lệ.

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss cuối (NB4) | target (NB5 §4) | thời gian train (s) | VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 | 4,884,480 | 1e-4 | 2.2754 | 0.000 | 205.8 | 0.48 |
| `attn_only` | q,v | 85 (matched) | 4,896,000 | 1e-4 | 2.2219 | 0.000 | 128.8 | 0.48 |
| `wrong_lr` | text-linear | 16 | 4,884,480 | 1e-5 | 2.9010 | 0.000 | 204.0 | 0.48 |
| `qlora` | text-linear | 16 | 4,884,480 | 1e-4 | 2.3699 | 0.000 | 260.4 | 0.34 |

NB5 autopsy của bốn run, theo các nhóm mà script thực sự chấm:

| Run | target | regression | format | inference latency (ms) |
|---|---:|---:|---:|---:|
| `correct` | 0.000 | 0.0667 | 0.000 | 2145.5 |
| `attn_only` | 0.000 | Không đo trong autopsy | 0.000 | 1826.5 |
| `wrong_lr` | 0.000 | Không đo trong autopsy | 0.000 | 3180.1 |
| `qlora` | 0.000 | Không đo trong autopsy | 0.000 | 3215.0 |

Theo thiết kế NB5, regression được đo cho `correct` để tính cổng bốn nhóm; ba contrast được chấm trên target, format và latency. Vì vậy các ô regression của contrast được ghi rõ là chưa đo, không suy diễn từ loss train.

**4.1 — `attn_only` so với `correct`.** Trên target, `attn_only` hoà với `correct`: cả hai đều đạt 0.000/50, tức không có field nào được scorer ghi nhận đúng trong đầu ra. Train loss của `attn_only` thấp hơn một chút (2.2219 so với 2.2754), nên thứ tự theo loss không tạo ra thứ hạng target; loss tốt hơn không giúp mô hình giải được tác vụ ở đây. Hai run có ngân sách tham số gần bằng nhau (4,896,000 so với 4,884,480), trong khi vị trí adapter khác nhau và rank của q,v được nâng lên 85. Kết quả chỉ cho thấy tăng rank/đổi vị trí không khắc phục được thất bại của setup này; không đủ bằng chứng để kết luận placement hay rank không quan trọng nói chung.

**4.2 — `wrong_lr`.** Với LR 1e-4, log của `correct` giảm từ khoảng 2.896 xuống 1.924 ở mốc log cuối; `final_loss` tổng hợp được ghi trong runs.csv là 2.2754. Với LR thấp hơn 10 lần (1e-5), `wrong_lr` dao động quanh 2.9 và `final_loss` là 2.9010. Sự khác biệt phù hợp với việc LR full-fine-tuning quá thấp để LoRA thích nghi đủ trong 30 bước. Tuy nhiên, nhìn loss đơn lẻ vẫn không chứng minh chất lượng: cả hai run đều có target 0.000 và format 0.000, nên sẽ sai nếu kết luận `correct` đã thành công chỉ vì loss thấp hơn.

**4.3 — `qlora`.** Trong phép đo này, QLoRA ghi peak VRAM 0.34 GB so với 0.48 GB của `correct`, giảm 0.14 GB (xấp xỉ 29% theo số đo đó). Đổi lại, thời gian train tăng từ 205.8 lên 260.4 giây, final loss tăng từ 2.2754 lên 2.3699, và target vẫn là 0.000. Với model 135M trên T4, phần VRAM tiết kiệm tuyệt đối nhỏ và không tạo ra lợi ích chất lượng; do đó QLoRA không đáng chọn cho run này nếu ưu tiên thời gian/chất lượng. Đây là SmolLM2 chứ không phải Qwen3.5, nên kết quả này không kiểm chứng hay bác bỏ khuyến nghị riêng cho họ Qwen3.5.

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = -0.065` · `regression Δ = +0.000` · `valid_trace_rate = 0.000`

Fine-tune đạt target 0.000, thấp hơn baseline (b) 0.065 đúng 0.065 điểm; vì vậy cổng không đạt điều kiện phải vượt prompt tốt nhất. Regression giữ nguyên ở 0.0667 so với baseline, nhưng đây chỉ là mức recall rất thấp trên tập regression 15 mẫu chứ không phải bằng chứng năng lực tổng quát tốt. `format=0.000` và `valid_trace_rate=0.000`; các output fine-tune không tạo JSON parse được và không có trace hợp lệ. Thất bại không đến từ việc eval bị rút gọn: pipeline dùng đủ 50 target và 15 regression, prompt baseline được giữ nguyên, và verifier xác nhận eval không đổi. Kết quả cho thấy fine-tune này không nên ship; prompt engineering có nhích điểm so với prompt ngây thơ nhưng cả model/prompt vẫn yếu trên bài toán tiếng Việt có nhiều trường ràng buộc. Cần cải thiện chất lượng/định dạng dữ liệu, kiểm tra tương thích template–generation và đánh giá thêm trước khi thử train lại; không nới gate sau khi thấy kết quả.

## 6. Định tính — bắt buộc có cả ca THUA

Năm ví dụ dưới đây là năm dòng đầu của eval cố định, không chọn theo kết quả. `field score` là tỷ lệ trường đúng trên bốn trường; fine-tune đạt 0.00 ở cả 50/50 mẫu nên không có ca FT thắng để đưa ra mà không bịa kết quả. Dự đoán baseline (b) cho năm ví dụ được tái tạo bằng greedy decode với đúng prompt đã đóng băng; artifact baseline chính thức vẫn là `baselines_frozen.json`.

| # | Ticket (rút gọn) | Nhãn đúng (rút gọn) | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Chuột không dây; trả lại; gấp; khách hài lòng | `doi_tra / cao / chuột không dây / tich_cuc` | Lặp lại nguyên ticket; 0/4 | “Thành trợ tốt. Gấp. Shop hỗ trợ tốt.”; 0/4 | Cả hai không xuất JSON; FT không giữ được nội dung phân loại. |
| 2 | Ốp lưng; hoàn tiền; sớm; bực mình | `hoan_tien / trung_binh / ốp lưng điện thoại / tieu_cuc` | JSON sai intent, product, sentiment; 1/4 | Câu tiếng Việt lỗi “Bạn có thể thựa…”; 0/4 | Prompt baseline có một field đúng; FT mất cả cấu trúc lẫn nội dung. |
| 3 | Đèn bàn LED; hoàn tiền; quá hạn; cảm ơn shop | `hoan_tien / cao / đèn bàn LED / tich_cuc` | Lặp ticket, thiếu/sai product; 0/4 | Lặp lại ticket rồi thêm “Cảm ơn shop nhiều”; 0/4 | Sinh tự do thay vì JSON. |
| 4 | Bình giữ nhiệt; chưa thấy tiền; khi nào tiện; cảm ơn | `hoan_tien / thap / bình giữ nhiệt / tich_cuc` | Văn bản lỗi, không JSON; 0/4 | “Cảm ơn shop nhiều…” rồi lặp từ vô nghĩa; 0/4 | Cả hai thất bại; FT không học được quy tắc field. |
| 5 | Đèn bàn LED; vỡ khi nhận; gấp; nhờ shop xem | `san_pham_loi / cao / đèn bàn LED / trung_tinh` | Echo ticket rồi lặp từ; 0/4 | Echo ticket rồi “Shop xem giúp”; 0/4 | Bắt chước văn phong ticket nhưng không phân loại. |

Mẫu chung của ca FT thua là output thường echo input, sinh câu hỗ trợ chung chung hoặc lặp cụm từ; không parse được thành object nên cả target lẫn format đều bằng 0. Không có bằng chứng rằng các lỗi chỉ tập trung ở một intent: thất bại xuất hiện ở đổi trả, hoàn tiền và sản phẩm lỗi. Vì cả 50 mẫu đều 0.00, không thể báo cáo ca thắng của FT; đây là kết quả bất lợi nhưng cần được giữ nguyên.

## 7. Kết luận & điều tôi học được

**Kết luận.** Tôi không deploy adapter `correct` này. Trên toàn bộ 50 ticket target, output không parse thành JSON và target score bằng 0.000; nó còn chậm hơn baseline (b) khoảng 487 ms mỗi mẫu theo phép đo, trong khi không cải thiện target. Baseline (b) chỉ đạt 0.065 và format 0.280, nên kết luận đúng không phải “prompt đã giải quyết bài toán”, mà là cả lựa chọn model 135M lẫn prompt hiện tại chưa đủ tốt; tuy vậy baseline b vẫn là mốc cần vượt và fine-tune còn kém hơn. Mask proof lành mạnh ở ví dụ NB1: câu hỏi được mask và câu trả lời được supervise, vì vậy không có dấu hiệu cho thấy lỗi cơ bản là mask đảo. LR 1e-4 cải thiện train loss so với 1e-5, nhưng không chuyển thành chất lượng tác vụ; attention-only và QLoRA cũng không nâng target. Đòn bẩy có tín hiệu tốt nhất trong các số đo hiện tại là cải thiện prompt so với prompt ngây thơ, nhưng mức tuyệt đối vẫn thấp. Trước lần train tiếp theo, tôi sẽ kiểm tra đầu ra/generation trên vài mẫu, dùng model phù hợp hơn cho tiếng Việt và JSON, xác nhận các target thật sự được học từ dữ liệu, rồi mới chạy lại toàn bộ eval cố định. Tôi sẽ không nới gate hoặc thay eval sau khi biết kết quả.

**Ba điều tôi học được**:
1. `supervised_fraction=0.313` cùng `answer_is_supervised=true` và `question_is_masked=true` là bằng chứng cần có; chỉ bật `assistant-only` mà không xem token preview sẽ không đủ.
2. Train loss thấp hơn không đảm bảo target tốt hơn: `attn_only` có loss thấp nhất 2.2219 nhưng vẫn target 0.000; mọi adapter đều format 0.000.
3. Chỉ so sánh với prompt baseline đã đóng băng mới trả lời được fine-tune có đáng ship không; trường hợp này câu trả lời là không.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** kiểm tra chính xác đầu ra và tokenizer/chat template với 5–10 ticket trước khi train; xác nhận pipeline tạo JSON giám sát đúng; thử model instruct lớn hơn hoặc model tiếng Việt; sau đó mới điều chỉnh `max_length` về 256 theo p95 và chạy một ablation có kiểm soát trên prompt/dữ liệu. Tôi sẽ giữ nguyên 50 mẫu eval và so lại với baseline (b) không đổi.

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub
