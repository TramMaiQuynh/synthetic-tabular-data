## MODULE 1: TIỀN XỬ LÝ DỮ LIỆU

### Tại sao Module 1 tồn tại?

**Vấn đề nghiệp vụ:** Mô hình AI học sâu chỉ hiểu số thực trong khoảng chuẩn hóa. Dữ liệu thực tế luôn lộn xộn: có cột là văn bản (`"Yes"/"No"`), có cột là số lớn (`$12,345`), có ô trống, có cột định danh cá nhân. Nếu đưa thẳng vào mô hình AI sẽ bị lỗi hoặc cho ra kết quả vô nghĩa.

Module 1 là **"cánh cổng kiểm soát đầu vào duy nhất"** — mọi dữ liệu phải qua đây trước khi bất kỳ thành phần nào khác được đụng vào.

---

### 1.1 — De-identification (Khử Nhận Dạng)

**Nghiệp vụ là gì?**

Trước khi làm bất cứ điều gì với dữ liệu, phải xóa ngay các cột có thể xác định danh tính trực tiếp. Đây là **yêu cầu pháp lý bắt buộc**, không phải tuỳ chọn.

```
Ví dụ thực tế — Telco Customer Churn dataset:

  CỘT PHẢI XÓA (Direct Identifiers):
    customerID = "7590-VHVEG"  → xóa
    phoneNumber = "0901234567" → xóa
    email = "nguyen@gmail.com" → xóa

  CỘT GIỮ LẠI (Quasi-Identifiers — vẫn cần cho AI):
    gender = "Female"          → GIỮ (quan trọng cho phân phối)
    SeniorCitizen = 0          → GIỮ
    tenure = 12 tháng          → GIỮ
    MonthlyCharges = $65.70    → GIỮ
```

**Tại sao quasi-identifiers không bị xóa ở bước này?**

Vì nếu xóa thêm → mô hình mất thông tin → dữ liệu tổng hợp kém trung thực. Trách nhiệm bảo vệ quasi-identifiers được chuyển giao cho **DP-SGD ở Module 2** — cơ chế có bảo đảm toán học chặt chẽ hơn việc xóa cột.

**Hai tuyến phòng thủ bổ sung cho nhau:**
- De-identification: Phòng thủ tầng 1 — loại bỏ rủi ro trực tiếp
- DP-SGD: Phòng thủ tầng 2 — bảo đảm toán học cho rủi ro gián tiếp

---

### 1.2 — Schema Inference (Nhận Diện Cấu Trúc Dữ Liệu)

**Nghiệp vụ là gì?**

Hệ thống phải tự động "hiểu" dữ liệu mà không cần người dùng khai báo thủ công từng cột. Trong thực tế, dataset có thể có 50–200 cột — không thể yêu cầu người dùng label từng cái.

```
Ví dụ Schema tự động phát hiện:

  tenure         → CONTINUOUS  (số tháng, range [0, 72])
  MonthlyCharges → CONTINUOUS  (số thực, range [18.25, 118.75])
  Contract       → CATEGORICAL (3 nhóm: Month-to-month/One year/Two year)
  Churn          → CATEGORICAL (binary: Yes/No) — đây là target column
  SeniorCitizen  → CATEGORICAL (binary: 0/1)
```

**Tại sao schema quan trọng?**

Schema artifact là "bản đồ" chung dùng cho cả ba module:
- Module 2 dùng để biết cách encode từng cột vào tensor
- Module 3 dùng làm **reference distribution** để so sánh: "Phân phối của cột `MonthlyCharges` trong synthetic có giống distribution gốc không?"
- Constraints Engine dùng để detect vi phạm: "Giá trị `-5` trong cột `tenure` là vô lý vì range gốc là [0, 72]"

---

### 1.3 — Missing Value Imputation (Xử Lý Dữ Liệu Thiếu)

**Nghiệp vụ là gì?**

Dữ liệu thực tế luôn có ô trống. Ví dụ: khách hàng không điền số điện thoại phụ, hoặc một cột chỉ xuất hiện từ năm 2020 nên data trước đó trống hết. Mô hình AI không thể xử lý `NaN` (Not a Number) — phải làm gì đó với chúng.

**Cách thông thường (sai) và tại sao sai:**

```
Cách naïve: điền mean/median
  MonthlyCharges: [65, NaN, 45, NaN, 90]
  → điền mean = 66.67
  → [65, 66.67, 45, 66.67, 90]

Vấn đề: Mô hình sẽ học rằng giá trị 66.67 xuất hiện
  rất thường xuyên (artificial spike) → phân phối bị bóp méo
  → dữ liệu tổng hợp cũng sẽ có đỉnh nhân tạo này
```

**Cách đúng trong hệ thống này:**

```
Bước 1: Thêm cột phụ is_missing_MonthlyCharges = [0, 1, 0, 1, 0]
Bước 2: Điền sentinel value -999 (nằm ngoài mọi range thực tế)
  → MonthlyCharges: [65, -999, 45, -999, 90]

Kết quả:
  Mô hình học: "Khi MonthlyCharges = -999 VÀ is_missing = 1,
  đây là trạng thái THIẾU — có xác suất xuất hiện riêng"
  → Data synthetic cũng sẽ có tỷ lệ thiếu tự nhiên đúng như gốc
```

---

### 1.4 — Encoding & Normalization (Mã Hóa và Chuẩn Hóa)

**Nghiệp vụ là gì?**

Mô hình AI chỉ hiểu số. Phải chuyển đổi:
- `"Female"` → số học được
- `$65.70` → về khoảng [0, 1] để gradient descent hoạt động ổn định

**Thiết kế quan trọng — Tại sao không one-hot ở bước này?**

```
Nếu one-hot sớm:
  Contract = [Month-to-month, One year, Two year]
  → [1,0,0], [0,1,0], [0,0,1]

  CTGAN cần xử lý theo "mode-specific normalization" riêng
  CTVAE cần embedding riêng
  Diffusion cần cách tiếp cận khác

→ Mỗi kiến trúc có cách xử lý categorical tối ưu khác nhau.
  Nếu one-hot sớm → bị lock vào một cách, không linh hoạt.

Giải pháp: Giữ label gốc ("Month-to-month") trong dữ liệu
trung gian. Từng mô hình tự xử lý theo cách phù hợp nhất
với kiến trúc của nó, dựa trên schema artifact.
```

Tất cả phép biến đổi được lưu vào **transformation artifact** — cần để đảo ngược (inverse transform) khi decode dữ liệu tổng hợp về giá trị gốc.

---

## MODULE 2: CỖ MÁY SINH DỮ LIỆU

### Tại sao Module 2 tồn tại?

**Vấn đề nghiệp vụ:** Sau khi có dữ liệu sạch, cần một cơ chế **học phân phối thống kê** của dữ liệu thật và **sinh ra dữ liệu mới** mang cùng phân phối đó — nhưng **không sao chép bất kỳ hàng nào** từ dữ liệu gốc.

Đây không phải chỉ là bài toán kỹ thuật — đây là bài toán kinh doanh: dữ liệu tổng hợp phải đủ tốt để người nhận (data scientist, nhà nghiên cứu) tin dùng cho công việc thực tế.

---

### 2.1 — Ba Kiến Trúc Mô Hình Sinh (CTGAN / CTVAE / Diffusion)

**Nghiệp vụ là gì?**

Không có kiến trúc nào là tốt nhất trong mọi tình huống. Dữ liệu khác nhau, bài toán khác nhau cần mô hình khác nhau. Mục tiêu là so sánh khách quan để chọn đúng công cụ cho từng bối cảnh.

```
CTGAN — phù hợp khi:
  + Dữ liệu bị mất cân bằng lớp nghiêm trọng
    (Ví dụ: chỉ 5% khách churn, 95% không churn)
  + Mối quan hệ giữa các cột phức tạp, phi tuyến
  - Khó train, dễ bị "mode collapse" (chỉ sinh 1 loại dữ liệu)
  - Dưới DP-SGD: chỉ áp dụng cho Discriminator

CTVAE — phù hợp khi:
  + Cần training ổn định, dễ debug
  + Nhiều cột continuous (số liên tục)
  + Cần latent space cho conditional generation
  - Có thể bị "oversmoothing" (dữ liệu sinh quá mượt, mất outlier)

Diffusion — phù hợp khi:
  + Cần chất lượng cao nhất (state-of-the-art)
  + Dữ liệu phức tạp, nhiều chiều
  - Chậm hơn khi inference (phải chạy nhiều bước denoising)
  - Mới hơn, ít tài liệu hơn trong bối cảnh tabular data
```

**Tại sao phải so sánh cả ba?**

Vì **không biết trước kiến trúc nào tốt hơn cho dataset cụ thể này**. Đây là lý do Nhóm A trong thực nghiệm tồn tại — để trả lời câu hỏi này một cách khách quan.

---

### 2.2 — Differential Privacy / DP-SGD

**Nghiệp vụ là gì?**

Đây là thành phần **đặt tên cho toàn bộ đề tài**. Mô hình AI có thể "nhớ" — và đây là rủi ro nghiêm trọng trong thực tế.

**Ví dụ tấn công thực tế:**

```
Tình huống: Một bệnh viện train model AI chẩn đoán ung thư
trên dữ liệu bệnh nhân, rồi release model đó.

Kẻ tấn công biết rằng bệnh nhân Nguyễn Văn A (tuổi 45,
huyết áp 130, đường huyết 6.5) có thể nằm trong training set.

Bằng cách query model với các biến thể của hồ sơ này,
kẻ tấn công có thể suy luận với độ chính xác cao
rằng Nguyễn Văn A CÓ trong dataset → biết ông ấy
bị ung thư (thông tin y tế tuyệt mật).

→ Đây là Membership Inference Attack (MIA)
```

**DP-SGD ngăn chặn điều này như thế nào?**

```
Training bình thường:
  Model học "chính xác" từ TỪNG mẫu cụ thể
  → Mẫu của ông A "để lại dấu vết" trong weights

DP-SGD:
  Bước 1 — Gradient Clipping:
    Giới hạn mức ảnh hưởng tối đa của MỌI mẫu đơn lẻ
    (dù mẫu của ông A là outlier đến đâu, ảnh hưởng
    của nó cũng bị cắt xuống tối đa C)

  Bước 2 — Noise Addition:
    Thêm nhiễu ngẫu nhiên Gaussian vào gradient tổng hợp
    (nhiễu đủ lớn để "che" đi đóng góp của bất kỳ cá nhân nào)

  Kết quả:
    Model học phân phối TỔNG THỂ nhưng không nhớ CHI TIẾT
    của bất kỳ cá nhân nào.

Bảo đảm toán học (ε, δ):
  "Xác suất để mô hình tiết lộ dữ liệu của bất kỳ cá nhân nào
  tăng tối đa e^ε lần so với nếu cá nhân đó không có trong dataset"

  ε = 1 → rất an toàn, khó phân biệt
  ε = 10 → ít an toàn hơn, nhưng chất lượng mô hình tốt hơn
```

**Tại sao phải chọn ε trước khi train?**

Vì ε là **ngân sách bảo mật** — một khi đã dùng hết không thể phục hồi. Nếu train xong mới quyết định ε → đã muộn. Đây là lý do Nhóm B trong thực nghiệm tồn tại: chạy trước với nhiều giá trị ε khác nhau để quyết định mức nào phù hợp cho từng use case.

---

### 2.3 — Memorization Mitigation (Hạn Chế Ghi Nhớ Mẫu)

**Nghiệp vụ là gì?**

DP-SGD cung cấp bảo đảm **lý thuyết** nhưng không loại trừ hoàn toàn rủi ro **thực tế**. Cần thêm tuyến phòng thủ bổ sung:

```
Vấn đề thực tế:
  Nếu train quá nhiều epochs → model "nhớ" các mẫu đặc biệt,
  outlier (bệnh nhân duy nhất có tổ hợp triệu chứng rất lạ)
  ngay cả khi có DP-SGD.

Giải pháp — hai cơ chế:

  1. Weight Decay (L2 Regularization):
     Phạt các weight quá lớn trong loss function
     → Buộc model dùng cách biểu diễn "tổng quát"
     → Không thể "chuyên tâm nhớ" một mẫu cụ thể

  2. Early Stopping (chỉ khi KHÔNG dùng DP):
     Dừng training khi validation loss không cải thiện
     → Tránh overfit vào training set
     (Không dùng khi có DP-SGD vì sẽ "tiêu thêm" privacy budget
     để đo validation loss một cách private)
```

---

### 2.4 — Hyperparameter Optimization / HPO (Tối Ưu Siêu Tham Số)

**Nghiệp vụ là gì?**

Mỗi dataset có đặc điểm riêng: số hàng, số cột, tỷ lệ missing, mức độ mất cân bằng lớp... Không có bộ hyperparameter nào tốt nhất cho mọi bài toán.

```
Ví dụ: Với Telco dataset (7000 hàng, 20 cột):
  learning_rate = 2e-4 có thể tốt
  batch_size = 256 có thể tốt

Với Adult Income dataset (48000 hàng, 14 cột):
  learning_rate = 1e-4 có thể tốt hơn
  batch_size = 512 có thể cho gradient estimate ổn định hơn

Nếu dùng cùng một bộ setting → một trong hai dataset sẽ
được train không tối ưu → kết quả so sánh không công bằng.
```

**Đặc biệt quan trọng dưới DP-SGD:**

Dưới DP-SGD, batch size lớn hơn = gradient estimate chính xác hơn = noise impact nhỏ hơn tương đối. Công thức heuristic: `batch_size ≈ √N` (N là số mẫu). Đây là lý do cần HPO riêng cho chế độ DP, không thể lấy kết quả HPO từ chế độ non-DP mà dùng.

---

### 2.5 — Scalable Sampling (Lấy Mẫu Có Thể Mở Rộng)

**Nghiệp vụ là gì?**

Sau khi model đã train xong, cần **sinh ra số lượng dữ liệu tùy ý**. Nhưng nếu sinh 1 triệu hàng cùng lúc → hết RAM → crash.

Hệ thống hỗ trợ hai chế độ sinh quan trọng trong thực tế:

```
Chế độ 1 — Unconditional (sinh theo phân phối gốc):
  "Sinh cho tôi 10,000 hàng giống phân phối khách hàng Telco"
  → Dùng cho: test hệ thống, phát triển AI nói chung

Chế độ 2 — Conditional (sinh có điều kiện):
  "Sinh cho tôi 5,000 khách hàng thuộc nhóm Churn=Yes"
  → Dùng cho: giải quyết mất cân bằng lớp
    (Churn dataset thường chỉ có 15-20% khách churn,
    model AI khó học được nhóm thiểu số)
  → Augment thêm data cho nhóm minority → model tốt hơn
```

---

### 2.6 — Business Logic Constraints Engine (Ràng Buộc Nghiệp Vụ)

**Nghiệp vụ là gì?**

Mô hình AI học thống kê, không học luật. Nó có thể sinh ra dữ liệu **thống kê đúng nhưng nghiệp vụ vô lý**.

```
Ví dụ vi phạm nghiệp vụ trong Telco dataset:

  Vi phạm 1: tenure < 0         (không thể có khách hàng âm tháng)
  Vi phạm 2: TotalCharges < MonthlyCharges
             (tổng tiền không thể nhỏ hơn tiền tháng gần nhất)
  Vi phạm 3: Contract = "Two year" nhưng tenure = 1
             (không thể ký 2 năm mà chỉ dùng 1 tháng)
  Vi phạm 4: SeniorCitizen = 1 nhưng age < 60
             (senior citizen phải >= 60)
```

Nếu dữ liệu tổng hợp có những vi phạm này → downstream system sẽ lỗi hoặc model AI train được kết quả sai → mất tin tưởng vào toàn bộ synthetic data pipeline.

**Bốn tầng thực thi constraints (theo thứ tự ưu tiên):**

```
Tầng 1 — Mathematical Transformation (trước training):
  Ví dụ: Biến đổi TotalCharges = MonthlyCharges × tenure
  → Constraint được "baked in" vào cấu trúc dữ liệu
  → Model không có cơ hội vi phạm

Tầng 2 — Soft Loss Penalty (trong training):
  Thêm penalty vào loss function khi model sinh giá trị vi phạm
  → Gradient descent tự động "học" tránh vi phạm
  → Chỉ hoạt động với constraints khả vi (differentiable)

Tầng 3 — Guided Sampling (trong inference):
  Hướng dẫn quá trình sinh để ưu tiên giá trị thỏa constraint
  → Áp dụng cho categorical constraints

Tầng 4 — Post-generation Correction (sau khi sinh):
  Phát hiện và sửa vi phạm còn lại
  → Tầng lưới cuối cùng, có retry limit để tránh vòng lặp vô hạn
```

---

### 2.7 — Inverse Transformation (Giải Mã Ngược)

**Nghiệp vụ là gì?**

Model sinh ra dữ liệu trong **không gian chuẩn hóa** (các số trong khoảng [-1, 1] hoặc [0, 1]). Người dùng cuối cần dữ liệu trong **không gian gốc** (`MonthlyCharges = $65.70`, `Contract = "Month-to-month"`).

```
Quy trình giải mã — thứ tự BẮT BUỘC:

  Bước 1: Inverse MinMax Scaling
    0.52 → $65.70  (MonthlyCharges)
    (Phải làm TRƯỚC, vì sentinel -999 đã bị scale thành
    giá trị khác trong không gian normalize)

  Bước 2: Giải mã missing state
    is_missing_MonthlyCharges = 1 → đặt về NULL/NaN
    is_missing_MonthlyCharges = 0 → giữ $65.70

  Bước 3: Type casting
    65.70001 → $65.70  (làm tròn)
    -0.003 → 0         (clamp giá trị âm về 0 cho age)
    "2023-01-15.0" → "2023-01-15"  (format lại datetime)
```

**Tại sao thứ tự Bước 1 → 2 là bắt buộc?**

Vì `-999` (sentinel) đã bị MinMax scaling biến thành giá trị khác trong training (ví dụ: `-9.83`). Nếu kiểm tra `is_missing` trước khi inverse scale → sẽ so sánh nhầm, không nhận ra sentinel → lỗi logic nghiêm trọng.

---

## MODULE 3: ĐÁNH GIÁ & BÁO CÁO TUÂN THỦ

### Tại sao Module 3 tồn tại?

**Vấn đề nghiệp vụ:** Ai sẽ tin dữ liệu tổng hợp nếu không có bằng chứng?

Một công ty muốn chia sẻ synthetic data cho đối tác. Đối tác hỏi: *"Làm sao tôi biết dữ liệu này đủ tốt để train model? Làm sao biết không lộ thông tin khách hàng thật?"*

Không có câu trả lời định lượng → đối tác không chấp nhận. Module 3 là **"phòng kiểm định chất lượng và văn phòng pháp lý"** trong một — tạo ra bằng chứng có thể kiểm chứng.

---

### 3.1 — Statistical Fidelity Assessment (Đánh Giá Độ Trung Thực)

**Nghiệp vụ là gì?**

Đo lường: "Dữ liệu tổng hợp **giống** dữ liệu thật đến mức nào về mặt thống kê?"

```
Ví dụ thực tế — Telco dataset:

  MonthlyCharges thực tế:
    Mean = $64.76, Std = $30.09
    Distribution: bimodal (đỉnh ở $20 và $80)

  MonthlyCharges tổng hợp (kết quả tốt):
    Mean = $64.51, Std = $29.87
    Distribution: cũng bimodal tương tự

  MonthlyCharges tổng hợp (kết quả kém):
    Mean = $65.00, Std = $15.00 ← bị "oversmoothing"
    Distribution: đơn đỉnh (mất cấu trúc bimodal)
    → Ai dùng data này để train model fraud detection sẽ
    miss hết pattern của nhóm khách hàng giá cao/thấp
```

**Ba chỉ số đo fidelity:**

| Chỉ số | Đo gì | Thang đo |
|---|---|---|
| **JSD** | Sai lệch phân phối xác suất cột phân loại | [0, 1], càng nhỏ càng tốt, < 0.05 là tốt |
| **Wasserstein** | Khoảng cách phân phối cột liên tục | [0, ∞), càng nhỏ càng tốt |
| **Correlation Diff** | Mức bảo toàn tương quan giữa các cột | [0, 1], < 0.1 là tốt |

---

### 3.2 — Privacy Leakage Audit (Kiểm Toán Rò Rỉ Quyền Riêng Tư)

**Nghiệp vụ là gì?**

DP-SGD cho bảo đảm lý thuyết về ε. Nhưng đối tác và cơ quan quản lý muốn **bằng chứng thực nghiệm** — không chỉ công thức toán học. Module này chạy các cuộc tấn công giả lập để chứng minh dữ liệu "chịu được" tấn công trong thực tế.

**Hai loại kiểm tra:**

```
1. Geometric Distance Check (Kiểm tra khoảng cách):

   DCR (Distance to Closest Record):
     Với mỗi điểm synthetic, tìm bản ghi thật GẦN NHẤT.
     Nếu DCR ≈ 0 → synthetic point gần như là bản sao của
     một bản ghi thật cụ thể → VI PHẠM PRIVACY.

   Ví dụ nguy hiểm:
     Synthetic row: age=67, gender=M, tenure=48, charges=$87.50
     Real row:      age=67, gender=M, tenure=48, charges=$87.50
     DCR ≈ 0 → Model đã ghi nhớ và sao chép bản ghi này!

   NNDR (Nearest Neighbor Distance Ratio):
     NNDR = DCR_1st / DCR_2nd
     Nếu NNDR → 1: điểm synthetic "cân bằng" giữa nhiều bản ghi thật → tốt
     Nếu NNDR → 0: điểm synthetic "dính" vào 1 bản ghi thật cụ thể → nguy hiểm

2. Attacker Simulation (Giả lập tấn công):

   MIA (Membership Inference Attack):
     Câu hỏi của kẻ tấn công: "Bản ghi X có trong training data không?"
     Kẻ tấn công được cho: mô hình đã train + tập dữ liệu tổng hợp

     Kết quả tốt: AUC-ROC ≈ 0.5
       (kẻ tấn công không đoán được tốt hơn tung đồng xu)
     Kết quả nguy hiểm: AUC-ROC > 0.7
       (kẻ tấn công đoán đúng 70%+ → dữ liệu bị rò rỉ)
```

---

### 3.3 — TSTR Utility Evaluation (Đánh Giá Tính Hữu Ích)

**Nghiệp vụ là gì?**

Đây là câu hỏi quan trọng nhất với **người dùng cuối**: *"Tôi có thể thay dữ liệu thật bằng dữ liệu tổng hợp này để train model AI không?"*

Chỉ số fidelity (JSD, Wasserstein) nói lên dữ liệu **trông giống** thật về mặt thống kê. Nhưng điều người dùng thực sự cần là: dữ liệu tổng hợp có **hoạt động được** trong bài toán AI thực tế không?

**Framework TSTR (Train on Synthetic, Test on Real):**

```
Bước 1: Tách dữ liệu thật
  Real_train (80%) ──→ dùng để TRAIN model sinh
  Real_test  (20%) ──→ KHÔNG ĐƯỢC ĐỤNG VÀO cho đến khi đánh giá

Bước 2: Train model sinh trên Real_train → Synthetic dataset

Bước 3: Train các classifier chuẩn trên Synthetic
  RandomForest, GradientBoosting, LogisticRegression
  Target: dự đoán khách hàng có Churn không?

Bước 4: Test các classifier này trên Real_test

Bước 5: So sánh với baseline TRTR
  TRTR = Train các classifier đó trên Real_train, test trên Real_test
  (đây là kết quả tốt nhất có thể đạt được khi dùng data thật)

Kết quả:
  TSTR F1 / TRTR F1 = 0.92 → data synthetic giữ được 92% utility
  TSTR F1 / TRTR F1 = 0.70 → chỉ giữ được 70% → không đủ tốt
```

**Tại sao phải dùng đúng Real_test để đánh giá?**

Vì mọi thứ đều phải test trên dữ liệu thật chưa từng thấy. Nếu test trên synthetic → đang tự chấm điểm mình → không có ý nghĩa khoa học.

---

### 3.4 — Visual Distribution Overlays (Biểu Đồ Trực Quan)

**Nghiệp vụ là gì?**

Con số (JSD = 0.03) có thể đúng về mặt tổng thể nhưng che giấu vấn đề cụ thể. Biểu đồ để **kiểm tra nhanh bằng mắt** và phát hiện các pattern bất thường mà metric tổng hợp bỏ sót.

```
Ví dụ:
  JSD = 0.04 (ngưỡng tốt)
  Nhưng nhìn histogram thấy:
    Real: có đỉnh thứ 2 ở $110/tháng (nhóm khách premium)
    Synthetic: đỉnh này biến mất → model không học được nhóm premium

  → Không thể detect bằng JSD tổng hợp
  → Phải nhìn biểu đồ từng cột
```

---

### 3.5 — Automated Compliance Report (Báo Cáo Tuân Thủ Tự Động)

**Nghiệp vụ là gì?**

Đây là **sản phẩm cuối cùng** mà người dùng thực sự cần — một tài liệu pháp lý và kỹ thuật có thể trình lên:
- Cơ quan quản lý (Data Protection Authority)
- Hội đồng kiểm toán nội bộ
- Đối tác nhận dữ liệu
- Hội đồng bảo vệ luận văn

```
Compliance Report bao gồm:
  ┌─────────────────────────────────────────────┐
  │ 1. Metadata của run                         │
  │    Dataset, model, epochs, seed, timestamp  │
  │                                             │
  │ 2. Privacy Budget đã sử dụng               │
  │    ε = 3.0, δ = 1e-5 (hoặc "DP disabled")  │
  │                                             │
  │ 3. Fidelity Metrics                         │
  │    JSD per column, Wasserstein, Corr diff   │
  │                                             │
  │ 4. Privacy Audit Results                    │
  │    DCR stats, NNDR stats, MIA AUC           │
  │                                             │
  │ 5. ML Utility (TSTR vs TRTR)               │
  │    F1-score, AUC-ROC per classifier         │
  │                                             │
  │ 6. Distribution plots                       │
  │    Histogram real vs synthetic per column   │
  │                                             │
  │ 7. Compliance verdict                       │
  │    PASS / FAIL theo ngưỡng cấu hình sẵn    │
  └─────────────────────────────────────────────┘
  → Lưu có versioning: ai chạy, lúc nào, với config gì
  → Có thể audit trail ngược lại bất kỳ lúc nào
```

---

## KIẾN TRÚC BẢO MẬT — Cross-Cutting Concern

### Tại sao đây là "cross-cutting concern", không phải một module?

**Vì bảo mật không phải là tính năng thêm vào — nó phải xuyên suốt toàn bộ pipeline từ đầu đến cuối.** Nếu bảo mật chỉ là một bước ở giữa, toàn bộ chuỗi vẫn có thể bị phá vỡ ở bước trước hoặc sau nó.

```
Module 1 — De-identification:
  Loại bỏ rủi ro TRỰC TIẾP (xóa danh tính)
  ↓
Module 2 — DP-SGD:
  Bảo đảm toán học chống rủi ro GIÁN TIẾP (rò rỉ qua model)
  ↓
Module 2 — Memorization Mitigation:
  Giảm rủi ro THỰC NGHIỆM (model ghi nhớ outlier)
  ↓
Module 3 — DCR/NNDR/MIA:
  CHỨNG MINH THỰC NGHIỆM rằng các cơ chế trên hoạt động
  ↓
Module 3 — Compliance Report:
  TỔNG HỢP toàn bộ bằng chứng thành tài liệu kiểm toán
```

**Tại sao cần cả De-identification VÀ DP-SGD?**

```
De-identification một mình không đủ:
  Xóa tên nhưng giữ (age=67, ZIP=12345, gender=M)
  → Kết hợp 3 quasi-identifiers này đủ để xác định
    87% dân số Mỹ (nghiên cứu Latanya Sweeney 2000)
  → Cần DP-SGD để bảo vệ phần còn lại

DP-SGD một mình không đủ:
  Không xóa tên → model training tiếp xúc trực tiếp
  với PII → vi phạm pháp lý ngay từ đầu,
  dù ε nhỏ đến đâu cũng không hợp lệ theo GDPR

→ Phải dùng CẢ HAI: De-id xử lý rủi ro trực tiếp,
  DP-SGD xử lý rủi ro gián tiếp. Không thay thế nhau.
```

---

## TÓM TẮT LOGIC TOÀN HỆ THỐNG

```
VẤN ĐỀ:  Cần dữ liệu để làm AI, nhưng dữ liệu thật quá nhạy cảm

GIẢI PHÁP: Sinh dữ liệu tổng hợp — học được từ thật, nhưng an toàn

CẤU TRÚC:
  Module 1 → Chuẩn bị: xóa rủi ro trực tiếp, chuẩn hóa về tensor
  Module 2 → Tạo ra:   học phân phối + bảo mật DP + ràng buộc nghiệp vụ
  Module 3 → Chứng minh: đủ tốt (fidelity) + đủ an toàn (privacy) + đủ hữu ích (utility)

TAM GIÁC ĐỐI LẬP:
  Privacy ↑ → Fidelity ↓ → Utility ↓
  Fidelity ↑ → Privacy ↓
  → Nhiệm vụ của ε và thực nghiệm: tìm điểm tối ưu

KẾT QUẢ:
  Dữ liệu tổng hợp + Compliance Report
  = Có thể chia sẻ, có thể dùng cho AI, có thể trình Hội đồng
```

---

## THIẾT KẾ THỰC NGHIỆM — Tại Sao Phải Chạy 8 Thực Nghiệm?

### Tại sao cần thực nghiệm, không chỉ chạy một lần?

**Vấn đề nghiệp vụ:** Nếu chỉ chạy một mô hình một lần rồi báo cáo kết quả, Hội đồng sẽ hỏi:
- "Tại sao chọn CTVAE mà không phải CTGAN?"
- "DP có thực sự tốt không, hay chỉ làm giảm chất lượng?"
- "Kết quả này có tổng quát được không, hay chỉ đúng với dataset này?"

Không có câu trả lời → không có tính khoa học. **Thực nghiệm có kiểm soát biến** là cách duy nhất trả lời những câu hỏi này một cách khách quan.

---

### Tại sao seed=42 là bắt buộc?

```
Vấn đề: Các mô hình AI có yếu tố ngẫu nhiên:
  - Khởi tạo weight ngẫu nhiên
  - Shuffling thứ tự data trong mỗi epoch
  - Sampling trong VAE (reparameterization trick)
  - Noise trong DP-SGD

Hệ quả: Chạy cùng code, cùng config, nhưng hai lần khác nhau
  → cho ra kết quả khác nhau:
  Lần 1: JSD = 0.058
  Lần 2: JSD = 0.071
  → Không biết sự khác nhau là do seed hay do gì khác

Với seed=42 cố định:
  → Python random, NumPy, PyTorch đều dùng cùng chuỗi số ngẫu nhiên
  → Hai lần chạy cùng config → kết quả HOÀN TOÀN GIỐNG NHAU
  → Mọi sự khác biệt giữa A1-CTVAE và A2-CTGAN
    chắc chắn 100% là do KIẾN TRÚC, không phải do may mắn
```

---

### Nhóm A — Tại sao tồn tại?

**Câu hỏi khoa học:** "Trong cùng điều kiện, kiến trúc nào học phân phối dữ liệu Telco tốt nhất?"

**Nghiệp vụ:** Trước khi đưa vào production, phải biết nên deploy model nào. Không thể chọn dựa trên cảm tính hay theo trào lưu.

```
Biến kiểm soát (giữ cố định):
  Dataset:    telco_customer_churn
  Epochs:     100
  Batch size: 256
  LR:         2e-4
  DP:         OFF (tắt — để isolate hiệu ứng kiến trúc)
  Seed:       42

Biến độc lập (thay đổi có chủ đích):
  model_type: ctvae | ctgan | diffusion

Biến phụ thuộc (đo kết quả):
  JSD, Wasserstein, Corr Diff, MIA AUC, TSTR F1
```

**Kết quả kỳ vọng:** Một bảng so sánh rõ ràng để chọn mô hình tốt nhất.

---

### Nhóm B — Tại sao tồn tại?

**Câu hỏi khoa học:** "Khi bật DP, chất lượng giảm bao nhiêu theo từng mức ε?"

**Nghiệp vụ:** Trong thực tế triển khai, cần trả lời câu hỏi của khách hàng: *"Nếu tôi muốn bảo mật mức cao nhất, tôi phải chấp nhận mất bao nhiêu % chất lượng dữ liệu?"*

```
Đây là bài toán "tìm sweet spot" trên đường cong Privacy-Utility:

Chất lượng (TSTR F1)
    │
0.85│ ●  (ε=∞, no DP)
    │  ●  (ε=10)
0.75│    ●  (ε=3)   ← Sweet spot?
    │       ●  (ε=1.5)
0.60│
    └────────────────────→ Mức bảo mật (1/ε)
       Thấp          Cao

→ Hội đồng có thể thấy rõ: "ε=3 giảm 10% chất lượng
  nhưng tăng 40% bảo mật → trade-off chấp nhận được"
```

Sử dụng **mô hình tốt nhất từ Nhóm A** để chạy Nhóm B — đây là thiết kế đúng: trước tiên chọn kiến trúc tốt nhất, sau đó nghiên cứu DP trên kiến trúc đó.

---

### Nhóm C — Tại sao tồn tại?

**Câu hỏi khoa học:** "Kết luận từ Nhóm A có tổng quát hóa được không, hay chỉ đúng với Telco?"

**Nghiệp vụ:** Một kết quả khoa học chỉ tin được nếu có thể **replicate** trên dataset khác. Nếu CTVAE tốt trên Telco nhưng kém trên Adult Income → kết luận "CTVAE là tốt nhất" không có giá trị tổng quát.

```
Adult Income dataset (khác biệt với Telco):
  48,842 hàng (vs ~7,000 của Telco → 7x lớn hơn)
  14 cột (vs ~20 cột)
  Mix: continuous (age, hours/week) + categorical (education, race, sex)
  Target: income >50K hay ≤50K (khác domain hoàn toàn)

Nếu CTVAE vẫn cho JSD thấp và TSTR tốt trên Adult Income
→ Kết luận "CTVAE phù hợp với mixed tabular data" có độ tin cậy cao
→ Đủ cơ sở để recommend trong báo cáo
```

---

### Tại sao 8 thực nghiệm, không phải 3 hay 20?

```
Ít hơn (3 thực nghiệm — chỉ Nhóm A):
  → Biết được kiến trúc nào tốt nhất
  → Nhưng KHÔNG biết DP ảnh hưởng thế nào
  → KHÔNG biết có tổng quát không
  → Báo cáo không đủ sức thuyết phục

Nhiều hơn (20+ thực nghiệm):
  → Thêm nhiều giá trị ε, nhiều dataset, nhiều seed...
  → Tốn thời gian compute không tương xứng
  → Kết quả phức tạp, khó tổng hợp trong báo cáo
  → "Diminishing returns" — thêm ít giá trị khoa học

8 thực nghiệm là điểm cân bằng:
  → Trả lời đủ 3 câu hỏi khoa học cốt lõi
  → Thực thi được trong thời gian hợp lý (~4-8 giờ)
  → Kết quả vừa đủ để tạo bảng so sánh có ý nghĩa
```

---

### Kết quả đang có (A1-CTVAE đã hoàn thành)

| Thực nghiệm | Model | DP | JSD ↓ | MIA AUC | Thời gian |
|---|---|---|---|---|---|
| **A1-CTVAE** | CTVAE | OFF | **0.0602** | **0.5024** ✅ | 472s |
| A2-CTGAN | CTGAN | OFF | đang chạy... | — | — |
| A3-Diffusion | Diffusion | OFF | chờ... | — | — |

> **MIA AUC = 0.5024** là kết quả xuất sắc — kẻ tấn công chỉ đoán đúng 50.24%, gần như đoán ngẫu nhiên. Ngay cả **không có DP**, CTVAE đã bảo mật tốt — chứng tỏ model không ghi nhớ (memorize) dữ liệu cá nhân.
