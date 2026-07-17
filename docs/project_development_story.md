# DỰ ÁN SYNTHETIC TABULAR DATA — CÂU CHUYỆN PHÁT TRIỂN

> **Vai trò**: Senior ML Engineer + Software Architect — người duy nhất đã thiết kế và xây dựng toàn bộ hệ thống từ con số 0.
> 
> **Mục tiêu tài liệu**: Giúp bạn hiểu dự án như thể bạn là tác giả, để tự tin trả lời mọi câu hỏi hội đồng.

---

## PHẦN I: HÀNH TRÌNH PHÁT TRIỂN (DEVELOPMENT TIMELINE)

---

### CHƯƠNG 1: Ý TƯỞNG & BÀI TOÁN KINH DOANH

#### 1.1 Vấn đề thực tế (Business Problem)

Tôi đang làm việc với một ngân hàng lớn. Họ có đội Data Science nội bộ cần xây dựng mô hình dự báo nợ xấu, nhưng:

- **Luật pháp cấm**: Nghị định 13/2023 (Việt Nam), GDPR (Châu Âu) cấm mang dữ liệu khách hàng thật ra khỏi hệ thống.
- **Không thể thuê ngoài**: Ngân hàng muốn thuê công ty AI bên ngoài xây mô hình, nhưng không thể gửi dữ liệu thật vì rủi ro pháp lý.
- **Mất cân bằng dữ liệu**: Trong 1 triệu khách hàng, chỉ có 5,000 người vỡ nợ (0.5%). Mô hình học trên dữ liệu này sẽ bị thiên lệch, luôn dự đoán "không vỡ nợ" và đạt accuracy 99.5% nhưng vô dụng.

**Pain point**: "Có dữ liệu nhưng không dám chia sẻ. Có bài toán nhưng không thể thuê người giải."

#### 1.2 Giải pháp đề xuất

Tôi đề xuất xây dựng một **hệ thống sinh dữ liệu tổng hợp (Synthetic Data Generator)**:

- **Đầu vào**: Dữ liệu thật của khách hàng (chứa PII, nhạy cảm).
- **Đầu ra**: Một bảng dữ liệu mới, hoàn toàn nhân tạo, không chứa bất kỳ hồ sơ thật nào.
- **Tính chất**: Dữ liệu mới giữ nguyên phân phối thống kê và mối tương quan giữa các cột, nên vẫn dùng để huấn luyện AI được.

**Ví dụ trực quan**: Giống như một họa sĩ vẽ chân dung. Anh ta học phong cách, tỷ lệ khuôn mặt từ nhiều người thật, nhưng bức tranh cuối cùng là một người hoàn toàn mới — không phải bản sao của ai.

#### 1.3 Yêu cầu nghiệp vụ (Business Requirements)

Tôi ngồi xuống và viết ra 3 câu hỏi mà hội đồng sẽ hỏi:

| # | Câu hỏi | Ý nghĩa |
|---|---------|---------|
| 1 | "Dữ liệu sinh ra có giống thật không?" | → **Fidelity**: Đo độ tương đồng thống kê |
| 2 | "Dữ liệu sinh ra có bị rò rỉ thông tin không?" | → **Privacy**: Đo khả năng chống tấn công |
| 3 | "Dữ liệu sinh ra có dùng được không?" | → **Utility**: Đo hiệu năng ML trên dữ liệu giả |

Tôi đặt tên cho 3 tiêu chí này là **"FPU Triangle"** — tam giác chất lượng dữ liệu tổng hợp.

#### 1.4 Yêu cầu chức năng (Functional Requirements)

Từ 3 câu hỏi trên, tôi phân rã thành các chức năng cụ thể:

1. **FR-1**: Hệ thống phải đọc được dữ liệu từ CSV, Excel, Parquet.
2. **FR-2**: Hệ thống phải tự động phát hiện và loại bỏ PII.
3. **FR-3**: Hệ thống phải xử lý dữ liệu thiếu (missing values) mà không làm mất thông tin.
4. **FR-4**: Hệ thống phải mã hóa được categorical features.
5. **FR-5**: Hệ thống phải chuẩn hóa continuous features.
6. **FR-6**: Hệ thống phải hỗ trợ ít nhất 3 mô hình sinh (GAN, VAE, Diffusion).
7. **FR-7**: Hệ thống phải hỗ trợ Differential Privacy (DP-SGD).
8. **FR-8**: Hệ thống phải đánh giá chất lượng theo cả 3 tiêu chí (FPU).
9. **FR-9**: Hệ thống phải tạo báo cáo Markdown + HTML.
10. **FR-10**: Hệ thống phải có CLI để người dùng gõ lệnh.

#### 1.5 Yêu cầu phi chức năng (Non-functional Requirements)

1. **NFR-1 (Reproducibility)**: Chạy 2 lần phải cho kết quả giống hệt nhau.
2. **NFR-2 (Modularity)**: Có thể thay thế mô hình mà không ảnh hưởng pipeline.
3. **NFR-3 (Memory Safety)**: Không bị OOM với dataset lớn.
4. **NFR-4 (Configurability)**: Tham số phải nằm trong file cấu hình, không hardcode.
5. **NFR-5 (Integrity)**: Dữ liệu không bị hỏng khi lưu trữ và tải lại.

---

### CHƯƠNG 2: THIẾT KẾ HỆ THỐNG (SYSTEM DESIGN)

#### 2.1 Suy nghĩ ban đầu

Tôi biết mình sẽ làm một **pipeline** — một đường ống dẫn dữ liệu. Dữ liệu đi vào đầu ống, qua từng công đoạn xử lý, và ra đầu kia là dữ liệu tổng hợp kèm báo cáo chất lượng.

Nhưng tôi KHÔNG biết trước tất cả tham số. Ví dụ:

- Tôi chưa biết dùng batch_size = 256 hay 512.
- Tôi chưa biết dùng model nào (CTGAN, CTVAE, Diffusion).
- Tôi chưa biết có bật DP hay không.

Tôi cần một **thiết kế linh hoạt**: thay đổi tham số mà không cần sửa code.

#### 2.2 Decision: YAML Configuration First

Tôi quyết định: **tham số sẽ nằm trong file YAML**. Code chỉ đọc, không hardcode.

Đây là quyết định kiến trúc đầu tiên và quan trọng nhất. Nó ảnh hưởng đến toàn bộ phần còn lại.

**Tư duy**: "Tôi không biết trước mọi thứ. Tôi sẽ tách biệt cấu hình (configuration) khỏi xử lý (processing)."

#### 2.3 Decision: 5 tầng kiến trúc (5-Layer Architecture)

Tôi phác thảo kiến trúc tổng thể:

```
Layer 1: Configuration   ← Đọc YAML, kiểm tra validation
Layer 2: Preprocessing   ← Impute → Encode → Scale
Layer 3: Modeling        ← CTGAN / CTVAE / Diffusion + DP-SGD
Layer 4: Evaluation      ← Fidelity + Privacy + Utility → Báo cáo
Layer 5: CLI / Pipeline  ← Kết nối tất cả, chạy tự động
```

**Tại sao 5 tầng?**
- Mỗi tầng có một nhiệm vụ rõ ràng.
- Có thể thay đổi một tầng mà không ảnh hưởng tầng khác.
- Dễ test, dễ bảo trì, dễ mở rộng.

---

### CHƯƠNG 3: XÂY DỰNG TỪNG BƯỚC (FILE-BY-FILE)

Bây giờ tôi bắt đầu code. Tôi sẽ tạo từng file theo đúng thứ tự phát sinh nhu cầu.

---

#### 3.1 FILE ĐẦU TIÊN: `config/default_config.yaml`

**Tại sao file này tồn tại?**

Trước khi làm bất cứ điều gì, tôi cần định nghĩa "bảng điều khiển" cho toàn bộ hệ thống. Giống như một chiếc xe hơi có bảng đồng hồ — tôi muốn mọi tham số đều có thể điều chỉnh từ một nơi duy nhất.

**Nội dung tôi nghĩ khi tạo file này:**

Tôi tự hỏi: "Mình cần những tham số gì?"

1. **ingestion**: Cách đọc dữ liệu (separator, encoding, v.v.)
2. **model**: Tham số huấn luyện (epochs, batch_size, learning_rate)
3. **privacy**: Tham số DP-SGD (epsilon, delta, max_grad_norm)

Tôi viết:

```yaml
ingestion:
  separator: ','
  has_header: true
  max_onehot_cardinality: 10
model:
  batch_size: 256
  epochs: 100
  learning_rate: 0.0002
  model_type: diffusion
  max_ram_gb: 8.0
privacy:
  enable_differential_privacy: true
  target_epsilon: 1.5
  target_delta: 1.0e-05
  max_grad_norm: 1.0
```

**Tại sao mặc định là diffusion?** Vì diffusion là SOTA (State of The Art) — mô hình mới nhất, mạnh nhất. Người dùng có thể đổi sang ctgan hoặc ctvae nếu muốn.

**Tại sao epochs = 100?** Đủ lớn để model học, nhưng không quá lớn để chờ đợi.

**Tại sao epsilon = 1.5?** Đây là mức "trung bình vàng" — vừa đủ bảo mật, vừa giữ được chất lượng dữ liệu.

**Nếu không có file này**: Tôi phải hardcode epochs, batch_size,... trong từng file. Muốn thay đổi, phải sửa nhiều chỗ.

---

#### 3.2 FILE THỨ HAI: `src/config/config_loader.py`

**Tại sao file này tồn tại?**

Có file YAML rồi, nhưng làm sao đọc nó bằng Python? Tôi cần một module chuyên đọc YAML và kiểm tra xem các giá trị có hợp lệ không.

**Tư duy thiết kế**:

Tôi dùng **Pydantic** — một thư viện kiểm tra kiểu dữ liệu (data validation). Nó tự động:
- Kiểm tra epochs có phải số nguyên dương không.
- Kiểm tra epsilon có lớn hơn 0 không.
- Báo lỗi rõ ràng nếu sai.

**Cấu trúc**:

```
AppConfig
├── IngestionConfig
├── PrivacyConfig
└── ModelConfig
```

**Tại sao dùng Pydantic thay vì `yaml.safe_load` trực tiếp?**

`yaml.safe_load` chỉ đọc file, không kiểm tra gì cả. Nếu người dùng gõ nhầm `epochs: "mot tram"`, chương trình sẽ crash ở đâu đó rất khó debug. Pydantic bắt lỗi ngay khi đọc config.

**deep_merge()**:

Tôi có config mặc định (default_config.yaml) và config riêng cho từng dataset (data_schema.yaml). `deep_merge()` hợp nhất chúng: config riêng ghi đè config mặc định.

**Nếu không có file này**: Toàn bộ hệ thống mù về cấu hình. Phải đọc YAML thủ công ở mọi nơi.

---

#### 3.3 FILE THỨ BA: `src/preprocessing/imputer.py`

**Tại sao file này tồn tại?**

Tôi mở file dữ liệu thật lên và thấy: **có ô trống**. Một số khách hàng không điền "nghề nghiệp", "thu nhập".

Tôi tự hỏi: "Xử lý thế nào?"

**Phân tích bài toán**:

| Giải pháp | Ưu điểm | Nhược điểm |
|-----------|---------|------------|
| Xóa dòng thiếu | Đơn giản | Mất dữ liệu quý |
| Điền giá trị trung bình | Giữ được dòng | Méo phân phối, mất thông tin "tại sao thiếu" |
| Tạo cột chỉ báo | Giữ được thông tin | Tăng số chiều |

Tôi chọn giải pháp **kết hợp**: Tạo cột chỉ báo (`_is_missing`) + điền median/mode.

**Chiến lược per-column**: Tôi cho phép người dùng chỉ định chiến lược cho từng cột qua `column_strategies`.

**fit() vs transform()**:
- `fit()`: Học tham số (median, mode) từ dữ liệu.
- `transform()`: Áp dụng tham số đã học.

Tách biệt fit/transform là **design pattern quan trọng** — nó cho phép fit trên tập train và áp dụng lên tập test với cùng tham số.

**Nếu không có file này**: Dữ liệu có ô trống sẽ gây lỗi DataFrame hoặc làm mô hình học sai.

---

#### 3.4 FILE THỨ TƯ: `src/preprocessing/encoder.py`

**Tại sao file này tồn tại?**

Dữ liệu có cột "Giới tính: Nam/Nữ", "Thành phố: Hà Nội/HCM/Đà Nẵng". Máy tính không hiểu chữ, chỉ hiểu số.

**Phân tích bài toán**:

| Phương pháp | Cách hoạt động | Khi nào dùng |
|-------------|----------------|--------------|
| **One-Hot** | Tạo N cột nhị phân cho N giá trị | Cardinality ≤ 10 |
| **Label Encoding** | Gán số 0, 1, 2,... cho từng giá trị | Cardinality > 10 |

**Tại sao One-Hot lại giới hạn ở 10?**

Nếu "Thành phố" có 63 giá trị, One-Hot tạo ra 63 cột mới. DataFrame phình to, tốn RAM, làm gradient trong neural network bị "loãng" (sparsity).

**Tại sao Label Encoding nguy hiểm?**

Nếu "Nam"=0, "Nữ"=1, mô hình có thể hiểu nhầm "Nữ > Nam". Tôi giảm nguy cơ này bằng **scale_labels**: chia giá trị cho K-1 để đưa về [0, 1].

**column_strategies**: Cho phép ép kiểu mã hóa cho từng cột. Nếu không chỉ định, tự động chọn dựa trên cardinality.

**inverse_transform()**: Khi model sinh dữ liệu số, tôi cần chuyển ngược lại thành chữ. Hàm này làm việc đó.

**Nếu không có file này**: Model không thể học categorical features. Dữ liệu chữ bị bỏ qua hoặc gây lỗi.

---

#### 3.5 FILE THỨ NĂM: `src/preprocessing/scaler.py`

**Tại sao file này tồn tại?**

Cột "tuổi" từ 0-100, cột "thu nhập" từ 0-10,000,000. Neural network hoạt động tốt nhất khi mọi cột cùng thang đo.

**Tư duy toán học**:

Nếu một cột có giá trị lớn hơn hàng triệu lần cột khác, nó sẽ chi phối gradient hoàn toàn. Các cột nhỏ gần như không đóng góp vào quá trình học.

**Các chiến lược**:

| Strategy | Công thức | Khi nào dùng |
|----------|-----------|--------------|
| **minmax** | (x - min) / (max - min) | Dữ liệu không có outlier, muốn giữ nguyên phân phối |
| **standard** | (x - mean) / std | Dữ liệu có phân phối chuẩn (Gaussian) |
| **log1p** | log(1 + x) → minmax | Dữ liệu lệch phải (skew > 1.5): nhiều giá trị nhỏ, vài giá trị rất lớn |

**feature_range**: Tôi cho phép scale về [-1, 1] hoặc [0, 1] tùy model. CTGAN dùng tanh → [-1, 1]. Diffusion clamp [0, 1] → [0, 1].

**inverse_transform()**: Quan trọng — sau khi model sinh dữ liệu, tôi phải đưa nó về thang đo gốc.

**Nếu không có file này**: Model sẽ bị "cột thu nhập" chi phối hoàn toàn, không học được gì từ các cột khác.

---

#### 3.6 FILE THỨ SÁU: `src/preprocessing/pipeline.py`

**Tại sao file này tồn tại?**

Tôi có 3 module riêng lẻ (Imputer, Encoder, Scaler). Nhưng mỗi lần xử lý dữ liệu, tôi phải gọi chúng theo đúng thứ tự:

```
Imputer → Encoder → Scaler
```

Tôi tự động hóa việc này bằng Pipeline.

**Tại sao thứ tự đó?**

1. **Impute trước**: Nếu còn NaN, Encoder và Scaler sẽ lỗi.
2. **Encode trước Scaler**: Scaler chỉ nên áp dụng lên cột số. Sau khi Encode, tôi mới biết cột nào là số.
3. **Scale cuối cùng**: Đảm bảo đầu ra là các cột số cùng thang đo.

**fit_transform() vs transform()**:

- `fit_transform()`: Học tham số + biến đổi. Chỉ gọi 1 lần trên tập train.
- `transform()`: Chỉ biến đổi (dùng tham số đã học). Gọi trên tập test và dữ liệu mới.

**Model-aware scaling**:

Tôi phát hiện một vấn đề: CTGAN dùng tanh activation, output nằm trong [-1, 1]. Diffusion clamp về [0, 1]. Nếu tôi scale data về [0, 1] mà dùng CTGAN, dữ liệu bị "bó hẹp" và model học rất chậm.

**Giải pháp**: Pipeline tự động nhận diện model_type:
- CTGAN/CTVAE → scale về [-1, 1]
- Diffusion → scale về [0, 1]

**inverse_transform()**:

Sau khi model sinh dữ liệu (dưới dạng số), pipeline làm ngược lại:
```
Scaler.inverse → Encoder.inverse → Imputer.inverse → Type casting
```

**SHA256 checksum**: Khi lưu pipeline ra file, tôi tính mã băm SHA256. Khi tải lại, tôi kiểm tra mã băm để phát hiện file bị sửa đổi.

**Nếu không có file này**: Tôi phải gọi Imputer, Encoder, Scaler thủ công — rất dễ quên bước hoặc sai thứ tự.

---

#### 3.7 FILE THỨ BẢY: `eda_framework/` (11 files)

**Tại sao file này tồn tại?**

Mỗi dataset mới, tôi mất hàng giờ để:
- Xác định cột nào là số, cột nào là phân loại.
- Tìm PII cần xóa.
- Kiểm tra tỷ lệ thiếu.
- Vẽ biểu đồ phân phối.
- Quyết định chiến lược encode/scale.

Tôi muốn tự động hóa toàn bộ. EDA Framework ra đời.

**Cấu trúc 7 module EDA**:

```
EDA Framework
├── validation.py      ← Stage 1: Đọc file, phát hiện delimiter, encoding
├── privacy_audit.py   ← Stage 2: Tìm PII bằng keyword + cardinality
├── missing_analysis.py← Stage 3: Phân tích thiếu, đề xuất imputation
├── profiling.py       ← Stage 4: Thống kê mô tả, outlier, skewness
├── association.py     ← Stage 5: Ma trận tương quan hỗn hợp
├── recommendation.py  ← Stage 6-7: Tự động sinh config YAML
├── utils/
│   ├── helpers.py     ← Đọc/ghi YAML, detect file properties
│   ├── statistics.py  ← Các hàm thống kê thuần túy (Cramer's V, v.v.)
│   └── plotting.py    ← Style biểu đồ
```

**Luồng chạy EDA Framework**:

```
File thô → validation.py → DataFrame sạch
                              ↓
                       privacy_audit.py → Danh sách PII cần drop
                              ↓
                       missing_analysis.py → % thiếu, đề xuất imputation
                              ↓
                       profiling.py → Skewness, outlier, encoding recommendation
                              ↓
                       association.py → Ma trận tương quan
                              ↓
                       recommendation.py → Xuất data_schema.yaml + pipeline_config.yaml
```

**Tại sao có cả statistics.py riêng?**

Tôi muốn tách biệt hoàn toàn logic thống kê (pure math) khỏi plot. `statistics.py` không import matplotlib — có thể chạy trên server không có GUI.

**Nếu không có file này**: Mỗi dataset mới tốn 2-3 giờ phân tích thủ công. EDA Framework giảm xuống còn 2-3 phút.

---

#### 3.8 FILE THỨ TÁM: `src/models/ctgan.py`, `ctvae.py`, `diffusion.py`

**Tại sao 3 model?**

Định lý **No Free Lunch** trong ML: không có model nào tốt nhất cho mọi dataset. Tôi cung cấp 3 lựa chọn:

| Model | Điểm mạnh | Điểm yếu | Khi nào dùng |
|-------|-----------|----------|-------------|
| **CTGAN** | Xử lý mất cân bằng lớp tốt | Khó train, unstable | Dữ liệu có class hiếm |
| **CTVAE** | Ổn định, dễ train | Chất lượng thấp hơn GAN | Baseline an toàn |
| **Diffusion** | Chất lượng cao nhất | Sinh chậm | Cần chất lượng cao nhất |

**Kiến trúc chung của cả 3 model**:

```
Input: (batch, data_dim) tensor từ pipeline
  ↓
Model-specific processing (GAN/VAE/Diffusion)
  ↓
Output: (batch, data_dim) tensor → pipeline inverse_transform
```

**col_meta** — "bản đồ" mô tả từng cột:

```
col_meta = [
  ColumnMeta(name="age", col_type="continuous", dim=1),
  ColumnMeta(name="education", col_type="onehot", dim=5),
  ColumnMeta(name="income", col_type="label", dim=1),
  ColumnMeta(name="age_is_missing", col_type="continuous", dim=1),
]
```

**Tại sao col_meta quan trọng?**

Model cần biết cột nào là continuous (áp dụng tanh activation), cột nào là onehot (áp dụng softmax), cột nào là label (áp dụng sigmoid).

**Conditional Sampler (dùng chung)**:

Vấn đề: Dataset mất cân bằng. 80% khách hàng không churn, 20% churn. Nếu train bình thường, model sẽ sinh 80% non-churn, 20% churn — nhưng tôi muốn nó học cả 2 lớp tốt như nhau.

**Giải pháp**: Conditional vector — mỗi batch, tôi chọn ngẫu nhiên một categorical column và một category, tạo one-hot vector. Model được "buộc" phải sinh ra category đó, kể cả khi nó hiếm.

**LayerNorm thay vì BatchNorm (rất quan trọng)**:

BatchNorm tính mean/std trên cả batch. Điều này rò rỉ thông tin giữa các sample — phá vỡ DP-SGD. LayerNorm chỉ tính trên từng sample riêng biệt, an toàn cho DP.

**Chi tiết từng model**:

**CTGAN**:
- Generator: noise → tanh(continuous) + GumbelSoftmax(onehot) + sigmoid(label)
- Discriminator: Phân biệt thật/giả. WGAN loss + Gradient Penalty.
- n_critic = 5: Discriminator train 5 lần, Generator train 1 lần.

**CTVAE**:
- Encoder: data → mu, log_var (latent)
- Decoder: z ~ N(mu, sigma) → reconstructed data
- Loss = Reconstruction (MSE + BCE + CrossEntropy) + β * KL

**Diffusion**:
- Forward: x₀ → x_T (thêm nhiễu dần)
- Reverse: x_T → x₀ (khử nhiễu dần, học bởi denoising MLP)
- Sampling: Từ noise trắng, khử nhiễu T bước.

---

#### 3.9 FILE THỨ CHÍN: `src/models/constraints.py`

**Tại sao file này tồn tại?**

Model học thống kê, không học logic kinh doanh. Nó có thể sinh ra khách hàng "dùng 0 tháng" nhưng "tổng cước > 0" — vô lý.

**Ví dụ**: `TotalCharges >= MonthlyCharges` — tổng cước không thể nhỏ hơn cước tháng.

**2 chế độ enforcement**:

1. **Soft penalty** (training): Thêm vào loss function. Nếu model vi phạm, loss tăng lên, model tự điều chỉnh.
2. **Post-correction** (sampling): Sau khi sinh dữ liệu, kiểm tra từng dòng. Nếu vi phạm, clamp giá trị hoặc sinh lại.

**Tại sao cần cả 2?**
- Soft penalty → model học luật trong quá trình train.
- Post-correction → lưới an toàn cuối cùng, bắt những gì model vẫn vi phạm.

**JS Divergence để phát hiện Over-Correction**: Nếu correction quá mạnh, phân phối data thay đổi. JS đo sự thay đổi này.

**Nếu không có file này**: Dữ liệu sinh ra có thể chứa các mẫu vô lý về mặt nghiệp vụ, gây mất niềm tin với đối tác.

---

#### 3.10 FILE THỨ MƯỜI: `src/training/dp_training.py`

**Tại sao file này tồn tại?**

Model có thể "nhớ" các mẫu huấn luyện. Nếu nó sinh ra một bản ghi giống hệt một khách hàng thật, đó là rò rỉ thông tin.

**DP-SGD giải quyết thế nào?**

Thuật toán SGD thông thường:
```
1. Tính gradient trên batch
2. Cập nhật weight: w = w - lr * gradient
```

Vấn đề: Nếu một mẫu duy nhất có gradient rất lớn (do nó khác thường), nó ảnh hưởng mạnh đến weight → model nhớ mẫu đó.

**DP-SGD thêm 2 bước**:

```
1. Clip gradient từng mẫu: g = g / max(1, ||g||/C)
   → Không mẫu nào có ảnh hưởng quá C.
2. Thêm nhiễu Gauss: g = g + N(0, C²σ²)
   → Làm mờ dấu vết của từng mẫu.
```

**Kết quả**: Dù có biết chính xác 999/1000 mẫu, cũng không thể suy ra mẫu thứ 1000.

**2 backends**:

| Backend | Nguồn gốc | Độ tin cậy |
|---------|-----------|-----------|
| **Opacus** | Thư viện chính thức của PyTorch | ✅ Đảm bảo DP toán học |
| **Custom** | Tự viết (fallback) | ⚠️ Chỉ heuristic, không formal DP |

**Tại sao có Custom backend?** Opacus có thể không cài được trên Windows. Custom backend là phương án dự phòng.

**Nếu không có file này**: Model có thể "học vẹt" dữ liệu thật, gây rò rỉ thông tin cá nhân.

---

#### 3.11 FILE THỨ MƯỜI MỘT: `src/training/trainer.py`

**Tại sao file này tồn tại?**

Tôi có 3 model, mỗi model có cách khởi tạo khác nhau. Tôi cần một **người điều phối** chung.

**Trainer làm gì?**

1. Nhận DataFrame từ pipeline.
2. Xây dựng col_meta (mô tả cấu trúc cột).
3. Chuyển DataFrame → Tensor.
4. Khởi tạo model (CTGAN/CTVAE/Diffusion).
5. Gắn ConstraintsEngine (nếu có).
6. Gắn DPTrainer (nếu DP bật).
7. Gọi model.fit().
8. Lưu checkpoint.

**seed = 42**:

Tôi set seed cho Python, NumPy, PyTorch để đảm bảo reproducibility. Cùng một seed → cùng một kết quả.

**Nếu không có file này**: Phải viết code huấn luyện riêng cho từng model — lặp lại và dễ sai.

---

#### 3.12 FILE THỨ MƯỜI HAI: `src/training/hpo.py`

**Tại sao file này tồn tại?**

Mỗi model có nhiều siêu tham số. Chọn sai → chất lượng kém. Tôi không thể thử tay từng bộ.

**Optuna Bayesian Optimization**:

Thay vì random search (thử ngẫu nhiên), tôi dùng TPE (Tree-structured Parzen Estimator) — một thuật toán Bayesian Optimization. Nó học từ các lần thử trước và tập trung vào vùng không gian có triển vọng.

**Objective function**:

```
Objective = fidelity_weight × Wasserstein + (1-fidelity_weight) × constraint_violation
```

- Wasserstein càng nhỏ → data fidelity càng tốt.
- constraint_violation càng nhỏ → càng ít vi phạm ràng buộc.

**Privacy-aware HPO**:

Khi DP bật, tổng epsilon được chia:
- 20% cho HPO (các trial tìm kiếm)
- 80% cho training cuối

**Nếu không có file này**: Phải thử tay từng bộ tham số — mất thời gian và không tối ưu.

---

#### 3.13 FILE THỨ MƯỜI BA: `src/inference/sampler.py`

**Tại sao file này tồn tại?**

Model sinh ra tensor số. Tôi cần chuyển nó thành DataFrame có tên cột, kiểu dữ liệu, giá trị con người hiểu được.

**Các bước**:

```
1. model.sample(n_rows) → tensor (batch, data_dim)
2. pipeline.inverse_transform(tensor) → DataFrame với cột gốc
3. ConstraintsEngine.post_correction(df) → sửa vi phạm
4. Xuất CSV/Parquet
```

**generate_and_save()**: Một hàm tiện lợi — sinh + lưu ra file.

**Nếu không có file này**: Phải tự viết code inverse + save mỗi lần sinh dữ liệu.

---

#### 3.14 FILE THỨ MƯỜI BỐN: `src/evaluation/` (6 files)

**Tại sao bộ evaluation này tồn tại?**

Tôi đã sinh dữ liệu. Nhưng làm sao chứng minh nó tốt? Tôi cần **3 loại bằng chứng định lượng** — FPU Triangle.

**File 1: `fidelity.py`** — Đo độ tương đồng thống kê

| Metric | Loại biến | Ý nghĩa | Ngưỡng |
|--------|-----------|---------|--------|
| Wasserstein Distance | Continuous | Khoảng cách giữa 2 phân phối | < 0.05: tốt |
| JS Divergence | Categorical | Độ lệch phân phối xác suất | < 0.03: tốt |
| Correlation Difference | Cả hai | Sai khác ma trận tương quan | < 0.1: tốt |

**Tại sao Wasserstein?** Nó đo "chi phí tối thiểu để biến phân phối A thành B". Trực quan và có ý nghĩa vật lý.

**File 2: `privacy.py`** — Đo rò rỉ thông tin

| Metric | Ý nghĩa | Lý tưởng |
|--------|---------|----------|
| DCR (Distance to Closest Record) | Khoảng cách đến bản ghi thật gần nhất | Càng lớn càng tốt |
| NNDR (Nearest Neighbor Distance Ratio) | d1/d2 — kiểm tra "bám" một cá nhân | ≈ 1.0 |
| DCR Leakage % | % synthetic gần bản ghi thật | < 1% |
| MIA AUC-ROC | Khả năng phân biệt member/non-member | ≈ 0.5 (random) |
| AIA | Dự đoán cột nhạy cảm từ synthetic | Càng thấp càng tốt |

**Tại sao DCR 5th percentile?** Thay vì dùng threshold cứng, tôi lấy phân vị thứ 5 của distribution holdout-to-train. Đây là cách thích ứng với từng dataset.

**File 3: `utility.py`** — Đo tính hữu ích ML

**TSTR (Train Synthetic, Test Real)**:
1. Train RandomForest/LogisticRegression trên synthetic data.
2. Test trên real test data.
3. So sánh với TRTR (Train Real, Test Real).

**Nếu F1-macro của TSTR đạt ≥ 90% TRTR → dữ liệu tổng hợp đạt yêu cầu**.

**File 4: `visual.py`** — Vẽ biểu đồ:
- Distribution grid: histogram/KDE cho continuous, bar chart cho categorical.
- Correlation heatmap: Real vs Synthetic vs Difference.
- DCR distribution: histogram + leakage threshold.

**File 5: `report.py`** — Tạo báo cáo Markdown + HTML.

**File 6: `orchestrator.py`** — Điều phối 5 file trên.

---

#### 3.15 FILE THỨ MƯỜI LĂM: `src/cli/main.py`

**Tại sao file này tồn tại?**

Người dùng cuối không muốn viết Python. Họ muốn gõ lệnh trong terminal.

**4 subcommands**:

| Command | Chức năng | Ví dụ |
|---------|-----------|-------|
| `train` | Huấn luyện model | `python -m src.cli.main train --dataset adult_income --model-type ctgan` |
| `sample` | Sinh dữ liệu | `python -m src.cli.main sample --dataset adult_income --n-rows 5000` |
| `evaluate` | Đánh giá | `python -m src.cli.main evaluate --real-csv real.csv --synth-csv synth.csv` |
| `hpo` | Tối ưu tham số | `python -m src.cli.main hpo --dataset adult_income --n-trials 30` |

**YAML-override pattern**: CLI flags override YAML values. Nếu không chỉ định, dùng YAML.

---

#### 3.16 FILE THỨ MƯỜI SÁU: `run_pipeline.py`

**Tại sao file này tồn tại?**

Kết nối tất cả thành một luồng duy nhất.

**Pipeline đầy đủ**:

```
1. set_global_seed(42)           ← Reproducibility
2. Load config từ YAML           ← Configuration
3. Load data từ CSV              ← Data Ingestion
4. Drop PII columns              ← Privacy Layer 1
5. Train/Test split 80/20       ← Chống data leakage
6. Preprocessing (Impute→Encode→Scale) ← Feature Engineering
7. Save pipeline artifacts       ← State serialization
8. Train model (with DP-SGD)     ← Generative Modeling
9. Save checkpoint               ← Model persistence
10. Load sampler                  ← Inference
11. Generate synthetic data       ← Sampling
12. Inverse transform             ← Decode to original space
13. Run evaluation (FPU)          ← Quality Assessment
14. Generate reports              ← Markdown + HTML
15. Print summary                 ← Results
```

**Nếu không có file này**: Phải chạy 15+ lệnh riêng lẻ — không ai làm.

---

#### 3.17 FILE THỨ MƯỜI BẢY: `run_experiments.py`

**Tại sao file này tồn tại?**

Tôi cần so sánh CTGAN vs CTVAE vs Diffusion trên cùng điều kiện. Tôi tạo một script tự động hóa việc so sánh.

**Cấu trúc thí nghiệm**:

```python
experiments = [
    {"model_type": "ctgan", "epochs": 100, "enable_dp": False},
    {"model_type": "ctvae", "epochs": 100, "enable_dp": False},
    {"model_type": "diffusion", "epochs": 100, "enable_dp": False},
    # ...
]
```

Mỗi thí nghiệm chạy pipeline riêng, kết quả được tổng hợp vào CSV.

---

#### 3.18 FILE KIỂM THỬ: `tests/` (13 files)

**Tại sao có 13 test files?**

Tôi muốn đảm bảo mỗi module hoạt động đúng trước khi tích hợp.

**Nguyên tắc**: Mỗi file source → ít nhất 1 test file tương ứng.

| Test file | Module test | Nội dung |
|-----------|-------------|----------|
| `test_encoder.py` | encoder.py | One-hot, Label, inverse |
| `test_imputer.py` | imputer.py | Missing indicators, fit/transform |
| `test_scaler.py` | scaler.py | MinMax, Standard, log1p |
| `test_pipeline.py` | pipeline.py | End-to-end preprocessing + inverse |
| `test_constraints.py` | constraints.py | Violation mask, correction |
| `test_fidelity.py` | fidelity.py | Wasserstein, JS, Cramer's V |
| `test_privacy_audit.py` | privacy.py | DCR, NNDR, MIA |
| `test_utility.py` | utility.py | TSTR/TRTR |
| `test_trainer.py` | trainer.py | Model training loop |
| `test_cli.py` | cli.py | CLI arguments parsing |
| ... | ... | ... |

---

### CHƯƠNG 4: DÒNG THỜI GIAN THỰC THI (EXECUTION FLOW)

Đây là cách hệ thống chạy khi người dùng gõ lệnh:

```
python run_pipeline.py --dataset adult_income --model-type ctgan --epochs 2
```

```

① CẤU HÌNH
   user input
      ↓
   run_pipeline.py (argparse)
      ↓
   config_loader.py (đọc YAML, validate Pydantic)
      ↓
   AppConfig (object chứa mọi tham số)
      ↓

② DỮ LIỆU
   data/adult/adult.data
      ↓
   PreprocessingPipeline.load_data()
      ↓
   DataFrame thô (15 cột, 48,000 dòng)
      ↓
   Drop PII columns
      ↓
   train_test_split 80/20
      ↓
   Real Train (38,400 dòng) ─────┐
   Real Test  (9,600 dòng)  ─────┤ (cô lập, không đụng đến)

③ TIỀN XỬ LÝ
   Real Train
      ↓
   Imputer.fit() → học median/mode
   Imputer.transform() → điền thiếu, tạo _is_missing
      ↓
   Encoder.fit() → học mapping category→index
   Encoder.transform() → one-hot/label encoding
      ↓
   Scaler.fit() → học min/max/mean/std
   Scaler.transform() → scale về [0,1] hoặc [-1,1]
      ↓
   Tensor (38,400 × ~120 cột) sẵn sàng train

④ HUẤN LUYỆN
   Trainer nhận tensor
      ↓
   build_col_meta() → ∀ cột: continuous/onehot/label
      ↓
   Khởi tạo TabularCTGAN(col_meta)
      ↓
   Nếu DP: DPTrainer wrap optimizer
      ↓
   Nếu constraints: ConstraintsEngine.soft_loss_penalty
      ↓
   model.fit(tensor, epochs=2)
      ↓
   [Epoch 1/2] G_loss=... D_loss=... GP=...
   [Epoch 2/2] G_loss=... D_loss=... GP=...
      ↓
   Lưu checkpoint → artifacts/adult_income/checkpoints/ctgan_model.pt

⑤ SINH DỮ LIỆU
   sampler.load() → load checkpoint + pipeline artifact
      ↓
   sampler.generate(n_rows=1000)
      ↓
   model.sample(1000) → tensor (1000, ~120)
      ↓
   pipeline.inverse_transform()
      ↓
   Scaler.inverse → Encoder.inverse → Imputer.inverse → Type casting
      ↓
   DataFrame (1000 dòng, 15 cột gốc)
      ↓
   ConstraintsEngine.post_correction() (nếu có)
      ↓
   df_synthetic → data/adult_income_synthetic_output.csv

⑥ ĐÁNH GIÁ
   EvaluationSuite nhận df_real + df_synthetic + df_train + df_test
      ↓
   FidelityAssessor → Wasserstein, JS, Correlation Difference
      ↓
   PrivacyAuditor → DCR, NNDR, MIA, AIA
      ↓
   UtilityEvaluator → TSTR vs TRTR (RandomForest, LogisticRegression)
      ↓
   VisualOverlayGenerator → distribution grid, correlation heatmap, DCR plot
      ↓
   ComplianceReporter → compliance_report.md + compliance_report.html

⑦ KẾT QUẢ
   Pipeline run completed successfully!
     - Average JS Divergence:  0.0234
     - MIA Attacker AUC-ROC:   0.5123
     - HTML Compliance Report: artifacts/.../compliance_report.html
```

---

### CHƯƠNG 5: GIẢI THÍCH CÁC KHÁI NIỆM HỘI ĐỒNG HAY HỎI

#### 5.1 "Tại sao chọn CTGAN mà không phải vanilla GAN?"

Vanilla GAN không xử lý được mixed-type data (số + phân loại). Nó dễ bị mode collapse (chỉ sinh 1 loại dữ liệu). CTGAN giải quyết 2 vấn đề này bằng:
- Conditional vector → oversample lớp hiếm.
- Mode-Specific Normalization (thông qua pipeline) → xử lý multi-modal distribution.

#### 5.2 "Tại sao chọn Diffusion? Nó chậm hơn GAN nhiều?"

Diffusion cho chất lượng cao nhất (SOTA). Ứng dụng của tôi là sinh dữ liệu offline — sinh 1 lần, dùng nhiều lần. Thời gian sampling lâu (1000 bước reverse) là chấp nhận được.

#### 5.3 "DP-SGD làm giảm chất lượng dữ liệu thế nào?"

DP-SGD thêm nhiễu → model khó học hơn → chất lượng synthetic giảm. Đây là trade-off:
- ε lớn (ví dụ 10): Ít nhiễu, quality cao, privacy thấp.
- ε nhỏ (ví dụ 0.1): Nhiều nhiễu, quality thấp, privacy cao.

Tôi chọn ε = 1.5 — cân bằng giữa quality và privacy.

#### 5.4 "Tại sao layer normalization thay vì batch normalization?"

BatchNorm tính mean/std trên cả batch → rò rỉ thông tin giữa các sample → phá vỡ DP guarantee. LayerNorm chỉ tính trên từng sample → an toàn cho DP.

#### 5.5 "Làm sao đảm bảo dữ liệu tổng hợp không sao chép dữ liệu thật?"

3 lớp bảo vệ:
1. **DCR**: Đo khoảng cách từ synthetic → real gần nhất. Cảnh báo nếu quá gần.
2. **MIA AUC**: Nếu ≈ 0.5, chứng tỏ không thể phân biệt synthetic member vs non-member.
3. **DP-SGD**: Đảm bảo toán học rằng model không thể nhớ bất kỳ mẫu cụ thể nào.

---

### CHƯƠNG 6: SƠ ĐỒ PHỤ THUỘC (DEPENDENCY GRAPH)

```
run_experiments.py
  └── run_pipeline.py
        ├── config_loader.py
        │     └── default_config.yaml
        ├── pipeline.py
        │     ├── imputer.py
        │     ├── encoder.py
        │     └── scaler.py
        ├── trainer.py
        │     ├── ctgan.py
        │     ├── ctvae.py
        │     ├── diffusion.py
        │     ├── constraints.py
        │     └── dp_training.py
        ├── sampler.py
        │     ├── pipeline.py
        │     └── ctgan/ctvae/diffusion.py
        └── orchestrator.py
              ├── fidelity.py
              ├── privacy.py
              ├── utility.py
              ├── visual.py
              └── report.py
```

---

**LỜI KẾT**: 

Bây giờ bạn đã có trong đầu toàn bộ câu chuyện phát triển dự án. Khi hội đồng hỏi:

- "Tại sao file X tồn tại?" → Bạn kể câu chuyện: "Khi tôi gặp vấn đề Y, tôi cần giải pháp Z..."
- "Tại sao chọn model A thay vì B?" → Bạn giải thích trade-off.
- "Làm sao đảm bảo không rò rỉ dữ liệu?" → Bạn liệt kê 3 lớp bảo vệ.

Bạn không cần học thuộc. Bạn cần **hiểu logic** — và logic đó chính là câu chuyện tôi vừa kể.