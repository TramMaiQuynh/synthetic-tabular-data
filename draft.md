# Synthetic Tabular Data Generation Platform — System Design
# Nền tảng Sinh Dữ liệu Dạng Bảng Tổng hợp — Thiết kế Hệ thống

---

## Module Architecture Overview / Tổng quan Kiến trúc Module

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT DATA                               │
│                     (Raw Tabular Data)                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         MODULE 1: Data Ingestion & Preprocessing                │
│   De-identification → Schema Inference → Imputation →           │
│   Encoding & Normalization → Schema Constraint Extraction       │
└────────────────────────────┬────────────────────────────────────┘
                             │  schema artifact + preprocessed data
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         MODULE 2: Tabular Data Generation Engine                │
│   Generative Modeling → DP-SGD Training →                        │
│   Memorization Mitigation → HPO →                                │
│   Scalable Sampling → Business Logic Constraints Engine →        │
│   Inverse Transform & Post-processing                            │
└────────────────────────────┬────────────────────────────────────┘
                             │  synthetic dataset
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         MODULE 3: Evaluation, Validation & Audit                │
│   Statistical Fidelity → Privacy Leakage Audit →                │
│   ML Utility (TSTR) → Compliance Reporting                      │
└────────────────────────────┬────────────────────────────────────┘
                             │  audit report + validated synthetic data
                             ▼
                        OUTPUT DATA
                  (Validated Synthetic Dataset)

─────────────────────────────────────────────────────────────────
  Privacy & Compliance Architecture  [Cross-cutting concern]
  Spans: Module 1 (De-id) → Module 2 (DP-SGD) → Module 3 (Audit)
─────────────────────────────────────────────────────────────────
```

---

## Module 1: Data Ingestion & Preprocessing
## Phân hệ 1: Tiếp nhận & Tiền xử lý Dữ liệu

**Purpose / Mục đích:**
Chuyển đổi dữ liệu đầu vào thô về định dạng chuẩn hóa — bao gồm khử nhận dạng, xác định kiểu dữ liệu, xử lý giá trị khuyết thiếu, mã hóa và chuẩn hóa đặc trưng, và trích xuất ràng buộc cấu trúc — nhằm đảm bảo tính tương thích toán học và nhất quán logic trước khi đưa vào phân hệ sinh dữ liệu. Module này là điểm kiểm soát đầu vào duy nhất của toàn pipeline.

Converts raw input data into a standardized format — encompassing de-identification, data type determination, missing value resolution, feature encoding and normalization, and structural constraint extraction — to ensure mathematical compatibility and logical consistency prior to ingestion into the generation engine. This module serves as the sole input control point of the entire pipeline.

---

### 1.1 De-identification & Anonymization / Khử nhận dạng & Ẩn danh hóa

**VI:** Xóa hoặc che giấu các định danh trực tiếp (họ tên, số định danh cá nhân, số tài khoản, địa chỉ) khỏi tập dữ liệu đầu vào trước khi bất kỳ thành phần nào khác của hệ thống tiếp cận dữ liệu. Các định danh gián tiếp (quasi-identifiers) như mã bưu chính, ngày sinh, giới tính được giữ nguyên trạng thái chi tiết nhất trong lúc huấn luyện (trừ các trường hợp đặc biệt không dùng DP-SGD thì mới áp dụng generalization hoặc bucketing) để tối ưu hóa khả năng học phân phối mịn của mô hình sinh. Nhiệm vụ bảo mật toán học sẽ được đảm nhận bởi Differential Privacy (DP-SGD) ở Module 2 để chống Linkage Attack. Đây là bước đầu tiên trong pipeline, thực thi trước Schema Inference.

**EN:** Removes or masks direct identifiers (full names, national identification numbers, account numbers, addresses) from the input dataset before any other system component accesses the data. Quasi-identifiers (such as ZIP codes, dates of birth, and gender) are preserved in their raw/high-granularity state during training (unless DP-SGD is disabled, in which case generalization or bucketing may be applied as fallback) to optimize the generative model's ability to learn fine-grained distributions. Mathematical privacy guarantees are deferred to Differential Privacy (DP-SGD) in Module 2 to prevent Linkage Attacks. This is the first step in the pipeline, executed prior to Schema Inference.

**Ghi chú kỹ thuật / Technical note:** Bước này là điều kiện tiên quyết cho Differential Privacy ở Module 2. DP cung cấp bảo đảm toán học chống rò rỉ thông tin gián tiếp từ quasi-identifiers thô; De-identification loại bỏ rủi ro rò rỉ trực tiếp. Hai cơ chế bổ sung cho nhau, không thay thế nhau. / This step is a prerequisite for Differential Privacy in Module 2. DP provides mathematical guarantees against indirect information leakage from raw quasi-identifiers; De-identification eliminates direct leakage risk. The two mechanisms are complementary, not interchangeable.

---

### 1.2 Automatic Schema Inference / Suy luận Lược đồ Tự động

**VI:** Phân tích tập dữ liệu đầu vào để tự động xác định kiểu dữ liệu của từng cột (liên tục, rời rạc, phân loại, thời gian) và trích xuất các ràng buộc cấu trúc quan sát được từ dữ liệu thật (phạm vi giá trị, cardinality, phân phối tần suất). Kết quả là một schema artifact được lưu lại và truyền xuống toàn bộ pipeline.

**EN:** Analyzes the input dataset to automatically determine the data type of each column (continuous, discrete, categorical, datetime) and extracts structural constraints observed from the real data (value ranges, cardinality, frequency distributions). The output is a schema artifact that is persisted and propagated throughout the entire pipeline.

**Ghi chú kỹ thuật / Technical note:** Schema artifact là đầu vào cho Business Logic Constraints Engine ở Module 2 (làm baseline để phát hiện vi phạm) và cho Statistical Fidelity Assessment ở Module 3 (làm reference distribution). / The schema artifact serves as input to the Business Logic Constraints Engine in Module 2 (as a baseline for violation detection) and to Statistical Fidelity Assessment in Module 3 (as the reference distribution).

---

### 1.3 Missing Value Imputation / Điền khuyết Dữ liệu

**VI:** Để tránh tạo ra các đỉnh nhọn nhân tạo (artificial spikes) làm méo mó phân phối gốc mà mô hình sinh sẽ học, hệ thống hạn chế tối đa việc điền khuyết tĩnh (mean/median/mode) trước huấn luyện. Thay vào đó, hệ thống sử dụng cột nhị phân bổ trợ (`is_missing`) để đánh dấu trạng thái thiếu cho từng đặc trưng và điền giá trị khuyết bằng một giá trị đặc biệt nằm ngoài miền phân phối thực tế (ví dụ: -999) để mô hình sinh học trạng thái khuyết thiếu như một phân phối xác suất tự nhiên. Phương pháp điền khuyết tĩnh (mean/median cho cột liên tục, mode cho cột phân loại) chỉ áp dụng làm fallback khi cấu hình yêu cầu cụ thể. Đối với dữ liệu chuỗi thời gian tuần tự (sequential/time-series), forward-fill hoặc interpolation được áp dụng. Phương pháp và tỷ lệ khuyết thiếu được ghi nhận vào schema artifact.

**EN:** To avoid creating artificial spikes that distort the original distribution learned by the generative model, the system minimizes pre-training static imputation (mean/median/mode). Instead, it utilizes binary auxiliary columns (`is_missing`) to flag missing states for each feature and fills missing cells with a distinct out-of-range value (e.g., -999), allowing the generative model to learn the missing state as a natural probability distribution. Static imputation (mean/median for continuous, mode for categorical) is only applied as a fallback when specifically configured. For sequential/time-series data, forward-fill or interpolation is applied. The imputation strategy and missing rate per column are recorded in the schema artifact.

---

### 1.4 Categorical Encoding & Feature Normalization / Mã hóa Phân loại & Chuẩn hóa Đặc trưng

**VI:** Chuẩn hóa các cột liên tục thông qua MinMax scaling hoặc standard scaling. Để tránh phá hủy cấu trúc nhóm của các biến phân loại trong quá trình huấn luyện mạng nơ-ron (ví dụ: mode-specific normalization của CTGAN), các cột phân loại được giữ nguyên dưới dạng nhãn gốc (raw labels) trong tập dữ liệu trung gian, và việc mã hóa (one-hot hoặc representation mapping) được trì hoãn cho bộ tiền xử lý nội bộ của mô hình sinh, sử dụng schema artifact làm hướng dẫn. Tất cả các phép biến đổi được lưu lại dưới dạng transformation artifact để áp dụng nghịch đảo khi decode dữ liệu tổng hợp đầu ra.

**EN:** Prepares continuous columns via MinMax scaling or standard scaling to build a transformation artifact for decoding. To prevent destroying the structural grouping of categorical variables in neural network training (e.g., in CTGAN's mode-specific normalization), categorical features are kept as raw labels in the intermediate preprocessed dataset, and their encoding (one-hot or representation mapping) is deferred to the generator model's internal preprocessor using the schema artifact as a guide. All transformations are persisted to enable inverse transformation when decoding synthetic output data.

---

## Module 2: Tabular Data Generation Engine
## Phân hệ 2: Cỗ máy Sinh Dữ liệu Dạng Bảng

**Purpose / Mục đích:**
Huấn luyện mô hình sinh trên dữ liệu đã tiền xử lý và tạo ra tập dữ liệu tổng hợp thỏa mãn đồng thời ba yêu cầu: (1) độ trung thực thống kê so với phân phối gốc, (2) các ràng buộc nghiệp vụ do người dùng định nghĩa, và (3) các bảo đảm quyền riêng tư hình thức.

Trains a generative model on preprocessed data and produces a synthetic dataset satisfying three simultaneous requirements: (1) statistical fidelity relative to the original distribution, (2) user-defined business logic constraints, and (3) formal privacy guarantees.

---

### 2.1 Conditional Generative Modeling / Mô hình Sinh Có Điều kiện

**VI:** Sử dụng các kiến trúc học sâu được thiết kế cho dữ liệu dạng bảng kiểu hỗn hợp, bao gồm Conditional Tabular GAN (CTGAN), Conditional Tabular Variational Autoencoder (CTVAE), và Tabular Diffusion Model. Các kiến trúc này mô hình hóa phân phối xác suất kết hợp của toàn bộ tập đặc trưng, xử lý mất cân bằng lớp thông qua conditional vector representation và training-by-sampling. Kiến trúc được sử dụng trong một lần chạy cụ thể do người dùng chỉ định hoặc do HPO lựa chọn tự động.

**EN:** Employs deep learning architectures designed for mixed-type tabular data, including Conditional Tabular GAN (CTGAN), Conditional Tabular Variational Autoencoder (CTVAE), and Tabular Diffusion Model. These architectures model the joint probability distribution across the full feature set, handling class imbalance via conditional vector representations and training-by-sampling strategies. The architecture used in a specific run is either user-specified or automatically selected by HPO.

---

### 2.2 Differential Privacy Integration / Tích hợp Differential Privacy

**VI:** Tích hợp cơ chế Differential Privacy vào training loop thông qua DP-SGD (Differentially Private Stochastic Gradient Descent): gradient của từng mẫu huấn luyện được cắt ngưỡng (gradient clipping) và nhiễu Gaussian được tiêm vào trước khi cập nhật tham số mô hình. Cơ chế này cung cấp bảo đảm toán học hình thức rằng sự hiện diện hay vắng mặt của bất kỳ một mẫu huấn luyện nào cũng không thể được suy luận từ mô hình đã huấn luyện, được đặc trưng bởi tham số privacy budget (ε, δ). Privacy budget được theo dõi và báo cáo sau mỗi lần huấn luyện.

**EN:** Integrates Differential Privacy into the training loop via DP-SGD (Differentially Private Stochastic Gradient Descent): per-sample gradients are clipped and Gaussian noise is injected prior to parameter updates. This mechanism provides a formal mathematical guarantee that the presence or absence of any individual training record cannot be inferred from the trained model, characterized by privacy budget parameters (ε, δ). Privacy budget is tracked and reported after each training run.

**Ghi chú kỹ thuật / Technical note:** DP-SGD làm chậm hội tụ và có thể giảm chất lượng thống kê của mô hình. Đây là trade-off có thể điều chỉnh thông qua ε: ε nhỏ = privacy mạnh hơn, fidelity thấp hơn. Giá trị ε phải được quyết định trước khi huấn luyện và ghi nhận vào audit report. Cách áp dụng DP-SGD phụ thuộc vào kiến trúc: với GAN, DP-SGD chỉ áp dụng trực tiếp lên Discriminator (Generator kế thừa bảo đảm DP thông qua post-processing theorem); với VAE và Diffusion Model, DP-SGD áp dụng trực tiếp lên toàn bộ mạng. / DP-SGD slows convergence and may reduce the model's statistical quality. This is a configurable trade-off via ε: smaller ε = stronger privacy, lower fidelity. The ε value must be determined before training and recorded in the audit report. The application of DP-SGD is architecture-dependent: for GANs, DP-SGD is applied directly to the Discriminator only (the Generator inherits DP guarantees via the post-processing theorem); for VAE and Diffusion architectures, DP-SGD is applied directly to the entire network.

---

### 2.3 Memorization Mitigation / Hạn chế Ghi nhớ Mẫu

**VI:** Giới hạn khả năng mô hình ghi nhớ các mẫu huấn luyện riêng lẻ thông qua các kỹ thuật regularization trong quá trình huấn luyện: weight decay để hạn chế độ phức tạp của mô hình. Khi Differential Privacy (DP-SGD) bị tắt, early stopping được sử dụng để ngăn overfitting khi validation loss không cải thiện sau một số epoch xác định. Khi DP-SGD được bật, early stopping phải bị vô hiệu hóa (sử dụng số epoch cố định) hoặc thực thi thông qua private validation metrics có thêm nhiễu để bảo toàn bảo đảm toán học của privacy budget. Các kỹ thuật này bổ sung cho DP-SGD: DP-SGD cung cấp bảo đảm hình thức, regularization giảm thiểu rủi ro thực tế.

**EN:** Limits the model's capacity to memorize individual training instances via regularization techniques applied during training: weight decay to constrain model complexity. When Differential Privacy (DP-SGD) is disabled, early stopping is used to prevent overfitting when validation loss fails to improve after a defined number of epochs. When DP-SGD is enabled, early stopping is either disabled (using a fixed epoch count) or executed via private validation metrics with added noise to preserve the differential privacy budget mathematical guarantee. These techniques complement DP-SGD: DP-SGD provides formal guarantees, regularization reduces empirical risk.

---

### 2.4 Automated Hyperparameter Optimization (HPO) / Tối ưu hóa Siêu tham số Tự động

**VI:** Tự động tìm kiếm cấu hình huấn luyện tối ưu trên không gian siêu tham số bao gồm: learning rate, batch size, số epoch, hệ số regularization (weight decay), độ sâu và chiều rộng mạng nơ-ron, và — trong trường hợp dùng DP-SGD — noise multiplier và max gradient norm. Phương pháp tìm kiếm mặc định là Bayesian Optimization. Mục tiêu tối ưu hóa là tổ hợp có trọng số của statistical fidelity score và constraint satisfaction rate trên tập validation.

**EN:** Automatically searches for the optimal training configuration over a hyperparameter space including: learning rate, batch size, number of epochs, regularization coefficient (weight decay), network depth and width, and — when using DP-SGD — noise multiplier and max gradient norm. The default search method is Bayesian Optimization. The optimization objective is a weighted combination of statistical fidelity score and constraint satisfaction rate on the validation set.

**Ghi chú kỹ thuật / Technical note:** Khi DP-SGD được bật, không gian siêu tham số tối ưu (đặc biệt là batch size lớn và learning rate ổn định) khác biệt hoàn toàn so với mô hình không có DP. Do đó, việc chạy HPO trực tiếp dưới điều kiện DP-SGD là cần thiết nhưng tiêu thụ privacy budget. Để cân bằng, hệ thống áp dụng chiến lược HPO bảo mật (Private HPO/DP-Optuna) với budget phân bổ riêng, hoặc áp dụng các heuristic tối ưu hóa DP-SGD được cấu hình sẵn (như cố định batch size lớn tỷ lệ thuận với $\sqrt{N}$) thay vì bê nguyên cấu hình tối ưu từ Non-DP sang. / When DP-SGD is enabled, the optimal hyperparameter space (especially large batch sizes and stable learning rates) differs fundamentally from non-DP training. Running HPO directly under DP-SGD is necessary but consumes privacy budget. To balance this, the system applies private HPO strategies (Private HPO/DP-Optuna) with a dedicated budget allocation, or enforces pre-configured DP-SGD tuning heuristics (such as fixing a large batch size proportional to $\sqrt{N}$) rather than directly transferring optimal configurations from non-DP runs.

---

### 2.5 Scalable Batched Inference & Sampling / Lấy mẫu Theo lô Quy mô Lớn

**VI:** Sinh tập dữ liệu tổng hợp ở quy mô tùy cấu hình từ mô hình sinh đã huấn luyện, sử dụng batched inference để kiểm soát mức tiêu thụ bộ nhớ và tránh lỗi OOM (Out-of-Memory). Giao diện sampling hỗ trợ hai chế độ: (1) unconditional generation — sinh N dòng dữ liệu phản ánh phân phối gốc, (2) conditional generation — sinh N dòng dữ liệu thuộc một nhóm điều kiện chỉ định (ví dụ: chỉ sinh dữ liệu của nhóm khách hàng nợ xấu) để giải quyết mất cân bằng lớp.

**EN:** Generates synthetic datasets at configurable scale from the trained generator model, using batched inference to control memory consumption and prevent OOM (Out-of-Memory) errors. The sampling interface supports two modes: (1) unconditional generation — generates N rows reflecting the original distribution, (2) conditional generation — generates N rows belonging to a specified condition group (e.g., generating only non-performing loan customer records) to address class imbalance.

---

### 2.6 Business Logic Constraints Engine / Cỗ máy Ràng buộc Nghiệp vụ

**VI:** Thực thi các ràng buộc quan hệ chéo (cross-column) và logic điều kiện do người dùng định nghĩa lên dữ liệu tổng hợp đầu ra, độc lập với chất lượng dữ liệu huấn luyện và hành vi của mô hình sinh. Ràng buộc được khai báo một lần và biên dịch thành một execution graph thống nhất hỗ trợ bốn cơ chế thực thi theo thứ tự ưu tiên:

**EN:** Enforces user-defined cross-column relational and conditional logic constraints on the synthetic output data, independent of training data quality and generative model behavior. Constraints are declared once and compiled into a unified execution graph supporting four enforcement mechanisms in order of priority:

| Cơ chế / Mechanism | Thời điểm / Timing | Mô tả / Description |
|---|---|---|
| Mathematical Transformations | Trước huấn luyện / Pre-training | Biến đổi dữ liệu để constraint được học trực tiếp vào mô hình / Transforms data so constraints are learned directly into the model |
| Soft Loss Penalties | Trong huấn luyện / During training | Phạt hàm mất mát khi mô hình sinh ra giá trị vi phạm constraint (chỉ áp dụng cho các ràng buộc liên tục khả vi) / Penalizes the loss function when the model generates constraint-violating values (restricted to differentiable continuous constraints) |
| Guided Sampling | Trong lấy mẫu / During sampling | Hướng dẫn quá trình giải mã để ưu tiên các giá trị thỏa mãn constraint (áp dụng cho cả ràng buộc rời rạc/phân loại) / Guides the decoding process to favor constraint-satisfying values (applicable to discrete/categorical constraints) |
| Post-generation Correction | Sau sinh / Post-generation | Phát hiện và hiệu chỉnh các vi phạm còn lại; cấu hình retry limit và fallback policy để tránh vòng lặp vô hạn / Detects and corrects remaining violations; configures retry limits and fallback policies to prevent infinite loops |

**Ghi chú kỹ thuật / Technical note:** Soft Loss Penalties chỉ áp dụng cho các kiến trúc có training loop (CTGAN, CTVAE) và yêu cầu các biểu thức ràng buộc phải khả vi (differentiable constraints). Với Tabular Diffusion Model, cơ chế này được thay thế bằng Guided Sampling trong quá trình khử nhiễu (reverse diffusion). Để cân bằng độ trung thực thống kê, hiệu năng sinh và tỷ lệ thỏa mãn ràng buộc, hệ thống hỗ trợ cấu hình ngưỡng tỷ lệ vi phạm chấp nhận được (violation rate threshold). Đồng thời, hệ thống đo lường độ lệch phân phối (distribution divergence/JS divergence) giữa dữ liệu sinh thô và dữ liệu sau hiệu chỉnh ràng buộc để cảnh báo rủi ro "Constraint Over-Correction" (hiệu chỉnh quá đà gây lệch phân phối gốc). / Soft Loss Penalties apply only to architectures with explicit training loops (CTGAN, CTVAE) and require the constraint expressions to be mathematically differentiable. For Tabular Diffusion Models, this mechanism is replaced by Guided Sampling during the reverse diffusion steps. To balance statistical fidelity, sampling throughput, and constraint satisfaction, the system supports a configurable violation rate threshold. Additionally, the system measures the distribution divergence (JS divergence) between raw generated data and post-corrected data to warn against the risk of "Constraint Over-Correction" distorting the original distribution.

---

### 2.7 Inverse Transformation & Post-processing / Giải mã Ngược & Hậu xử lý

**VI:** Chuyển đổi dữ liệu tổng hợp từ không gian chuẩn hóa (normalized space) về không gian giá trị gốc (original value space) theo trình tự sau:
1. **Nghịch đảo phép biến đổi:** Áp dụng nghịch đảo các phép biến đổi đã lưu trong transformation artifact (inverse MinMax scaling, giải mã nhãn phân loại) để đưa toàn bộ giá trị — bao gồm cả giá trị sentinel đã bị scale — về không gian giá trị gốc.
2. **Giải mã trạng thái khuyết thiếu:** Kiểm tra các cột nhị phân bổ trợ `is_missing` do mô hình sinh ra — nếu `is_missing = 1`, thay giá trị tương ứng ở cột gốc bằng `NULL/NaN` thật; nếu `is_missing = 0`, giữ nguyên giá trị đã được inverse transform. Sau đó loại bỏ toàn bộ các cột `is_missing` phụ trợ khỏi tập dữ liệu đầu ra.
3. **Hậu xử lý ép kiểu:** Ép kiểu dữ liệu (type casting) theo schema artifact — làm tròn các cột kiểu integer, clamp giá trị ngoài biên (boundary clamping), và định dạng lại các cột datetime.

Bước này đảm bảo dữ liệu đầu ra ở dạng có thể sử dụng trực tiếp bởi người dùng cuối và các hệ thống nghiệp vụ hạ nguồn.

**Ghi chú kỹ thuật / Technical note:** Thứ tự bước 1 → 2 là bắt buộc: giá trị sentinel (ví dụ: -999) đã bị biến đổi bởi MinMax scaling ở bước tiền xử lý, nên tồn tại trong normalized space dưới dạng một giá trị scaled khác. Inverse scaling phải được thực hiện trước để đưa tất cả giá trị về original space, sau đó mới kiểm tra `is_missing` flag và thay thế bằng NULL/NaN. / Step 1 → 2 ordering is mandatory: sentinel values (e.g., -999) were transformed by MinMax scaling during preprocessing and exist in normalized space as a different scaled value. Inverse scaling must be applied first to return all values to original space before inspecting `is_missing` flags and replacing with NULL/NaN.

**EN:** Converts synthetic data from normalized space back to the original value space in the following sequence:
1. **Inverse transformation:** Applies the inverse of transformations stored in the transformation artifact (inverse MinMax scaling, categorical label decoding) to return all values — including scaled sentinel values — to the original value space.
2. **Missing state resolution:** Inspects the binary auxiliary `is_missing` columns generated by the model — if `is_missing = 1`, replaces the corresponding value in the original column with a true `NULL/NaN`; if `is_missing = 0`, retains the inverse-transformed value. All auxiliary `is_missing` columns are then dropped from the output dataset.
3. **Type-casting post-processing:** Enforces data type casting per the schema artifact — rounding integer-typed columns, boundary clamping for out-of-range values, and reformatting datetime columns.

This step ensures the output data is in a directly consumable format for end users and downstream business systems.

---

## Module 3: Evaluation, Validation & Audit
## Phân hệ 3: Đánh giá, Kiểm định & Kiểm toán

**Purpose / Mục đích:**
Định lượng chất lượng của tập dữ liệu tổng hợp trên ba chiều độc lập — độ trung thực thống kê, an toàn quyền riêng tư, và tính hữu dụng trong machine learning — và tổng hợp kết quả thành báo cáo kiểm toán có cấu trúc.

Quantifies the quality of the synthetic dataset across three independent dimensions — statistical fidelity, privacy safety, and machine learning utility — and consolidates results into a structured audit report.

---

### 3.1 Statistical Fidelity Assessment / Đánh giá Độ trung thực Thống kê

**VI:** Tính toán các độ đo khoảng cách thống kê giữa phân phối của tập dữ liệu thật và tập dữ liệu tổng hợp trên cả phân phối biên (marginal) và phân phối kết hợp (joint): Wasserstein distance cho các cột liên tục, Jensen-Shannon divergence cho các cột phân loại, và ma trận chênh lệch tương quan chéo (cross-correlation difference matrix) để đánh giá mức độ bảo toàn cấu trúc phụ thuộc giữa các cặp đặc trưng.

**EN:** Computes statistical distance metrics between the distributions of the real and synthetic datasets across both marginal and joint distributions: Wasserstein distance for continuous columns, Jensen-Shannon divergence for categorical columns, and a cross-correlation difference matrix to assess the preservation of pairwise feature dependency structure.

---

### 3.2 Privacy Leakage Audit / Kiểm toán Rò rỉ Quyền riêng tư

**VI:** Đo lường mức độ cô lập không gian và tính chống chịu tấn công của dữ liệu tổng hợp thông qua hai nhóm đánh giá:
1. *Đánh giá khoảng cách hình học:* Distance to Closest Record (DCR) — khoảng cách từ mỗi điểm tổng hợp đến bản ghi thật gần nhất, và Nearest Neighbor Distance Ratio (NNDR) — tỷ lệ giữa khoảng cách đến bản ghi thật gần nhất và bản ghi thật gần thứ hai. DCR thấp bất thường chỉ ra nguy cơ sao chép bản ghi (privacy kém). NNDR tiệm cận 1 là kết quả mong muốn — cho thấy điểm tổng hợp không "dính" vào bất kỳ bản ghi thật cụ thể nào (privacy tốt); ngược lại, NNDR tiệm cận 0 chỉ ra điểm tổng hợp cực kỳ gần một bản ghi thật duy nhất (nguy cơ memorization cao). Để đảm bảo tính bất biến thang đo (scale-invariance), DCR/NNDR được tính trên không gian đặc trưng đã chuẩn hóa (normalized feature space). Với các tập dữ liệu lớn, sử dụng Tìm kiếm Lân cận Gần đúng (ANN) tăng tốc bằng GPU (Faiss hoặc HNSW) để đảm bảo khả năng mở rộng.
2. *Tấn công thực nghiệm giả lập (Empirical Attacker Simulation):* Tích hợp bộ giả lập chạy thử tấn công Membership Inference Attack (MIA) và Attribute Inference Attack (AIA) để đánh giá khả năng phòng ngự thực tế của dữ liệu tổng hợp. Báo cáo tỷ lệ đoán đúng của kẻ tấn công (yêu cầu tiệm cận đoán ngẫu nhiên ~50% đối với MIA).

**EN:** Measures the spatial isolation and attack resistance of synthetic data via two assessment types:
1. *Geometric distance evaluation:* Distance to Closest Record (DCR) — the distance from each synthetic point to its nearest real record, and Nearest Neighbor Distance Ratio (NNDR) — the ratio of the distance to the nearest real record versus the second nearest. To ensure scale-invariance, DCR/NNDR are computed on the normalized feature space. For large datasets, GPU-accelerated Approximate Nearest Neighbor (ANN) search (Faiss or HNSW) is used to maintain scalability.
2. *Empirical Attacker Simulation:* Integrates simulators to perform mock Membership Inference Attacks (MIA) and Attribute Inference Attacks (AIA) to evaluate empirical privacy risk. The success rate of mock attacks is reported (targeting a random guess baseline of ~50% for MIA).

---

### 3.3 ML Utility Evaluation — TSTR Framework / Đánh giá Hiệu năng Máy học — Khung TSTR

**VI:** Kiểm định tính hữu dụng của tập dữ liệu tổng hợp theo khung Train-on-Synthetic, Test-on-Real (TSTR). Tập dữ liệu thật được phân tách nghiêm ngặt thành $Real_{train}$ và $Real_{test}$ (holdout). Mô hình sinh được huấn luyện trên $Real_{train}$ để xuất ra tập $Synthetic$. Các mô hình dự đoán chuẩn (XGBoost, Random Forest, Logistic Regression) được huấn luyện trên tập $Synthetic$ và đánh giá trên tập $Real_{test}$. Kết quả được so sánh với baseline Train-on-Real, Test-on-Real (TRTR) (trong đó mô hình dự đoán được huấn luyện trên $Real_{train}$ và đánh giá trên $Real_{test}$) để tính mức suy hao hiệu năng (performance degradation) theo các chỉ số F1-score, AUC-ROC, và Accuracy, đảm bảo không rò rỉ dữ liệu (zero data leakage).

**EN:** Validates the utility of the synthetic dataset using the Train-on-Synthetic, Test-on-Real (TSTR) framework. The real dataset is strictly partitioned into $Real_{train}$ and $Real_{test}$ (holdout). The generator model is trained on $Real_{train}$ to output the $Synthetic$ dataset. Standard predictive models (XGBoost, Random Forest, Logistic Regression) are trained on the $Synthetic$ dataset and evaluated on the held-out $Real_{test}$ dataset. Results are compared against the Train-on-Real, Test-on-Real (TRTR) baseline (where predictive models are trained on $Real_{train}$ and evaluated on $Real_{test}$) to compute performance degradation across F1-score, AUC-ROC, and Accuracy metrics, ensuring zero data leakage.

---

### 3.4 Visual Distribution Overlays / Biểu đồ Trực quan hóa Phân phối

**VI:** Tự động sinh các biểu đồ so sánh trực quan đặt phân phối thật và phân phối tổng hợp chồng lên nhau: histogram theo từng cột, scatter plot cho các cặp đặc trưng liên tục, và correlation heatmap. Biểu đồ phục vụ kiểm tra trực quan nhanh bên cạnh các chỉ số định lượng.

**EN:** Automatically generates visual comparison charts overlaying real and synthetic distributions: per-column histograms, scatter plots for continuous feature pairs, and correlation heatmaps. Charts serve as rapid visual inspection alongside quantitative metrics.

---

### 3.5 Automated Compliance Reporting / Báo cáo Tuân thủ Tự động

**VI:** Tổng hợp toàn bộ kết quả đánh giá thành một báo cáo kiểm toán có cấu trúc (định dạng PDF và HTML) bao gồm: chỉ số tương đồng thống kê, kết quả kiểm toán quyền riêng tư (DCR, NNDR, giá trị ε đã sử dụng), và bảng suy hao hiệu năng máy học. Báo cáo được đánh phiên bản và liên kết với schema artifact và transformation artifact của lần chạy tương ứng.

**EN:** Consolidates all evaluation results into a structured audit report (PDF and HTML formats) comprising: statistical similarity metrics, privacy audit results (DCR, NNDR, applied ε value), and the machine learning performance degradation matrix. The report is versioned and linked to the schema artifact and transformation artifact of the corresponding run.

---

## Privacy & Compliance Architecture
## Kiến trúc Quyền riêng tư & Tuân thủ

*Cross-cutting concern — không phải một module độc lập trong pipeline mà là một tập hợp các cơ chế phân bố qua nhiều module, phối hợp để đảm bảo toàn bộ pipeline tuân thủ các yêu cầu bảo vệ dữ liệu.*

*Cross-cutting concern — not an independent pipeline module but a set of mechanisms distributed across multiple modules that coordinate to ensure the entire pipeline satisfies data protection requirements.*

### Thứ tự thực thi các cơ chế privacy / Execution order of privacy mechanisms

```
Module 1 — De-identification & Anonymization
    Loại bỏ định danh trực tiếp trước khi bất kỳ thành phần nào xử lý dữ liệu.
    Removes direct identifiers before any component processes the data.
          │
          ▼
Module 2 — Differential Privacy (DP-SGD)
    Tiêm nhiễu vào gradient để ngăn mô hình ghi nhớ mẫu riêng lẻ.
    Injects noise into gradients to prevent the model from memorizing individual records.
    Privacy budget (ε, δ) được tiêu thụ và ghi nhận tại đây.
    Privacy budget (ε, δ) is consumed and recorded here.
          │
          ▼
Module 2 — Memorization Mitigation
    Regularization bổ sung giảm thiểu rủi ro ghi nhớ thực tế.
    Additional regularization reduces empirical memorization risk.
          │
          ▼
Module 3 — Privacy Leakage Audit (DCR, NNDR)
    Kiểm tra thực nghiệm rằng không có bản ghi nào bị sao chép.
    Empirically verifies that no records have been replicated.
          │
          ▼
Module 3 — Compliance Report
    Tổng hợp toàn bộ bằng chứng privacy thành tài liệu kiểm toán.
    Consolidates all privacy evidence into an audit document.
```

### Bảo vệ Artifact / Artifact Protection

**VI:** Schema artifact và transformation artifact chứa các giá trị biên (min/max) và ánh xạ category cụ thể của tập dữ liệu thật — bản thân chúng là thông tin nhạy cảm. Các artifact này phải được lưu trữ trong môi trường cô lập, mã hóa khi lưu trữ (encryption at rest) và chỉ có tài khoản dịch vụ (service account) của pipeline mới có quyền truy cập.

**EN:** Schema artifacts and transformation artifacts contain boundary values (min/max) and specific category mappings of the real dataset — they are sensitive information in their own right. These artifacts must be stored in an isolated environment, encrypted at rest, and accessible only to the pipeline's service account.

### Ánh xạ tới tiêu chuẩn tuân thủ / Mapping to compliance standards

| Yêu cầu / Requirement | Cơ chế đáp ứng / Satisfying mechanism | Module |
|---|---|---|
| Không xử lý PII trực tiếp / No direct PII processing | De-identification (direct identifiers) + DP-SGD (quasi-identifiers) | 1, 2 |
| Data residency — dữ liệu không rời khỏi hạ tầng nội bộ | Localized Training Pipeline | 2 |
| Bảo đảm toán học chống re-identification | DP-SGD (ε, δ guarantee) | 2 |
| Kiểm tra không sao chép bản ghi / Empirical privacy verification | DCR / NNDR audit + MIA / AIA Attacker Simulation | 3 |
| Tài liệu kiểm toán có thể truy vết | Versioned Compliance Report | 3 |
| Bảo vệ artifact nhạy cảm / Sensitive artifact protection | Encryption at rest + access control | 1, 2, 3 |