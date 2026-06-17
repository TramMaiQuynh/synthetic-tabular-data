# Cấu Trúc Mục Lục Báo Cáo Nghiên Cứu Khoa Học

**Đề tài:** Sinh Dữ liệu Dạng Bảng Tổng hợp Bảo toàn Quyền riêng tư bằng Trí tuệ Nhân tạo Sinh (Privacy-Preserving Tabular Data Synthesis via Generative AI)

> [!NOTE]
> Cấu trúc này tuân thủ chuẩn trình bày báo cáo nghiên cứu ML theo best practices của các hội nghị NeurIPS, ICML, AAAI, và các luận văn thạc sĩ CNTT tại Việt Nam. Mỗi mục đều có ghi chú ánh xạ trực tiếp tới các module/file mã nguồn trong dự án.

---

## Phần mở đầu (Ngoài chương)

- **Trang bìa** — Tên đề tài, Tên tác giả, Giảng viên hướng dẫn, Trường/Khoa, Năm
- **Lời cảm ơn**
- **Tóm tắt (Abstract)** — 200–300 từ, tóm gọn bài toán, phương pháp, kết quả chính
- **Mục lục**
- **Danh mục bảng biểu**
- **Danh mục hình ảnh**
- **Danh mục từ viết tắt** — CTGAN, CTVAE, DP-SGD, DCR, NNDR, TSTR, TRTR, JSD, MIA, AIA, HPO, BCE, ELBO, KL, ...

---

## Chương 1: Tổng quan & Phát biểu Bài toán

### 1.1 Đặt vấn đề (Problem Statement)
> Trình bày nhu cầu thực tiễn từ doanh nghiệp tài chính, viễn thông, y tế: dữ liệu nhạy cảm không thể chia sẻ cho đối tác/AI vendor, hạn chế nghiên cứu nội bộ, mất cân bằng lớp trong dữ liệu gian lận.
> 📎 Tham chiếu: [README.md — Problem Statement](file:///T:/coding/synthetic-tabular-data/README.md#L22-L28)

### 1.2 Mục tiêu nghiên cứu (Research Objectives)
> Ba mục tiêu đồng thời: (1) Sinh dữ liệu trung thực thống kê, (2) Bảo toàn quyền riêng tư, (3) Đảm bảo tính hữu dụng cho ML downstream.
> 📎 Tham chiếu: [README.md — Proposed Solution](file:///T:/coding/synthetic-tabular-data/README.md#L30-L34)

### 1.3 Phạm vi nghiên cứu (Scope)
> Giới hạn trên dữ liệu dạng bảng (tabular data) kiểu hỗn hợp (mixed-type: continuous + categorical). Không bao gồm dữ liệu ảnh, văn bản, hay chuỗi thời gian phức tạp.

### 1.4 Đóng góp chính (Key Contributions)
> Liệt kê 4–5 đóng góp kỹ thuật cụ thể: hệ thống end-to-end pipeline, tích hợp 3 kiến trúc sinh, cơ chế ràng buộc nghiệp vụ 4 tầng, hệ thống đánh giá 3 chiều tự động, tích hợp DP-SGD.

### 1.5 Cấu trúc báo cáo (Report Organization)
> Tóm tắt nội dung từng chương.

---

## Chương 2: Cơ sở Lý thuyết & Công trình Liên quan

### 2.1 Mạng đối sinh (Generative Adversarial Networks — GAN)
> Trình bày kiến trúc Generator–Discriminator, hàm mục tiêu minimax, vấn đề mode collapse và training instability.

### 2.2 Conditional Tabular GAN (CTGAN)
> Training-by-sampling, conditional vector, mode-specific normalization cho dữ liệu dạng bảng hỗn hợp (Xu et al., 2019).
> 📎 Tham chiếu: [ctgan.py](file:///T:/coding/synthetic-tabular-data/src/models/ctgan.py)

### 2.3 Variational Autoencoder (VAE) & Conditional Tabular VAE (CTVAE)
> Encoder–Decoder, hàm mục tiêu ELBO = Reconstruction Loss + KL Divergence, reparameterization trick. Giải thích vai trò của $\beta$-VAE trong cân bằng tái tạo vs. regularization.
> 📎 Tham chiếu: [ctvae.py](file:///T:/coding/synthetic-tabular-data/src/models/ctvae.py)

### 2.4 Mô hình Khuếch tán (Denoising Diffusion Probabilistic Models — DDPM)
> Forward process (thêm nhiễu Gaussian theo lịch trình $\beta_t$), reverse process (mạng khử nhiễu $\epsilon_\theta$), lịch trình tuyến tính vs cosine, công thức lấy mẫu DDPM/DDIM.
> 📎 Tham chiếu: [diffusion.py](file:///T:/coding/synthetic-tabular-data/src/models/diffusion.py)

### 2.5 Bảo mật vi phân (Differential Privacy)
> Định nghĩa $(ε, δ)$-Differential Privacy, cơ chế Gaussian, thuật toán DP-SGD (Abadi et al., 2016), per-sample gradient clipping, Moments Accountant / RDP Accountant. Phân biệt DP-SGD chuẩn vs. xấp xỉ heuristic batch-level clipping.
> 📎 Tham chiếu: [dp_training.py](file:///T:/coding/synthetic-tabular-data/src/training/dp_training.py)

### 2.6 Các phương pháp đánh giá dữ liệu tổng hợp
> - Wasserstein Distance cho phân phối biên liên tục
> - Jensen-Shannon Divergence cho phân phối biên phân loại
> - KL Divergence trong ràng buộc phân phối
> - Distance to Closest Record (DCR) & Nearest Neighbor Distance Ratio (NNDR)
> - Membership Inference Attack (MIA) & Attribute Inference Attack (AIA)
> - Khung TSTR (Train-on-Synthetic, Test-on-Real)
>
> 📎 Tham chiếu: [fidelity.py](file:///T:/coding/synthetic-tabular-data/src/evaluation/fidelity.py), [privacy.py](file:///T:/coding/synthetic-tabular-data/src/evaluation/privacy.py), [utility.py](file:///T:/coding/synthetic-tabular-data/src/evaluation/utility.py)

### 2.7 Tối ưu hóa siêu tham số Bayesian (Bayesian HPO)
> Tree-structured Parzen Estimator (TPE), tích hợp Optuna, hàm mục tiêu đa tiêu chí (fidelity + constraint satisfaction).
> 📎 Tham chiếu: [hpo.py](file:///T:/coding/synthetic-tabular-data/src/training/hpo.py)

### 2.8 Các công trình liên quan (Related Work)
> So sánh với SDV (Synthetic Data Vault), CTAB-GAN, TabDDPM, GOGGLE, DP-MERF, v.v. Nêu rõ điểm khác biệt/ưu việt của hệ thống đề xuất.

---

## Chương 3: Thiết kế Hệ thống & Kiến trúc Phần mềm

### 3.1 Tổng quan kiến trúc 3 phân hệ (System Architecture Overview)
> Sơ đồ kiến trúc pipeline: Module 1 (Tiền xử lý) → Module 2 (Sinh dữ liệu) → Module 3 (Đánh giá). Privacy là cross-cutting concern xuyên suốt.
> 📎 Tham chiếu: [README.md — System Architecture](file:///T:/coding/synthetic-tabular-data/README.md#L43-L82), [draft.md](file:///T:/coding/synthetic-tabular-data/draft.md)

### 3.2 Phân hệ 1: Tiếp nhận & Tiền xử lý Dữ liệu (Data Ingestion & Preprocessing)

#### 3.2.1 Suy luận lược đồ tự động (Automatic Schema Inference)
> 📎 Tham chiếu: [pipeline.py](file:///T:/coding/synthetic-tabular-data/src/preprocessing/pipeline.py), thư mục [config/](file:///T:/coding/synthetic-tabular-data/config)

#### 3.2.2 Xử lý giá trị khuyết thiếu (Missing Value Imputation)
> Cơ chế cột phụ trợ `is_missing`, sentinel value, chiến lược fallback.
> 📎 Tham chiếu: [imputer.py](file:///T:/coding/synthetic-tabular-data/src/preprocessing/imputer.py)

#### 3.2.3 Mã hóa đặc trưng phân loại (Categorical Encoding)
> One-hot encoding cho cardinality thấp, Label encoding cho cardinality cao, chuẩn hóa nhãn về $[0, 1]$ để tương thích với BCE loss.
> 📎 Tham chiếu: [encoder.py](file:///T:/coding/synthetic-tabular-data/src/preprocessing/encoder.py)

#### 3.2.4 Chuẩn hóa đặc trưng liên tục (Feature Scaling)
> MinMax scaling, Standard scaling, xử lý cột hằng số ($rng = 0$).
> 📎 Tham chiếu: [scaler.py](file:///T:/coding/synthetic-tabular-data/src/preprocessing/scaler.py)

#### 3.2.5 Quản lý Artifact & Pipeline tuần tự hóa (Pipeline Serialization)
> Lưu trữ/nạp trạng thái pipeline qua joblib, tham số hóa đường dẫn.
> 📎 Tham chiếu: [pipeline.py](file:///T:/coding/synthetic-tabular-data/src/preprocessing/pipeline.py)

### 3.3 Phân hệ 2: Cỗ máy Sinh Dữ liệu (Tabular Data Generation Engine)

#### 3.3.1 Kiến trúc CTGAN
> Generator (MLP + BatchNorm/ReLU), Discriminator (PacGAN), Conditional Vector, Training-by-Sampling.
> 📎 Tham chiếu: [ctgan.py](file:///T:/coding/synthetic-tabular-data/src/models/ctgan.py)

#### 3.3.2 Kiến trúc CTVAE
> Encoder, Decoder, Reparameterization, hàm ELBO loss tùy biến (BCE cho label, MSE cho continuous, CrossEntropy cho one-hot).
> 📎 Tham chiếu: [ctvae.py](file:///T:/coding/synthetic-tabular-data/src/models/ctvae.py)

#### 3.3.3 Kiến trúc Tabular Diffusion
> Denoising MLP, Sinusoidal Time Embedding, Forward/Reverse process, lịch trình $\beta_t$ (linear/cosine).
> 📎 Tham chiếu: [diffusion.py](file:///T:/coding/synthetic-tabular-data/src/models/diffusion.py)

#### 3.3.4 Cơ chế ràng buộc nghiệp vụ mềm (Soft Constraints Engine)
> Phân tích cú pháp ràng buộc, biên dịch thành execution graph, 4 tầng thực thi (pre-training → soft loss → guided sampling → post-correction), giám sát trôi phân phối sau hiệu chỉnh (JS divergence).
> 📎 Tham chiếu: [constraints.py](file:///T:/coding/synthetic-tabular-data/src/models/constraints.py)

#### 3.3.5 Tích hợp Differential Privacy (DP-SGD)
> Backend Opacus vs Custom, per-sample clipping, Gaussian noise injection, Moments Accountant, cấu hình privacy budget $(ε, δ)$.
> 📎 Tham chiếu: [dp_training.py](file:///T:/coding/synthetic-tabular-data/src/training/dp_training.py)

#### 3.3.6 Tối ưu hóa siêu tham số tự động (Automated HPO)
> Bayesian Optimization qua Optuna/TPE, hàm mục tiêu đa tiêu chí, không gian tìm kiếm tùy biến theo kiến trúc và chế độ DP.
> 📎 Tham chiếu: [hpo.py](file:///T:/coding/synthetic-tabular-data/src/training/hpo.py)

#### 3.3.7 Lấy mẫu có điều kiện & không điều kiện (Conditional/Unconditional Sampling)
> Batched inference, conditional generation theo nhóm biến phân loại, inverse transform & post-processing.
> 📎 Tham chiếu: [sampler.py](file:///T:/coding/synthetic-tabular-data/src/inference/sampler.py)

### 3.4 Phân hệ 3: Đánh giá, Kiểm định & Kiểm toán (Evaluation & Audit)

#### 3.4.1 Đánh giá độ trung thực thống kê (Statistical Fidelity)
> Wasserstein Distance (continuous), Jensen-Shannon Divergence (categorical), Cross-correlation difference matrix (Pearson, Cramér's V, Correlation Ratio).
> 📎 Tham chiếu: [fidelity.py](file:///T:/coding/synthetic-tabular-data/src/evaluation/fidelity.py)

#### 3.4.2 Kiểm toán rò rỉ quyền riêng tư (Privacy Leakage Audit)
> DCR, NNDR, ngưỡng rò rỉ động (5th-percentile holdout), giả lập tấn công MIA (distance-based), giả lập tấn công AIA (attribute inference).
> 📎 Tham chiếu: [privacy.py](file:///T:/coding/synthetic-tabular-data/src/evaluation/privacy.py)

#### 3.4.3 Đánh giá hiệu năng ML (TSTR Framework)
> TSTR vs TRTR, RandomForest, GradientBoosting, Linear/LogisticRegression, F1-Macro, R², Accuracy.
> 📎 Tham chiếu: [utility.py](file:///T:/coding/synthetic-tabular-data/src/evaluation/utility.py)

#### 3.4.4 Trực quan hóa phân phối (Visual Distribution Overlays)
> Feature distribution grid, Correlation heatmap comparison, DCR distribution curve.
> 📎 Tham chiếu: [visual.py](file:///T:/coding/synthetic-tabular-data/src/evaluation/visual.py)

#### 3.4.5 Báo cáo tuân thủ tự động (Automated Compliance Reporting)
> Sinh báo cáo Markdown & HTML, badge trạng thái tự động, tích hợp hình ảnh.
> 📎 Tham chiếu: [report.py](file:///T:/coding/synthetic-tabular-data/src/evaluation/report.py), [orchestrator.py](file:///T:/coding/synthetic-tabular-data/src/evaluation/orchestrator.py)

### 3.5 Giao diện Dòng lệnh (CLI Interface)
> 4 câu lệnh con: `train`, `sample`, `evaluate`, `hpo`. Hỗ trợ override cấu hình YAML, DP flags, constraint injection.
> 📎 Tham chiếu: [main.py](file:///T:/coding/synthetic-tabular-data/src/cli/main.py)

### 3.6 Cấu trúc thư mục & Quản lý cấu hình (Directory Layout & Configuration)
> Sơ đồ cây thư mục dự án, hệ thống cấu hình YAML phân tầng (default → dataset-specific), artifact management.
> 📎 Tham chiếu: [config_loader.py](file:///T:/coding/synthetic-tabular-data/src/config/config_loader.py), thư mục [config/](file:///T:/coding/synthetic-tabular-data/config)

---

## Chương 4: Phương pháp Thực nghiệm (Experimental Methodology)

### 4.1 Dữ liệu thực nghiệm (Datasets)
> Mô tả chi tiết 9 tập dữ liệu EDA đã phân tích (Adult Income, Bank Marketing, Breast Cancer Wisconsin, Census, Heart Disease, Credit Card Fraud, Credit Card Default, Online Shoppers, Telco Customer Churn). Cho mỗi tập: số dòng, số cột, kiểu biến, tỷ lệ lớp, tỷ lệ khuyết thiếu.
> 📎 Tham chiếu: thư mục [notebooks/](file:///T:/coding/synthetic-tabular-data/notebooks), thư mục [data/](file:///T:/coding/synthetic-tabular-data/data)

### 4.2 Phân tích dữ liệu khám phá (Exploratory Data Analysis — EDA)
> Tóm tắt các kết quả EDA chính: phân phối biên, tương quan, phát hiện ngoại lai, mất cân bằng lớp.
> 📎 Tham chiếu: 9 notebooks EDA ([01](file:///T:/coding/synthetic-tabular-data/notebooks/01_eda_adult_income.ipynb) → [09](file:///T:/coding/synthetic-tabular-data/notebooks/09_eda_telco_customer_chum.ipynb))

### 4.3 Thiết lập thí nghiệm (Experimental Setup)
> - Môi trường phần cứng/phần mềm (Python 3.11, PyTorch, scikit-learn, ...)
> - Siêu tham số huấn luyện mặc định và tối ưu (epochs, batch_size, lr, hidden_dim, ...)
> - Cấu hình DP-SGD ($ε$, $δ$, max_grad_norm, noise_multiplier)
> - Chiến lược chia dữ liệu (train/test split)
>
> 📎 Tham chiếu: [default_config.yaml](file:///T:/coding/synthetic-tabular-data/config/default_config.yaml), các thư mục cấu hình dataset-specific

### 4.4 Quy trình đánh giá (Evaluation Protocol)
> Mô tả chính xác quy trình 5 bước: (1) Tiền xử lý, (2) Huấn luyện mô hình sinh, (3) Sinh dữ liệu tổng hợp, (4) Chạy bộ đánh giá 3 chiều, (5) Sinh báo cáo tuân thủ.
> 📎 Tham chiếu: [run_pipeline.py](file:///T:/coding/synthetic-tabular-data/run_pipeline.py)

### 4.5 Các chỉ số đánh giá (Evaluation Metrics)
> Bảng tổng hợp:

| Chiều đánh giá | Chỉ số | Ý nghĩa | Ngưỡng tốt |
|---|---|---|---|
| Fidelity (Liên tục) | Wasserstein Distance | Khoảng cách phân phối biên | < 0.05 |
| Fidelity (Phân loại) | Jensen-Shannon Divergence | Độ lệch phân phối nhãn | < 0.03 |
| Fidelity (Kết hợp) | Correlation Difference | Bảo toàn cấu trúc tương quan | < 0.10 |
| Privacy | DCR Mean | Khoảng cách trung bình đến bản ghi thật gần nhất | Càng cao càng tốt |
| Privacy | DCR Leakage % | Tỷ lệ bản ghi tổng hợp quá gần bản ghi thật | < 1% |
| Privacy | NNDR Mean | Tỷ lệ khoảng cách lân cận gần nhất | → 1.0 |
| Privacy | MIA AUC | Khả năng đoán thành viên của kẻ tấn công | → 0.50 |
| Utility | TSTR vs TRTR (F1/R²) | Suy hao hiệu năng ML | < 5% |

---

## Chương 5: Kết quả Thực nghiệm & Phân tích (Results & Analysis)

### 5.1 Kết quả trên tập dữ liệu Telco Customer Churn (Case Study chính)

#### 5.1.1 Kết quả độ trung thực thống kê
> Bảng Wasserstein distance và JSD theo từng cột, ma trận tương quan so sánh. Kèm hình ảnh từ biểu đồ phân phối và heatmap.
> 📎 Tham chiếu: [distributions_grid.png](file:///T:/coding/synthetic-tabular-data/artifacts/telco_customer_churn/evaluation/plots/distributions_grid.png), [correlation_comparison.png](file:///T:/coding/synthetic-tabular-data/artifacts/telco_customer_churn/evaluation/plots/correlation_comparison.png)

#### 5.1.2 Kết quả kiểm toán quyền riêng tư
> DCR mean/min, NNDR mean, DCR leakage %, MIA AUC, AIA accuracy. Kèm biểu đồ phân phối DCR.
> 📎 Tham chiếu: [dcr_distribution.png](file:///T:/coding/synthetic-tabular-data/artifacts/telco_customer_churn/evaluation/plots/dcr_distribution.png)

#### 5.1.3 Kết quả hiệu năng ML (TSTR)
> Bảng so sánh TRTR vs TSTR theo từng mô hình (RandomForest, GradientBoosting, Logistic Regression), F1-Macro, R², Accuracy.

#### 5.1.4 Báo cáo tuân thủ tự động
> Trích xuất nội dung từ báo cáo tự động sinh ra.
> 📎 Tham chiếu: [compliance_report.md](file:///T:/coding/synthetic-tabular-data/artifacts/telco_customer_churn/evaluation/compliance_report.md), [compliance_report.html](file:///T:/coding/synthetic-tabular-data/artifacts/telco_customer_churn/evaluation/compliance_report.html)

### 5.2 So sánh giữa các kiến trúc mô hình sinh (CTGAN vs CTVAE vs Diffusion)
> Bảng so sánh đa chiều: fidelity, privacy, utility, thời gian huấn luyện, số tham số.

### 5.3 Ảnh hưởng của Differential Privacy đến chất lượng dữ liệu
> Thí nghiệm với các mức $ε$ khác nhau (1.0, 3.0, 10.0, ∞). Biểu đồ trade-off privacy–fidelity.

### 5.4 Hiệu quả của Cơ chế Ràng buộc Nghiệp vụ
> So sánh tỷ lệ vi phạm ràng buộc trước/sau khi bật soft constraint penalty. Đo lường JS divergence giữa dữ liệu sinh thô vs sau hiệu chỉnh.

### 5.5 Kết quả HPO (Tối ưu hóa Siêu tham số)
> Biểu đồ hội tụ Optuna, cấu hình tối ưu tìm được, so sánh hiệu năng trước/sau HPO.

---

## Chương 6: Thảo luận & Hạn chế (Discussion & Limitations)

### 6.1 Thảo luận kết quả (Discussion)
> Phân tích sâu các kết quả: tại sao mô hình X cho fidelity tốt hơn Y? DP-SGD ảnh hưởng như thế nào? Cơ chế ràng buộc cải thiện/suy giảm gì?

### 6.2 Hạn chế (Limitations)
> - Custom DP-SGD backend chỉ cung cấp bảo đảm xấp xỉ (batch-level clipping), không phải per-sample
> - Chưa hỗ trợ dữ liệu chuỗi thời gian và dữ liệu quan hệ đa bảng
> - Đánh giá chỉ trên tập dữ liệu bảng kích thước vừa (< 200K dòng)
> - MIA/AIA simulation là mock attacker, chưa phải adversarial training thực sự

### 6.3 Hướng phát triển tương lai (Future Work)
> - Tích hợp Opacus hoàn chỉnh cho DP-SGD chuẩn trên mọi nền tảng
> - Hỗ trợ dữ liệu quan hệ đa bảng (multi-table relational synthesis)
> - Tích hợp TabDDPM và các kiến trúc Transformer-based mới
> - Triển khai REST API và giao diện web cho người dùng không chuyên

---

## Chương 7: Kết luận (Conclusion)
> Tóm tắt đóng góp chính, kết quả đạt được, ý nghĩa thực tiễn.

---

## Phần phụ lục (Ngoài chương)

- **Tài liệu tham khảo (References)** — Định dạng IEEE hoặc APA
- **Phụ lục A: Cấu trúc thư mục mã nguồn** — Cây thư mục đầy đủ
- **Phụ lục B: Hướng dẫn cài đặt & sử dụng** — Từ clone → install → train → sample → evaluate
- **Phụ lục C: Bảng siêu tham số chi tiết** — Bảng đầy đủ các siêu tham số mặc định và phạm vi HPO
- **Phụ lục D: Kết quả EDA trên các tập dữ liệu khác** — Tóm tắt từ 9 notebooks
- **Phụ lục E: Mã nguồn trích dẫn** — Các đoạn code quan trọng (hàm loss, hàm DCR, DP-SGD wrapper, ...)

---

## Ghi chú về Phong cách Trình bày

> [!TIP]
> - **Mỗi công thức toán học** ($D_{KL}$, $D_{JS}$, Wasserstein, ELBO, DP-SGD noise scale) nên được trình bày chính thức với ký hiệu LaTeX
> - **Mỗi bảng kết quả** nên có caption rõ ràng và được tham chiếu trong văn bản
> - **Mỗi hình ảnh/biểu đồ** từ hệ thống đánh giá nên được nhúng trực tiếp và có chú thích
> - **Mỗi thuật toán** (DP-SGD, DDPM sampling, Constraint enforcement) nên có pseudocode dạng Algorithm block
> - **So sánh** nên dùng bảng đa chiều thay vì chỉ mô tả bằng văn bản

> [!IMPORTANT]
> **Tài liệu tham khảo chính (Key References):**
> - Xu et al., 2019 — *Modeling Tabular Data using Conditional GAN* (CTGAN)
> - Kingma & Welling, 2014 — *Auto-Encoding Variational Bayes* (VAE)
> - Ho et al., 2020 — *Denoising Diffusion Probabilistic Models* (DDPM)
> - Abadi et al., 2016 — *Deep Learning with Differential Privacy* (DP-SGD)
> - Stadler et al., 2022 — *Synthetic Data — Anonymisation Groundhog Day* (MIA)
> - Esteban et al., 2017 — *Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs* (TSTR framework)
