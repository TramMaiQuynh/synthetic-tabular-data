<h1 align="center">Privacy-Preserving Tabular Data Synthesis via Generative AI</h1>

<p align="center">
  <img src="link-to-your-banner-or-demo.gif" alt="Project Demo" width="600">
</p>

<p align="center">
  <a href="#about">About</a> •
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#dataset">Dataset</a> •
  <a href="#training">Training</a> •
  <a href="#inference">Inference</a> •
  <a href="#results">Results</a> •
  <a href="#weights">Weights</a>
</p>

---

## About

### Problem Statement
Financial institutions, banks, healthcare providers, and telecommunications companies (collectively referred to as enterprises) possess valuable data assets, such as transaction histories and medical records. While there is a strong demand to leverage this data—either by contracting external AI companies or enabling internal data science teams—to train risk forecasting and personalization models, enterprises face significant barriers:
* **Regulatory Compliance**: Privacy regulations (such as GDPR in Europe, HIPAA in the US, and national cybersecurity laws) strictly prohibit exporting raw customer data to external environments or allowing development engineers direct access. Traditional anonymization techniques (e.g., masking names, phone numbers, or email addresses) are insufficient, as attackers can utilize linkage attacks to correlate transaction patterns and re-identify users.
* **Outsourcing and Partnership Barriers**: Enterprises cannot share raw datasets with external partners or AI startups to collaborate on analytical models.
* **Internal Data Silos**: Internal AI engineers are restricted from transferring customer data to development/testing environments (Dev/Test Cloud) or local workstations for daily research and programming due to leakage risks.
* **Imbalanced Data**: Critical applications, such as fraud detection, often suffer from a severe scarcity of minority-class samples (e.g., fraudulent transactions) in real-world datasets, leading to model bias and sub-optimal predictive performance.
* **Restricted Open Innovation**: Enterprises are unable to release datasets for hackathons, academic collaborations, or data insight commercialization.

### Proposed Solution & Objectives
This project focuses on the design and training of Generative AI architectures, specifically Conditional Tabular GAN (CTGAN) and Tabular Diffusion models, to address these limitations:
* **Synthetic Data Generation**: The generative model learns the underlying distribution of real tabular data and generates synthetic datasets.
* **Privacy by Design**: The synthetic datasets contain records of non-existent entities, mitigating legal and compliance risks, with provisions for Differential Privacy.
* **High Fidelity**: The synthetic data preserves the statistical distribution of the original dataset. When used to train machine learning models (e.g., XGBoost or Random Forest for credit risk assessment), the performance of the models trained on synthetic data should reach 95% to 99% of the accuracy achieved by training on the real dataset (referred to as Machine Learning Efficacy).

### Deliverables & Evaluation Framework

The deliverables of this project include a core **Data Generation Engine** and an automated **Evaluation Framework (Audit Framework)** designed to validate the synthetic data against three critical criteria:
1. **Statistical Similarity**: Validated using mathematical distance metrics to prove that the joint distribution, marginal distributions, and cross-correlations of the synthetic dataset achieve a high degree of fidelity (typically >= 95% match) compared to the original dataset.
2. **Privacy Preservation**: Demonstrated by ensuring the generative model outputs novel samples and does not memorize, replicate, or leak any individual records from the training dataset.
3. **Machine Learning Efficacy**: Verified by training downstream predictive models (e.g., XGBoost, Random Forest) on the synthetic data and evaluating them on a holdout real dataset. The performance degradation (e.g., accuracy loss, F1-score drop) must remain minimal (< 5%) compared to models trained on the original real dataset.

## System Architecture

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

## Features
### Data Ingestion & Preprocessing Module

Converts raw input data into a standardized format — encompassing de-identification, data type determination, missing value resolution, feature encoding and normalization, and structural constraint extraction — to ensure mathematical compatibility and logical consistency prior to ingestion into the generation engine. This module serves as the sole input control point of the entire pipeline.

* **De-identification & Anonymization**: Removes or masks direct identifiers (full names, national identification numbers, account numbers, addresses) from the input dataset before any other system component accesses the data. Quasi-identifiers (such as ZIP codes, dates of birth, and gender) are preserved in their raw/high-granularity state during training (unless DP-SGD is disabled, in which case generalization or bucketing may be applied as fallback) to optimize the generative model's ability to learn fine-grained distributions. Mathematical privacy guarantees are deferred to Differential Privacy (DP-SGD) in Module 2 to prevent Linkage Attacks. This is the first step in the pipeline, executed prior to Schema Inference. *De-identification eliminates direct leakage risk; DP provides mathematical guarantees against indirect information leakage. The two mechanisms are complementary, not interchangeable.*
* **Automatic Schema Inference**: Analyzes the input dataset to automatically determine the data type of each column (continuous, discrete, categorical, datetime) and extracts structural constraints observed from the real data (value ranges, cardinality, frequency distributions). The output is a schema artifact that is persisted and propagated throughout the entire pipeline. The schema artifact serves as input to the Business Logic Constraints Engine in Module 2 (as a baseline for violation detection) and to Statistical Fidelity Assessment in Module 3 (as the reference distribution).
* **Missing Value Imputation**: To avoid creating artificial spikes that distort the original distribution learned by the generative model, the system minimizes pre-training static imputation (mean/median/mode). Instead, it utilizes binary auxiliary columns (`is_missing`) to flag missing states for each feature and fills missing cells with a distinct out-of-range value (e.g., -999), allowing the generative model to learn the missing state as a natural probability distribution. Static imputation (mean/median for continuous, mode for categorical) is only applied as a fallback when specifically configured. For sequential/time-series data, forward-fill or interpolation is applied. The imputation strategy and missing rate per column are recorded in the schema artifact.
* **Categorical Encoding & Feature Normalization**: Prepares continuous columns via MinMax scaling or standard scaling to build a transformation artifact for decoding. To prevent destroying the structural grouping of categorical variables in neural network training (e.g., in CTGAN's mode-specific normalization), categorical features are kept as raw labels in the intermediate preprocessed dataset, and their encoding (one-hot or representation mapping) is deferred to the generator model's internal preprocessor using the schema artifact as a guide. All transformations are persisted to enable inverse transformation when decoding synthetic output data.

### Tabular Data Generation Engine

Trains a generative model on preprocessed data and produces a synthetic dataset satisfying three simultaneous requirements: (1) statistical fidelity relative to the original distribution, (2) user-defined business logic constraints, and (3) formal privacy guarantees.

* **Conditional Generative Modeling**: Employs deep learning architectures designed for mixed-type tabular data, including Conditional Tabular GAN (CTGAN), Conditional Tabular Variational Autoencoder (CTVAE), and Tabular Diffusion Model. These architectures model the joint probability distribution across the full feature set, handling class imbalance via conditional vector representations and training-by-sampling strategies. The architecture used in a specific run is either user-specified or automatically selected by HPO.
* **Differential Privacy Integration**: Integrates Differential Privacy into the training loop via DP-SGD (Differentially Private Stochastic Gradient Descent): per-sample gradients are clipped and Gaussian noise is injected prior to parameter updates. This mechanism provides a formal mathematical guarantee that the presence or absence of any individual training record cannot be inferred from the trained model, characterized by privacy budget parameters (ε, δ). Privacy budget is tracked and reported after each training run. *The application of DP-SGD is architecture-dependent: for GANs, DP-SGD is applied directly to the Discriminator only (the Generator inherits DP guarantees via the post-processing theorem); for VAE and Diffusion architectures, DP-SGD is applied directly to the entire network.*
* **Memorization Mitigation**: Limits the model's capacity to memorize individual training instances via regularization techniques applied during training: weight decay to constrain model complexity. When Differential Privacy (DP-SGD) is disabled, early stopping is used to prevent overfitting when validation loss fails to improve after a defined number of epochs. When DP-SGD is enabled, early stopping is either disabled (using a fixed epoch count) or executed via private validation metrics with added noise to preserve the differential privacy budget mathematical guarantee. These techniques complement DP-SGD: DP-SGD provides formal guarantees, regularization reduces empirical risk.
* **Automated Hyperparameter Optimization (HPO)**: Automatically searches for the optimal training configuration over a hyperparameter space including: learning rate, batch size, number of epochs, regularization coefficient (weight decay), network depth and width, and — when using DP-SGD — noise multiplier and max gradient norm. The default search method is Bayesian Optimization. The optimization objective is a weighted combination of statistical fidelity score and constraint satisfaction rate on the validation set. *When DP-SGD is enabled, the optimal hyperparameter space (especially large batch sizes and stable learning rates) differs fundamentally from non-DP training. The system applies private HPO strategies (Private HPO/DP-Optuna) with a dedicated budget allocation, or enforces pre-configured DP-SGD tuning heuristics (such as fixing a large batch size proportional to √N) rather than directly transferring optimal configurations from non-DP runs.*
* **Scalable Batched Inference & Sampling**: Generates synthetic datasets at configurable scale from the trained generator model, using batched inference to control memory consumption and prevent OOM (Out-of-Memory) errors. The sampling interface supports two modes: (1) unconditional generation — generates N rows reflecting the original distribution, (2) conditional generation — generates N rows belonging to a specified condition group (e.g., generating only non-performing loan customer records) to address class imbalance.
* **Business Logic Constraints Engine**: Enforces user-defined cross-column relational and conditional logic constraints on the synthetic output data, independent of training data quality and generative model behavior. Constraints are declared once and compiled into a unified execution graph supporting four enforcement mechanisms in order of priority: (1) pre-training mathematical transformations, (2) soft loss penalties during training (restricted to differentiable continuous constraints in CTGAN/CTVAE), (3) guided sampling during inference (applicable to discrete/categorical constraints and used in place of soft loss for Diffusion Models), and (4) post-generation correction with configurable retry limits and fallback policies. *To balance statistical fidelity, sampling throughput, and constraint satisfaction, the system supports a configurable violation rate threshold. Additionally, it measures distribution divergence (JS divergence) between raw generated data and post-corrected data to warn against the risk of "Constraint Over-Correction" distorting the original distribution.*
* **Inverse Transformation & Post-processing**: Converts synthetic data from normalized space back to the original value space in the following sequence: (1) **Inverse transformation** — applies the inverse of transformations stored in the transformation artifact (inverse MinMax scaling, categorical label decoding) to return all values, including scaled sentinel values, to the original value space; (2) **Missing state resolution** — inspects the binary auxiliary `is_missing` columns generated by the model, replacing values with true `NULL/NaN` where `is_missing = 1` and dropping all auxiliary columns from the output; (3) **Type-casting post-processing** — enforces data type casting per the schema artifact (rounding integer columns, boundary clamping, datetime reformatting). *Step 1→2 ordering is mandatory: sentinel values (e.g., -999) were transformed by MinMax scaling during preprocessing and must be inverse-scaled before is_missing flags can be interpreted.*

### Privacy Preservation & Compliance Module

Privacy is implemented as a cross-cutting concern — not an independent pipeline module but a coordinated set of mechanisms distributed across all three modules. The execution order of privacy mechanisms is:

1. **Module 1 — De-identification & Anonymization**: Removes direct identifiers before any component processes the data.
2. **Module 2 — Differential Privacy (DP-SGD)**: Injects noise into gradients to prevent the model from memorizing individual records. Privacy budget (ε, δ) is consumed and recorded here.
3. **Module 2 — Memorization Mitigation**: Additional regularization (weight decay, early stopping) reduces empirical memorization risk.
4. **Module 3 — Privacy Leakage Audit**: Empirically verifies that no records have been replicated (DCR, NNDR, MIA, AIA).
5. **Module 3 — Compliance Report**: Consolidates all privacy evidence into a versioned audit document.

* **Artifact Protection**: Schema artifacts and transformation artifacts contain boundary values (min/max) and specific category mappings of the real dataset — they are sensitive information in their own right. These artifacts must be stored in an isolated environment, encrypted at rest, and accessible only to the pipeline's service account.
* **Secure Localized Training Pipeline**: Restricts all stages of data ingestion, preprocessing, and model training to a secure, isolated local environment or private cloud (on-premise deployment). This architectural constraint guarantees that raw customer data never crosses the enterprise boundary, satisfying data residency and privacy compliance frameworks (such as GDPR and HIPAA).

| Requirement | Satisfying Mechanism | Module |
|---|---|---|
| No direct PII processing | De-identification (direct identifiers) + DP-SGD (quasi-identifiers) | 1, 2 |
| Data residency — data never leaves internal infrastructure | Localized Training Pipeline | 2 |
| Mathematical guarantee against re-identification | DP-SGD (ε, δ guarantee) | 2 |
| Empirical privacy verification | DCR / NNDR audit + MIA / AIA Attacker Simulation | 3 |
| Traceable audit documentation | Versioned Compliance Report | 3 |
| Sensitive artifact protection | Encryption at rest + access control | 1, 2, 3 |

### Evaluation, Validation & Audit Module

Quantifies the quality of the synthetic dataset across three independent dimensions — statistical fidelity, privacy safety, and machine learning utility — and consolidates results into a structured audit report.

* **Statistical Fidelity Assessment**: Computes statistical distance metrics between the distributions of the real and synthetic datasets across both marginal and joint distributions: Wasserstein distance for continuous columns, Jensen-Shannon divergence for categorical columns, and a cross-correlation difference matrix to assess the preservation of pairwise feature dependency structure.
* **Privacy Leakage Audit**: Measures the spatial isolation and attack resistance of synthetic data via two assessment types: (1) *Geometric distance evaluation* — Distance to Closest Record (DCR) and Nearest Neighbor Distance Ratio (NNDR) computed on the normalized feature space, with GPU-accelerated Approximate Nearest Neighbor search (Faiss or HNSW) for large datasets. NNDR close to 1 indicates good privacy (synthetic point is not uniquely close to any single real record); NNDR close to 0 indicates high memorization risk. (2) *Empirical Attacker Simulation* — integrates mock Membership Inference Attacks (MIA) and Attribute Inference Attacks (AIA) to evaluate empirical privacy risk, targeting a random guess baseline of ~50% for MIA.
* **ML Utility Evaluation — TSTR Framework**: Validates the utility of the synthetic dataset using the Train-on-Synthetic, Test-on-Real (TSTR) framework. The real dataset is strictly partitioned into Real_train and Real_test (holdout). Standard predictive models (XGBoost, Random Forest, Logistic Regression) are trained on the Synthetic dataset and evaluated on the held-out Real_test dataset. Results are compared against the Train-on-Real, Test-on-Real (TRTR) baseline to compute performance degradation across F1-score, AUC-ROC, and Accuracy metrics, ensuring zero data leakage.
* **Visual Distribution Overlays**: Automatically generates visual comparison charts overlaying real and synthetic distributions: per-column histograms, scatter plots for continuous feature pairs, and correlation heatmaps. Charts serve as rapid visual inspection alongside quantitative metrics.
* **Automated Compliance Reporting**: Consolidates all evaluation results into a structured audit report (PDF and HTML formats) comprising: statistical similarity metrics, privacy audit results (DCR, NNDR, applied ε value), and the machine learning performance degradation matrix. The report is versioned and linked to the schema artifact and transformation artifact of the corresponding run.

## Installation
Detail the exact steps to set up the environment. Specifying library versions is critical for ML reproducibility.

```bash
# Clone the repository
git clone https://github.com/TramMaiQuynh/synthetic-tabular-data.git
cd synthetic-tabular-data

# Create a virtual environment (Conda recommended)
conda create -n ai_env python=3.10
conda activate ai_env

# Install dependencies
pip install -r requirements.txt