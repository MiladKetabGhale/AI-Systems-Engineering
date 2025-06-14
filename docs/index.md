
Welcome to my AI engineering portfolio, where I showcase modular, production-grade systems at the intersection of machine learning, AI, and cloud infrastructure. Each project is designed to highlight core strengths in scalable AI pipeline development, model optimization, and MLOps practices. From fine-tuned LLMs for cybersecurity to end-to-end fraud detection systems deployed on AWS

## Repository Structure

Each project is hosted in an independent repository for better modularity and ease of access. You can find the standalone repository for each project by clicking on the repository name provided below.

## 🛠️ Tech Stack

| Category | Tools & Libraries |
|----------|-------------------|
| **Languages** | Python • Bash • SQL |
| **AI / ML** | PyTorch • Hugging Face Transformers • Scikit-learn • Optuna • SHAP • DeepSeek R1 (LLaMA 7B, distilled) |
| **Cloud & MLOps** | AWS (S3 • SageMaker • Glue • IAM) • Docker • GitHub Actions • MLflow • CI/CD automation • Bash scripting|
| **Distributed & Big Data** | Apache Spark • PySpark • Pandas • NumPy |
| **Model Optimisation** | LoRA • Distillation • Quantisation • Dropout • Batch Normalisation |
| **Testing & Logging** | Pytest • Fixtures • Structured CLI logging • Custom error handling |

<sub>*Each project uses a subset of these tools as required.*</sub>

---

### Deep Learning, NLP, and LLMs

### 🔗 [LLM Cybersecurity Summarization System](https://github.com/MiladKetabGhale/LLM_Cybersecurity_Summarizer)

- **Focus**: Building a domain-adapted NLP system for summarizing cybersecurity knowledge using GPT-2 and Transformer-based architectures.
- **Key Features**:
    - **Custom Dataset Creation**: Curated a dataset by scraping 435 MITRE ATT&CK entries; generated reference summaries using a locally deployed Distilled DeepSeek R1 (LLaMA 7B); validated summary quality with statistical tests (t-test, KS-test, p > 0.05) and human evaluation.
    - **Ground-Up GPT-2 Implementation**: Built and pre-trained a custom GPT-2 model from scratch on WikiText-103 before fine-tuning on the cybersecurity dataset.
    - **Fine-Tuning & Benchmarking**: Fine-tuned both custom and pretrained GPT-2 models; benchmarked against BART and Pegasus zero-shot summarization, achieving competitive ROUGE scores with significantly smaller model sizes.
    - **LoRA Fine-Tuning Efficiency**: Demonstrated LoRA fine-tuning achieving ~99% of full fine-tuned performance (ROUGE-1) with <1% of parameters updated.
    - **Modular Architecture**: Independent modules for data preprocessing, model training, fine-tuning, and evaluation; each documented with detailed module-level documentation.
    - **Performance Engineering**: Optimized training runtime by ~3× on Apple M1 hardware via multithreaded data loading and GPU utilization improvements.
    - **Three-Tier Documentation**: Provides README-level overview, module-level `documentation.md`, and function-level docstrings for extensibility and developer onboarding.

**Goal**: Deliver a reproducible, efficient, and modular system bridging general-purpose language models and specialized cybersecurity summarization needs, validated against both statistical and human benchmarks.

---

### MLOps Infrastructure

#### 🔗 [MLOps Pipeline System for API-based ETL](https://github.com/MiladKetabGhale/FinTech-API-Based-ETL-Framework)

**Overview**  
This project involves building a modular, extensible DataOps pipeline for ingesting and processing data from multiple APIs. The architecture emphasizes clean separation of concerns, robust error handling, and comprehensive test coverage.

**Key Components**
- **Ingestion Layer**: Interfaces with external APIs to fetch data in a standardized format.
- **Validation Layer**: Performs schema validation and logical checks on incoming data before transformation.
- **Transformation Layer**: Normalizes and processes raw API responses into clean, analysis-ready datasets.
- **Dockerization**: Encapsulates services into Docker containers for consistent deployment and simplified environment management.

**Testing Architecture**
- Source-agnostic unit tests focusing on network behavior, retries, timeouts, and common ingestion issues. Each test resides in its own file for clarity and reuse.
- Source-specific tests validating data integrity, edge cases, and transformation logic tailored to individual APIs.
- **Fixtures**: Used to decouple test logic and ensure smooth plugin integration.

**Logging Strategy and Error Handling**
- All logs are centralized under `Error_Handling/logs/` for structured monitoring and troubleshooting.

**Goal**  
A production-grade, test-driven API ingestion and processing system, deployable in modern MLOps pipelines.

---

### Machine Learning Pipelines & Analysis

#### 🔗 [Fraud Detection System Using Machine Learning, AWS, And Spark](https://github.com/MiladKetabGhale/Credit_Card_Fraud_Detection_System)
- **Focus**: Integrating AWS services with Spark to build scalable, automated pipelines for anomaly detection
- **Key Features**:
    - Data Preprocessing with PySpark: Scales, normalizes, and transforms the dataset before training
    - Multiple Sampling Techniques: Implements 9 sampling methods, including SMOTE variants and under-sampling techniques, to improve fraud detection
    - Cross-Validation & Hyperparameter Tuning: Fine-tunes the XGBoost model for optimal performance
    - AWS-Based Automated Deployment: Deploys the fraud detection system using AWS Glue, SageMaker, S3, and IAM through a fully automated Bash script
    - Automated Model Training & Inference: Uses SageMaker to train and evaluate the model, then deploys an inference script
    - Automated logging, tracking, and model versioning using MLflow, together with hyperparamter tuning automation using Optuna
    - Contextualizes Fraud Detection, the choice of metrics, and evaluations as part of Security Risk Management

#### 🔗 [Image Classification Project](https://github.com/MiladKetabGhale/Image_Classification)
- **Focus**: Tackling classification challenges on the EMNIST dataset.
- **Key Features**:
    - Statistical analysis and visualization to uncover patterns and feature relationships
    - Training, evaluation, and tuning multiple classifiers to establish performance benchmarks
    - Thorough error analysis and reliability analysis including tools such as SHAP, calibration curves, and Brier scores.
    - Detailed timing analysis for model training
    - Producing top classifer using ensemble techniques against classification performance metrics
    - Feature selection and parallel training to handle costly computational complexity of some models
    - Tackling class imbalance with SMOTE, class weighting, and augmentation techniques

---

## License
All projects in this portfolio are licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).  
You are free to use, modify, and distribute the code, provided proper attribution is given.  
The software is provided "as is," without warranties or guarantees of any kind.


