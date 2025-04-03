# Machine Learning Engineering And Data Science Projects

Welcome to my portfolio of projects, designed to demonstrate my skills in **machine learning engineering**, **data science**, and elements of **data engineering**. This repository contains standalone projects, each showcasing a specific set of capabilities, methodologies, and tools. The projects highlight the ability to build end-to-end data pipelines, engineer scalable solutions for machine learning tasks, and solve complex analytical problems.

## Repository Structure

Each project is maintained in a separate folder containing all relevant code, documentation, and results. Additionally, every project is hosted in an independent repository for better modularity and ease of access. You can find the standalone repository for each project by clicking on the repository name provided below.

### 1. [End-to-end Machine Learning Pipeline](https://github.com/MiladKetabGhale/House_Price_Prediction/tree/main)
- **Focus**: End-to-end pipeline for predicting housing prices using regression models.
- **Key Features**:
  - Automated ingestion, transformation, extraction, model training, and evaluation for house price prediction.
  - Highly configurable custom ETL logic, model selection, and evaluation metrics via configuration files.
  - Robust parsing, and dynamic data handling, error handling including error logging
  - Scalable and extensible modular architecture for production-ready deployment

### 2. [Image Classification Project](https://github.com/MiladKetabGhale/Image_Classification)
- **Focus**: Tackling classification challenges on the EMNIST dataset.
- **Key Features**:
    - Statistical analysis and visualization to uncover patterns and feature relationships
    - Training, evaluation, and tuning multiple classifiers to establish performance benchmarks
    - thorough error analysis and reliability analysis including tools such as SHAP, calibration curves, and Brier scores. Detailed timing analysis for model training
    - Producing top classifer using ensemble techniques against classification performance metrics
    - Feature selection and parallel training to handle costly computational complexity of some models
    - Tackling class imbalance with SMOTE, class weighting, and augmentation techniques

### 3. [Fraud Detection System Using Machine Learning, AWS, And Spark](https://github.com/MiladKetabGhale/Credit_Card_Fraud_Detection_System)
- **Focus**: Integrating AWS services with Spark to build scalable, automated pipelines for anomaly detection
- **Key Features**:
    - Data Preprocessing with PySpark: Scales, normalizes, and transforms the dataset before training
    - Multiple Sampling Techniques: Implements 9 sampling methods, including SMOTE variants and under-sampling techniques, to improve fraud detection
    - Cross-Validation & Hyperparameter Tuning: Fine-tunes the XGBoost model for optimal performance
    - AWS-Based Automated Deployment: Deploys the fraud detection system using AWS Glue, SageMaker, S3, and IAM through a fully automated Bash script
    - Automated Model Training & Inference: Uses SageMaker to train and evaluate the model, then deploys an inference script
    - Automated logging, tracking, and model versioning using MLflow, together with hyperparamter tuning automation using Optuna
    - Contextualizes Fraud Detection, the choice of metrics, and evaluations as part of Security Risk Management

---

## 4. DataOps Pipeline for API-based ETL *(Upcoming April)*

**Overview**  
This project involves building a modular, extensible DataOps pipeline for ingesting and processing data from multiple APIs. The architecture emphasizes clean separation of concerns, robust error handling, and comprehensive test coverage.

**Key Components**
- **Ingestion Layer**: Interfaces with external APIs to fetch data in a standardized format.
- **Validation Layer**: Performs schema validation and logical checks on incoming data before transformation.
- **Transformation Layer**: Normalizes and processes raw API responses into clean, analysis-ready datasets.
- **Dockerization**: Encapsulates services into Docker containers for consistent deployment and simplified environment management.
- **Airflow Integration**: Utilizes Airflow for orchestrating, scheduling, and monitoring complex ETL workflows.

**Testing Architecture**
- `tests/base/`: Source-agnostic unit tests focusing on network behavior, retries, timeouts, and common ingestion issues. Each test resides in its own file for clarity and reuse.
- `tests/plugins/`: Source-specific tests validating data integrity, edge cases, and transformation logic tailored to individual APIs.
- **Fixtures**: Used to decouple test logic and ensure smooth plugin integration.

**Logging Strategy and Error Handling**
- All logs are centralized under `Error_Handling/logs/` for structured monitoring and troubleshooting.

**Goal**  
A production-grade, test-driven API ingestion and processing system, deployable in modern MLOps pipelines.


## 5. NLP for Cybersecurity Summarization Using Transformers *(Upcoming Mid May)*

**Overview**  
This project focuses on extracting and summarizing cybersecurity intelligence using Transformer-based architectures. It involves both building a custom Transformer model and fine-tuning state-of-the-art pretrained LLMs.

#### Part 1: Custom Transformer Implementation
- **Objective**: Implement a Transformer architecture from scratch tailored for cybersecurity summarization.
- **Scope**:
  - Encoder-decoder architecture with scaled dot-product attention and multi-head attention.
  - Configurable number of layers and attention heads (starting with 2–3 heads for prototyping).
- **Tokenization and Embedding**: Handled externally to streamline model development.
- **Output**: A functional Transformer capable of generating concise summaries of cybersecurity text and used as baseline for benchmarking the BART LLM below.

#### Part 2: Fine-Tuning Pretrained LLMs (Hugging Face)
- **Objective**: Fine-tune pretrained models (BART LLM) on the same cybersecurity dataset.
- **Process**:
  - Dataset preprocessing and tokenization using Hugging Face Transformers.
  - Fine-tuning with a focus on performance, robustness, and generalization.
- **Evaluation**: Direct performance comparison with the custom Transformer using standard summarization metrics (e.g., ROUGE, BLEU).
- **Deployment Readiness**: Code structured for integration into a production pipeline or as a service endpoint.

**Goal**  
Deliver both a ground-up and transfer learning-based solution for cybersecurity summarization, showcasing deep understanding of NLP architecture and applied modeling.


## License
All projects in this portfolio are licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).  
You are free to use, modify, and distribute the code, provided proper attribution is given.  
The software is provided "as is," without warranties or guarantees of any kind.
