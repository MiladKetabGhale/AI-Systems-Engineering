# Data Science, Machine Learning Engineering, and Data Engineering Projects

Welcome to my portfolio of projects, designed to demonstrate my skills in **machine learning engineering**, **data science**, and elements of **data engineering**. This repository contains standalone projects, each showcasing a specific set of capabilities, methodologies, and tools. The projects highlight the ability to build end-to-end data pipelines, engineer scalable solutions for machine learning tasks, and solve complex analytical problems.

## Repository Structure

Each project is maintained in a separate folder containing all relevant code, documentation, and results. Additionally, every project is hosted in an independent repository for better modularity and ease of access. You can find the standalone repository for each project by clicking on the repository name provided below.

### 1. [Housing Price Prediction](https://github.com/MiladKetabGhale/House_Price_Prediction/tree/main)
- **Focus**: End-to-end pipeline for predicting housing prices using regression models.
- **Key Features**:
  - Automated ingestion, transformation, extraction, model training, and evaluation for house price prediction.
  - Highly configurable custom ETL logic, model selection, and evaluation metrics via configuration files.
  - Robust parsing, and dynamic data handling, error handling including error logging
  - Scalable and extensible modular architecture for production-ready deployment

### 2. [EMNIST Classification Project](https://github.com/MiladKetabGhale/Image_Classification)
- **Focus**: Tackling classification challenges on the EMNIST dataset.
- **Key Features**:
    - Statistical analysis and visualization to uncover patterns and feature relationships
    - Training, evaluation, and tuning multiple classifiers to establish performance benchmarks
    - thorough error analysis and reliability analysis including tools such as SHAP, calibration curves, and Brier scores. Detailed timing analysis for model training
    - Producing top classifer using ensemble techniques against classification performance metrics
    - Feature selection and parallel training to handle costly computational complexity of some models
    - Tackling class imbalance with SMOTE, class weighting, and augmentation techniques

### 3. [Fraud Detection System Using Machine Learning, AWS, And Spark](https://github.com/MiladKetabGhale/Credit_Card_Fraud_Detection_System)
- **Focus**: Integrating AWS services with Spark using Boto3 to build scalable, automated pipelines for anomaly detection
- **Key Features**:
    - Data Preprocessing with PySpark: Scales, normalizes, and transforms the dataset before training
    - Multiple Sampling Techniques: Implements 9 sampling methods, including SMOTE variants and under-sampling techniques, to improve fraud detection
    - Cross-Validation & Hyperparameter Tuning: Fine-tunes the XGBoost model for optimal performance
    - AWS-Based Automated Deployment: Deploys the fraud detection system using AWS Glue, SageMaker, S3, and IAM through a fully automated Bash script
    - Automated Model Training & Inference: Uses SageMaker to train and evaluate the model, then deploys an inference script
    - Contextualizes Fraud Detection, the choice of metrics, and evaluations as part of Security Risk Management

### 4. Survival Analysis Using Healthcare Dataset *(upcoming in February)*
- **Focus**: Focuses on use of advanced statistical analysis frameworks and tools in combination with real and synthetic healthcare data for survival analysis.

## License
All projects in this portfolio are licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).  
You are free to use, modify, and distribute the code, provided proper attribution is given.  
The software is provided "as is," without warranties or guarantees of any kind.
