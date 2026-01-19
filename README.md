# FairLend: Ethical Loan Default Prediction Pipeline

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Polars](https://img.shields.io/badge/ETL-Polars-orange)
![CatBoost](https://img.shields.io/badge/Model-CatBoost-green)
![Fairlearn](https://img.shields.io/badge/AI%20Ethics-Fairlearn-purple)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

## 📋 Executive Summary
**FairLend** is an end-to-end machine learning pipeline designed to predict loan defaults using the **LendingClub dataset (>2 Million rows)**. Unlike standard Kaggle projects, this solution prioritizes **AI Fairness** and **Resource-Constrained Engineering**. 

It implements a bias audit to detect and mitigate geographic/demographic discrimination in loan approvals, ensuring the model complies with ethical banking standards (e.g., Equal Credit Opportunity Act principles).



---

## 🚀 Key Technical Highlights
* **Scalable Data Engineering:** Leveraged **Polars Lazy execution and Streaming API** to process 2.2M rows (2GB+) on an 8GB RAM local machine, achieving 15x speedup over Pandas.
* **AI Ethics & Bias Mitigation:** Integrated **Fairlearn** to audit the model for disparate impact based on Zip Codes and Gender. Implemented a `ThresholdOptimizer` to enforce **Demographic Parity**.
* **Production-Ready Code:** Structured as a modular Python package (`src/`) rather than a monolithic notebook, featuring modular pipelines for Data Ingestion, Training, and Inference.
* **Advanced Modeling:** Utilized **CatBoost** for its native handling of high-cardinality categorical features (e.g., `emp_title`, `zip_code`), avoiding memory-intensive One-Hot Encoding.

---

## 🛠️ Tech Stack
* **ETL & Data Processing:** Polars, Numpy
* **Machine Learning:** CatBoost Classifier, Scikit-Learn
* **Fairness & Auditing:** Microsoft Fairlearn
* **Experiment Tracking:** MLflow (Logs metrics, params, and artifacts)
* **Visualization:** Matplotlib, Seaborn

---

## 📂 Project Structure
```text
├── data/               # Raw and Processed Data (DVC Versioned)
├── models/             # Serialized Models (.cbm) and Fairness Wrappers
├── notebooks/          # Exploratory Data Analysis (EDA)
├── reports/            # Generated Fairness Audits & Plots
├── src/                # Source Code
│   ├── data_loader.py  # Polars Streaming Ingestion
│   ├── preprocessing.py# Feature Engineering
│   ├── train.py        # CatBoost Training Pipeline with MLflow
│   └── fairness.py     # Bias Audit & Mitigation Logic
├── requirements.txt    # Dependencies
└── README.md           # Project Documentation
