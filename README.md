Zepto AI/ML Capstone

An end-to-end AI/ML engineering capstone covering data engineering, SQL analytics, business analytics, Retrieval-Augmented Generation (RAG), API development, Docker, exploratory data analysis, machine learning, and model interpretation.

Project Overview

This project contains four complementary modules:

Data Pipeline + SQL — collects and cleans book-catalog data, stores it in SQLite, and validates SQL queries.

Business Analytics — analyzes the SQLite catalog using SQL, Pandas, and Matplotlib.

Policy RAG Assistant — answers customer-policy questions using document retrieval, embeddings, ChromaDB, LangGraph, FastAPI, and Docker.

Predictive Analytics + ML API — performs Titanic EDA, trains classification and regression models, interprets model behavior, and exposes predictions through FastAPI.

Project Structure
zepto-ai-ml-capstone/
│
├── data_pipeline/
│   ├── pipeline.py
│   ├── README.md
│   └── output/
│
├── analytics_module2/
│   ├── analysis.py
│   ├── README.md
│   ├── outputs/
│   └── plots/
│
├── support_assistant/
│   ├── main.py
│   ├── models.py
│   ├── prompts.py
│   ├── rag.py
│   ├── ingest.py
│   ├── docs/
│   ├── chroma_db/
│   ├── Dockerfile
│   └── requirements.txt
│
├── analytics/
│   ├── 01_eda.py
│   ├── 02_model.py
│   ├── 03_interpretation.py
│   ├── api.py
│   ├── titanic.csv
│   ├── titanic_cleaned.csv
│   ├── models/
│   ├── outputs/
│   └── plots/
│
├── tests/
│   └── test_apis.py
│
├── requirements.txt
└── README.md

Module 1 — Data Pipeline + SQL

The data pipeline collects book-catalog data from Books to Scrape, cleans and validates the data, converts prices from GBP to INR, and stores the results in SQLite.

Results

100 books collected

29 categories identified

Missing price_gbp: 0

Missing price_inr: 0

Missing rating: 0

Price conversion validation: PASS

Data validation: PASS

100 books inserted into SQLite

29 categories inserted into SQLite

SQL queries validated

Pandas/SQL JOIN equivalence: PASS

Outputs

data_pipeline/output/clean_books.csv

data_pipeline/output/books.db

data_pipeline/output/sql_results.txt

data_pipeline/output/join_comparison.txt

Run
python data_pipeline/pipeline.py

Module 2 — Business Analytics

This module analyzes the SQLite book catalog using SQL, Pandas, and Matplotlib.

The analysis produces category-level summaries, price and rating distributions, and visualizations for business interpretation.

Outputs

analytics_module2/outputs/analytics_report.txt

analytics_module2/outputs/category_summary.csv

analytics_module2/outputs/rating_distribution.csv

analytics_module2/outputs/top_expensive_books.csv

Visualizations

Category average price

Category average rating

Category book count

Run
python analytics_module2/analysis.py

Module 3 — Policy RAG Support Assistant

The Policy Support Assistant uses Retrieval-Augmented Generation concepts to answer customer-policy questions from a collection of policy documents.

Components

Sentence Transformer embeddings

ChromaDB vector database

LangGraph workflow

FastAPI REST API

Policy document retrieval

Source-document references

Confidence score

Docker support

The current implementation exposes the retrieval-based assistant through FastAPI.

API

Start the service:

python -m support_assistant.main


The API runs on:

http://127.0.0.1:8000

Health Check
curl http://127.0.0.1:8000/health


Example response:

{
  "service": "Zepto Policy Support Assistant",
  "status": "healthy",
  "mock_llm": true
}

Ask a Policy Question
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the refund policy?"}'


Example response:

{
  "answer": "Perishable and grocery items can be returned within 24 hours if they are damaged, spoiled, or incorrect. Non-perishable items can generally be returned within 7 days if unopened. Eligible refunds are processed within 3–5 business days or may be issued instantly to the Zepto wallet.",
  "sources": [
    "doc_02",
    "doc_06"
  ],
  "confidence": 0.95
}

Module 4 — Predictive Analytics + ML API

This module performs exploratory data analysis and machine learning using the Titanic dataset.

Exploratory Data Analysis

The EDA workflow examines:

Age and fare distributions

Correlations between numerical variables

Survival by sex

Survival by passenger class

Survival by sex and class

Fare relationships

Standardization behavior

Run:

python analytics/01_eda.py

Classification Models

Three classification models were evaluated:

Logistic Regression

Decision Tree

Random Forest

The evaluation includes:

Accuracy

Precision

Recall

F1 score

ROC-AUC

Confusion matrix

Model Results
Model	Accuracy	Precision	Recall	F1	ROC-AUC
Logistic Regression	0.8427	0.8125	0.7647	0.7879	0.8668
Decision Tree	0.7921	0.8163	0.5882	0.6838	0.8315
Random Forest	0.8146	0.7612	0.7500	0.7556	0.8350

The results are reported for the evaluation split used by the project.

Random Forest Grid Search

The GridSearch process selected:

max_depth = 10
min_samples_split = 2
n_estimators = 200

Fare Regression

A regression model was also evaluated for fare prediction.

Results:

MAE:         20.6700
RMSE:        42.4424
R²:           0.3248
Adjusted R²:  0.3052

Model Interpretation

The project also generates:

Feature coefficients

Feature importance

Confusion matrices

Regression residual analysis

Random Forest tuning results

Out-of-bag evaluation results

Run
python analytics/02_model.py


Then:

python analytics/03_interpretation.py


The trained classification model is saved to:

analytics/models/titanic_model.joblib

Titanic Prediction API

The trained model is exposed through FastAPI.

Start the API:

uvicorn analytics.api:app --host 127.0.0.1 --port 8001


The API runs on:

http://127.0.0.1:8001

Health Check
curl http://127.0.0.1:8001/health

Prediction
curl -X POST http://127.0.0.1:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pclass": 1,
    "sex": "female",
    "age": 30,
    "sibsp": 0,
    "parch": 0,
    "fare": 100,
    "embarked": "C",
    "adult_male": false,
    "deck": "C",
    "alone": true
  }'


Example response:

{
  "prediction": 1,
  "prediction_label": "survived",
  "survival_probability": 0.989
}

Automated Tests

The project includes API tests covering:

Titanic API health endpoint

Titanic prediction endpoint

Request validation

Policy assistant health endpoint

Policy question endpoint

Run the complete test suite with:

python -m pytest -q


Current result:

5 passed


Using python -m pytest is recommended because it ensures pytest runs with the project's virtual environment.

Installation

Create and activate a virtual environment:

python3 -m venv .venv
source .venv/bin/activate


Install the main dependencies:

python -m pip install -r requirements.txt


For the Policy RAG Assistant:

python -m pip install -r support_assistant/requirements.txt

Environment Configuration

The project supports environment variables through .env.

Do not commit secrets, API keys, or authentication tokens to Git.

If Hugging Face authentication is required for higher download limits, configure the appropriate token in the local environment rather than committing it to the repository.

Validation Summary

The project has been validated through both manual API testing and automated tests.

Component	Status
Data pipeline	PASS
SQLite database	PASS
SQL analysis	PASS
Business analytics	PASS
RAG health endpoint	PASS
RAG question endpoint	PASS
Titanic model training	PASS
Titanic model persistence	PASS
Titanic prediction API	PASS
Automated API tests	PASS
git diff --check	PASS

Automated test result:

5 passed

Technologies

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

SQLite

FastAPI

Uvicorn

Pydantic

ChromaDB

Sentence Transformers

LangGraph

Joblib

Pytest

Docker

Git

Notes

This repository is structured as an end-to-end AI/ML engineering capstone, demonstrating data processing, analytics, machine learning, model interpretation, retrieval-based question answering, REST API development, testing, and deployment-oriented project organization.
