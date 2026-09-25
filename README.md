# Zepto AI/ML Capstone

An end-to-end AI/ML engineering capstone combining data engineering, SQL analytics, business analytics, Retrieval-Augmented Generation (RAG), API development, Docker, exploratory data analysis, machine learning, and model interpretation.

## Project Overview

This project contains four complementary modules:

1. Data Pipeline + SQL — collects and cleans book-catalog data, stores it in SQLite, and validates SQL queries.
2. Business Analytics — analyzes the SQLite catalog using SQL, Pandas, and Matplotlib.
3. Policy RAG Assistant — answers customer-policy questions using document retrieval, embeddings, ChromaDB, LangGraph, FastAPI, and Docker.
4. Predictive Analytics + ML API — performs Titanic dataset EDA, trains classification models, interprets model coefficients, and exposes predictions through FastAPI.

## Module 1 — Data Pipeline + SQL

- 100 books collected
- 29 categories identified
- SQLite database created
- SQL queries validated
- JOIN equivalence check passed

Database: data_pipeline/output/books.db

## Module 2 — Business Analytics

Analytics include category book counts, average price, average rating, stock rate, top 10 expensive books, rating distribution, and overall catalog metrics.

Current dataset metrics:
- Total books: 100
- Categories: 29
- Average price: INR 3,646.15
- Average rating: 2.93
- Stock rate: 100%

Run: python analytics_module2/analysis.py

Expected result: MODULE 2 ANALYTICS: PASS

Outputs are generated under analytics_module2/outputs/ and analytics_module2/plots/.

## Module 3 — Policy RAG Assistant

Eight policy documents cover delivery, returns, subscription plans, rider tracking, cancellation, damaged or missing items, gift cards, and customer support.

RAG workflow:
User question → embedding → ChromaDB retrieval → relevant policy documents → answer generation → source attribution

Validation: 9/9 local regression questions passed.

Run locally:
python -m uvicorn support_assistant.main:app --host 127.0.0.1 --port 7860

Endpoints: GET /health and POST /ask

Docker build:
docker build -f support_assistant/Dockerfile -t zepto-policy-assistant .

Docker run:
docker run --rm -p 7860:7860 zepto-policy-assistant

## Module 4 — Predictive Analytics + ML API

The module performs Titanic dataset EDA and evaluates Logistic Regression, Random Forest, and Gradient Boosting models using accuracy, precision, recall, F1, and ROC-AUC.

The cleaned dataset contains 889 observations with no missing values. Target leakage was avoided by excluding fields such as alive from the model features.

Model interpretation uses Logistic Regression feature coefficients.

Saved model: analytics/models/titanic_model.joblib

Run API:
python -m uvicorn analytics.api:app --host 127.0.0.1 --port 8000

Endpoints: GET /health and POST /predict

## Technologies

Python, SQLite, SQL, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib, Sentence Transformers, ChromaDB, LangGraph, LangChain Core, FastAPI, Uvicorn, Docker.

## Installation

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r support_assistant/requirements.txt

## Validation Summary

Module 1: data extraction, database creation, SQL validation, and JOIN equivalence passed.

Module 2: analytics pipeline passed.

Module 3: RAG regression 9/9 passed; FastAPI and Docker tested.

Module 4: EDA, model training, model comparison, interpretation, model reload, and prediction API tested.

## Git Hygiene

Generated databases, CSV outputs, plots, trained models, Python cache files, and ChromaDB data are excluded from version control.

## Reproducibility

Module 2 depends on the SQLite database produced by Module 1. Modules 3 and 4 can be run independently.

## Author

Zepto AI/ML Capstone — end-to-end data engineering, analytics, RAG, machine learning, API, and deployment project.
