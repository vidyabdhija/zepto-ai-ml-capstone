# Zepto AI/ML Capstone

An end-to-end AI/ML engineering capstone combining data engineering, SQL analytics, business analytics, Retrieval-Augmented Generation (RAG), API development, Docker, exploratory data analysis, machine learning, and model interpretation.

## Project Overview

This project contains four complementary modules:

1. **Data Pipeline + SQL** — collects and cleans book-catalog data, stores it in SQLite, and validates SQL queries.
2. **Business Analytics** — analyzes the SQLite catalog using SQL, Pandas, and Matplotlib.
3. **Policy RAG Assistant** — answers customer-policy questions using document retrieval, embeddings, ChromaDB, LangGraph, FastAPI, and Docker.
4. **Predictive Analytics + ML API** — performs Titanic dataset EDA, trains classification and regression models, interprets the saved model, and exposes predictions through FastAPI.

---

## Module 1 — Data Pipeline + SQL

The data pipeline collects book-catalog data from Books to Scrape, cleans and validates the data, converts prices from GBP to INR, and stores the results in SQLite.

### Results

- 100 books collected
- 29 categories identified
- Missing `price_gbp`: 0
- Missing `price_inr`: 0
- Missing `rating`: 0
- Price conversion validation: PASS
- Data validation: PASS
- 100 books inserted into SQLite
- 29 categories inserted into SQLite
- SQL queries validated
- Pandas/SQL JOIN equivalence: PASS

### Outputs

- Clean dataset: `data_pipeline/output/clean_books.csv`
- SQLite database: `data_pipeline/output/books.db`
- SQL results: `data_pipeline/output/sql_results.txt`
- JOIN comparison: `data_pipeline/output/join_comparison.txt`

### Run

```bash
python data_pipeline/pipeline.py
