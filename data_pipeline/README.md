# Module 1 — Data Pipeline

## Overview

This module implements an end-to-end data engineering pipeline:

1. Scrape raw book data from `books.toscrape.com`.
2. Clean and type-convert the scraped fields.
3. Convert GBP prices to INR using the required fixed project rate.
4. Store the cleaned data in a normalized SQLite database.
5. Execute SQL queries demonstrating filtering, ordering, limiting, distinct values, range filtering, and table joins.
6. Read SQL results into pandas.
7. Reproduce the SQL JOIN using `pandas.merge()` and verify that both approaches produce equivalent results.

## Data Source

The source is:

`https://books.toscrape.com/`

The pipeline scrapes the first five catalogue pages, producing 100 book records.

The final dataset contains:

- 100 books
- 29 categories

This exceeds the assignment requirement of at least 60 books across at least 3 categories.

## Cleaning Decisions

### Price

The scraped GBP price is parsed into the numeric `price_gbp` column.

Unexpected numeric parsing failures are represented as missing values and handled using median imputation.

### Rating

The textual star rating is mapped as follows:

- One → 1
- Two → 2
- Three → 3
- Four → 4
- Five → 5

Unexpected numeric parsing failures are handled using median imputation.

### Availability

Availability text containing `In stock` is converted to `True`.

Other availability values are represented as `False`.

### Currency Conversion

The assignment-required fixed conversion rate is:

**1 GBP = 105.50 INR**

This is an artificial project-defined constant and is not obtained from an external API.

The `price_inr` value is calculated as:

```text
price_inr = price_gbp × 105.50

