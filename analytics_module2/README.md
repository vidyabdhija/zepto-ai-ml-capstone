# Module 2 — Book Catalog Business Analytics

## Objective

Build a business analytics layer on top of the SQLite database created in Module 1.

The module uses SQL, Pandas, and Matplotlib to analyze the book catalog and produce reusable CSV reports, a text summary, and visualizations.

## Data Source

Module 1 SQLite database: data_pipeline/output/books.db

Tables: books, categories

## Analytics Performed

- Category summary: book count, average price, average rating, stock rate
- Top 10 most expensive books
- Rating distribution
- Overall catalog metrics
- Highest-rated category
- Highest-price category

## Outputs

- outputs/category_summary.csv
- outputs/top_expensive_books.csv
- outputs/rating_distribution.csv
- outputs/analytics_report.txt

## Visualizations

- plots/category_book_count.png
- plots/category_avg_price.png
- plots/category_avg_rating.png

## Run

python analytics_module2/analysis.py

Expected result: MODULE 2 ANALYTICS: PASS

## Validation

The analytics were independently checked against the Module 1 SQLite database. The current dataset contains 100 books across 29 categories, with ratings from 1 through 5.

## Technologies

Python, SQLite, SQL, Pandas, Matplotlib
