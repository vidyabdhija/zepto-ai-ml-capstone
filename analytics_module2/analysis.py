import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(
    os.path.dirname(BASE_DIR),
    "data_pipeline",
    "output",
    "books.db",
)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


def run_query(connection, query):
    return pd.read_sql_query(query, connection)


connection = sqlite3.connect(DB_PATH)

category_summary = run_query(
    connection,
    """
    SELECT
        c.category_name,
        COUNT(b.book_id) AS book_count,
        ROUND(AVG(b.price_inr), 2) AS avg_price_inr,
        ROUND(AVG(b.rating), 2) AS avg_rating,
        ROUND(AVG(b.in_stock) * 100, 2) AS stock_rate_pct
    FROM categories c
    LEFT JOIN books b
        ON c.category_id = b.category_id
    GROUP BY c.category_id, c.category_name
    ORDER BY book_count DESC;
    """,
)

category_summary.to_csv(
    os.path.join(OUTPUT_DIR, "category_summary.csv"),
    index=False,
)

top_expensive = run_query(
    connection,
    """
    SELECT
        b.title,
        c.category_name,
        b.price_gbp,
        b.price_inr,
        b.rating,
        b.in_stock
    FROM books b
    JOIN categories c
        ON b.category_id = c.category_id
    ORDER BY b.price_inr DESC
    LIMIT 10;
    """,
)

top_expensive.to_csv(
    os.path.join(OUTPUT_DIR, "top_expensive_books.csv"),
    index=False,
)

rating_summary = run_query(
    connection,
    """
    SELECT
        rating,
        COUNT(*) AS book_count
    FROM books
    GROUP BY rating
    ORDER BY rating;
    """,
)

rating_summary.to_csv(
    os.path.join(OUTPUT_DIR, "rating_distribution.csv"),
    index=False,
)

overall = run_query(
    connection,
    """
    SELECT
        COUNT(*) AS total_books,
        ROUND(AVG(price_inr), 2) AS avg_price_inr,
        ROUND(AVG(rating), 2) AS avg_rating,
        ROUND(AVG(in_stock) * 100, 2) AS stock_rate_pct,
        ROUND(SUM(price_inr), 2) AS total_catalog_value_inr
    FROM books;
    """,
)

highest_rated_category = run_query(
    connection,
    """
    SELECT
        c.category_name,
        ROUND(AVG(b.rating), 2) AS avg_rating
    FROM books b
    JOIN categories c
        ON b.category_id = c.category_id
    GROUP BY c.category_id, c.category_name
    ORDER BY avg_rating DESC, c.category_name
    LIMIT 1;
    """,
)

highest_price_category = run_query(
    connection,
    """
    SELECT
        c.category_name,
        ROUND(AVG(b.price_inr), 2) AS avg_price_inr
    FROM books b
    JOIN categories c
        ON b.category_id = c.category_id
    GROUP BY c.category_id, c.category_name
    ORDER BY avg_price_inr DESC, c.category_name
    LIMIT 1;
    """,
)

report_path = os.path.join(
    OUTPUT_DIR,
    "analytics_report.txt",
)

with open(report_path, "w", encoding="utf-8") as report:
    report.write("BOOK CATALOG ANALYTICS REPORT\n")
    report.write("=" * 60 + "\n\n")

    report.write("OVERALL METRICS\n")
    report.write("-" * 60 + "\n")
    report.write(overall.to_string(index=False))
    report.write("\n\n")

    report.write("HIGHEST-RATED CATEGORY\n")
    report.write("-" * 60 + "\n")
    report.write(highest_rated_category.to_string(index=False))
    report.write("\n\n")

    report.write("HIGHEST-PRICE CATEGORY\n")
    report.write("-" * 60 + "\n")
    report.write(highest_price_category.to_string(index=False))
    report.write("\n\n")

    report.write("TOP 10 MOST EXPENSIVE BOOKS\n")
    report.write("-" * 60 + "\n")
    report.write(top_expensive.to_string(index=False))
    report.write("\n")


category_plot = category_summary.sort_values(
    "book_count",
    ascending=True,
)

plt.figure(figsize=(10, 8))
plt.barh(
    category_plot["category_name"],
    category_plot["book_count"],
)
plt.xlabel("Number of books")
plt.ylabel("Category")
plt.title("Book Count by Category")
plt.tight_layout()
plt.savefig(
    os.path.join(PLOT_DIR, "category_book_count.png"),
    dpi=150,
)
plt.close()


price_plot = category_summary.sort_values(
    "avg_price_inr",
    ascending=True,
)

plt.figure(figsize=(10, 8))
plt.barh(
    price_plot["category_name"],
    price_plot["avg_price_inr"],
)
plt.xlabel("Average price (INR)")
plt.ylabel("Category")
plt.title("Average Book Price by Category")
plt.tight_layout()
plt.savefig(
    os.path.join(PLOT_DIR, "category_avg_price.png"),
    dpi=150,
)
plt.close()


rating_plot = category_summary.sort_values(
    "avg_rating",
    ascending=True,
)

plt.figure(figsize=(10, 8))
plt.barh(
    rating_plot["category_name"],
    rating_plot["avg_rating"],
)
plt.xlabel("Average rating")
plt.ylabel("Category")
plt.title("Average Rating by Category")
plt.xlim(0, 5)
plt.tight_layout()
plt.savefig(
    os.path.join(PLOT_DIR, "category_avg_rating.png"),
    dpi=150,
)
plt.close()


connection.close()

print("MODULE 2 ANALYTICS: PASS")
print(f"Database: {DB_PATH}")
print(f"Category summary: {OUTPUT_DIR}/category_summary.csv")
print(f"Top expensive books: {OUTPUT_DIR}/top_expensive_books.csv")
print(f"Rating distribution: {OUTPUT_DIR}/rating_distribution.csv")
print(f"Report: {OUTPUT_DIR}/analytics_report.txt")
print(f"Plots: {PLOT_DIR}")
print()

print("OVERALL METRICS")
print(overall.to_string(index=False))
print()

print("HIGHEST-RATED CATEGORY")
print(highest_rated_category.to_string(index=False))
print()

print("HIGHEST-PRICE CATEGORY")
print(highest_price_category.to_string(index=False))
