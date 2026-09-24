"""
Zepto AI/ML Capstone
Module 1: Data Engineering Pipeline

Pipeline:
    1. Scrape books.toscrape.com
    2. Clean scraped data
    3. Convert GBP to INR using fixed rate 105.50
    4. Load normalized data into SQLite
    5. Run required SQL queries
    6. Read SQL results with pandas
    7. Reproduce JOIN using pandas.merge
"""

from pathlib import Path
import re
import sqlite3

import pandas as pd
import requests
from bs4 import BeautifulSoup


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

DB_PATH = OUTPUT_DIR / "books.db"
CLEAN_CSV_PATH = OUTPUT_DIR / "clean_books.csv"
SQL_OUTPUT_PATH = OUTPUT_DIR / "sql_results.txt"
JOIN_OUTPUT_PATH = OUTPUT_DIR / "join_comparison.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HTTP / SCRAPING HELPERS
# ============================================================

def get_soup(url):
    """
    Download a webpage and return BeautifulSoup.

    Explicitly checks the HTTP status code.
    """

    response = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent": (
                "Mozilla/5.0 "
                "(Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 "
                "Chrome/120 Safari/537.36"
            )
        },
    )

    response.raise_for_status()

    return BeautifulSoup(response.text, "html.parser")


def parse_price(value):
    """
    Convert a scraped price such as '£51.77' into 51.77.

    If parsing fails, return None so the cleaning stage
    can apply median imputation.
    """

    if pd.isna(value):
        return None

    text = str(value).strip()

    match = re.search(r"(\d+(?:\.\d+)?)", text)

    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None

    return None


def parse_rating(value):
    """
    Convert textual star rating into an integer from 1 to 5.
    """

    rating_map = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5,
    }

    if pd.isna(value):
        return None

    return rating_map.get(str(value).strip())


def parse_stock(value):
    """
    Convert availability text into boolean.

    Example:
        'In stock (22 available)' -> True
        'Out of stock' -> False
    """

    if pd.isna(value):
        return False

    return "in stock" in str(value).lower()


# ============================================================
# SCRAPING
# ============================================================

def scrape_books():
    """
    Scrape the first five pages of the All Products catalogue.

    Five pages x 20 books per page = approximately 100 books.
    """

    records = []

    for page_number in range(1, 6):

        if page_number == 1:
            page_url = BASE_URL
        else:
            page_url = (
                f"{BASE_URL}catalogue/"
                f"page-{page_number}.html"
            )

        print()
        print("=" * 70)
        print(f"Scraping catalogue page {page_number}")
        print(page_url)
        print("=" * 70)

        soup = get_soup(page_url)

        products = soup.select("article.product_pod")

        print(f"Books found on page: {len(products)}")

        for product in products:

            # ------------------------------
            # Title
            # ------------------------------

            title_tag = product.select_one("h3 a")

            title = ""

            if title_tag:
                title = title_tag.get("title", "").strip()

            # ------------------------------
            # Price
            # ------------------------------

            price_tag = product.select_one(".price_color")

            price = ""

            if price_tag:
                price = price_tag.get_text(strip=True)

            # ------------------------------
            # Availability
            # ------------------------------

            availability_tag = product.select_one(
                ".availability"
            )

            availability = ""

            if availability_tag:
                availability = availability_tag.get_text(
                    " ",
                    strip=True
                )

            # ------------------------------
            # Rating
            # ------------------------------

            rating = ""

            rating_tag = product.select_one(
                ".star-rating"
            )

            if rating_tag:

                classes = rating_tag.get(
                    "class",
                    []
                )

                valid_ratings = {
                    "One",
                    "Two",
                    "Three",
                    "Four",
                    "Five",
                }

                for class_name in classes:

                    if class_name in valid_ratings:
                        rating = class_name
                        break

            # ------------------------------
            # Category
            # ------------------------------

            category = "Unknown"

            if title_tag:

                detail_link = title_tag.get(
                    "href"
                )

                if detail_link:

                    detail_url = requests.compat.urljoin(
                        page_url,
                        detail_link
                    )

                    try:

                        detail_soup = get_soup(
                            detail_url
                        )

                        breadcrumbs = detail_soup.select(
                            "ul.breadcrumb li a"
                        )

                        # Home -> Books -> Category
                        if len(breadcrumbs) >= 3:

                            category = breadcrumbs[
                                -1
                            ].get_text(strip=True)

                    except requests.RequestException as exc:

                        print(
                            "Warning: could not retrieve "
                            f"category for '{title}': {exc}"
                        )

            records.append(
                {
                    "title": title,
                    "price": price,
                    "star_rating": rating,
                    "availability": availability,
                    "category": category,
                }
            )

    df = pd.DataFrame(records)

    print()
    print("=" * 70)
    print("SCRAPING SUMMARY")
    print("=" * 70)

    print(f"Rows scraped: {len(df)}")

    if not df.empty:
        print(
            "Categories found:",
            df["category"].nunique()
        )

    return df


# ============================================================
# CLEANING
# ============================================================

def clean_books(raw_df):
    """
    Clean raw scraped data.

    Numeric parsing failures are handled using median
    imputation as required by the assignment.
    """

    df = raw_df.copy()

    # ------------------------------
    # Parse numeric fields
    # ------------------------------

    df["price_gbp"] = df["price"].apply(
        parse_price
    )

    df["rating"] = df["star_rating"].apply(
        parse_rating
    )

    # ------------------------------
    # Parse availability
    # ------------------------------

    df["in_stock"] = df["availability"].apply(
        parse_stock
    )

    # ------------------------------
    # Numeric median imputation
    # ------------------------------

    for column in [
        "price_gbp",
        "rating",
    ]:

        missing_before = df[column].isna().sum()

        if missing_before > 0:

            median_value = df[column].median()

            if pd.isna(median_value):

                raise ValueError(
                    f"All values failed to parse "
                    f"for column '{column}'."
                )

            print(
                f"Imputing {missing_before} missing "
                f"values in {column} using median "
                f"{median_value}"
            )

            df[column] = df[column].fillna(
                median_value
            )

    # ------------------------------
    # Ensure numeric types
    # ------------------------------

    df["price_gbp"] = pd.to_numeric(
        df["price_gbp"],
        errors="coerce"
    )

    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )

    # Second safety check
    for column in [
        "price_gbp",
        "rating",
    ]:

        if df[column].isna().any():

            median_value = df[column].median()

            df[column] = df[column].fillna(
                median_value
            )

    # ------------------------------
    # Remove rows missing essential
    # identifying fields
    # ------------------------------

    before_drop = len(df)

    df = df.dropna(
        subset=[
            "title",
            "category",
        ]
    )

    dropped = before_drop - len(df)

    if dropped > 0:

        print(
            f"Dropped {dropped} rows because "
            "title/category was missing."
        )

    # ------------------------------
    # Required types
    # ------------------------------

    df["price_gbp"] = df["price_gbp"].astype(
        float
    )

    df["rating"] = df["rating"].astype(
        int
    )

    df["in_stock"] = df["in_stock"].astype(
        bool
    )

    # ------------------------------
    # Required fixed conversion
    # ------------------------------

    df["price_inr"] = (
        df["price_gbp"] * GBP_TO_INR
    ).round(2)

    # ------------------------------
    # Select final columns
    # ------------------------------

    cleaned = df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category",
        ]
    ].reset_index(drop=True)

    return cleaned


# ============================================================
# DATABASE
# ============================================================

def create_database(df):
    """
    Create normalized SQLite database.

    Tables:
        categories
        books

    Relationship:
        books.category_id -> categories.category_id
    """

    if DB_PATH.exists():
        DB_PATH.unlink()

    connection = sqlite3.connect(
        DB_PATH
    )

    try:

        cursor = connection.cursor()

        # Enable foreign-key enforcement
        cursor.execute(
            "PRAGMA foreign_keys = ON"
        )

        # ------------------------------
        # Categories table
        # ------------------------------

        cursor.execute(
            """
            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_name TEXT NOT NULL UNIQUE
            )
            """
        )

        # ------------------------------
        # Books table
        # ------------------------------

        cursor.execute(
            """
            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL,
                price_inr REAL NOT NULL,
                rating INTEGER NOT NULL
                    CHECK(rating BETWEEN 1 AND 5),
                in_stock INTEGER NOT NULL
                    CHECK(in_stock IN (0, 1)),
                category_id INTEGER NOT NULL,
                FOREIGN KEY(category_id)
                    REFERENCES categories(category_id)
            )
            """
        )

        # ------------------------------
        # Insert categories
        # ------------------------------

        categories = sorted(
            df["category"].unique()
        )

        cursor.executemany(
            """
            INSERT INTO categories
                (category_name)
            VALUES (?)
            """,
            [
                (category,)
                for category in categories
            ],
        )

        # ------------------------------
        # Build category lookup
        # ------------------------------

        category_rows = cursor.execute(
            """
            SELECT
                category_id,
                category_name
            FROM categories
            """
        ).fetchall()

        category_map = {
            category_name: category_id
            for category_id, category_name
            in category_rows
        }

        # ------------------------------
        # Prepare books
        # ------------------------------

        books_to_insert = []

        for _, row in df.iterrows():

            books_to_insert.append(
                (
                    row["title"],
                    float(row["price_gbp"]),
                    float(row["price_inr"]),
                    int(row["rating"]),
                    int(row["in_stock"]),
                    category_map[row["category"]],
                )
            )

        # ------------------------------
        # Insert books
        # ------------------------------

        cursor.executemany(
            """
            INSERT INTO books (
                title,
                price_gbp,
                price_inr,
                rating,
                in_stock,
                category_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            books_to_insert,
        )

        connection.commit()

        print()
        print("=" * 70)
        print("DATABASE CREATED")
        print("=" * 70)

        print(f"Database: {DB_PATH}")
        print(
            f"Books inserted: "
            f"{len(books_to_insert)}"
        )
        print(
            f"Categories inserted: "
            f"{len(categories)}"
        )

    finally:

        connection.close()


# ============================================================
# SQL QUERIES
# ============================================================

QUERIES = {

    "query_1_select_where": """
        SELECT
            title,
            price_gbp,
            rating
        FROM books
        WHERE rating >= 4
        ORDER BY rating DESC
        LIMIT 10;
    """,

    "query_2_order_by_limit": """
        SELECT
            title,
            price_gbp,
            price_inr
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10;
    """,

    "query_3_distinct": """
        SELECT DISTINCT
            rating
        FROM books
        ORDER BY rating;
    """,

    "query_4_between": """
        SELECT
            title,
            price_gbp,
            rating
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40
        ORDER BY price_gbp;
    """,

    "query_5_join": """
        SELECT
            c.category_name,
            b.title,
            b.rating,
            b.price_gbp,
            b.price_inr,
            b.in_stock
        FROM books AS b
        JOIN categories AS c
            ON b.category_id = c.category_id
        ORDER BY
            c.category_name,
            b.rating DESC,
            b.title
        LIMIT 20;
    """,
}


def execute_sql_queries():
    """
    Execute all required SQL queries and save
    both query text and output.
    """

    connection = sqlite3.connect(
        DB_PATH
    )

    output = []

    try:

        for query_name, query in QUERIES.items():

            print()
            print("=" * 70)
            print(query_name)
            print("=" * 70)

            print(query.strip())

            result = pd.read_sql_query(
                query,
                connection
            )

            print()
            print(
                result.to_string(
                    index=False
                )
            )

            output.append(
                "=" * 70
            )

            output.append(
                query_name
            )

            output.append(
                "=" * 70
            )

            output.append(
                query.strip()
            )

            output.append("")

            output.append(
                result.to_string(
                    index=False
                )
            )

            output.append("")

    finally:

        connection.close()

    SQL_OUTPUT_PATH.write_text(
        "\n".join(output),
        encoding="utf-8"
    )

    print()
    print(
        f"SQL output saved to: "
        f"{SQL_OUTPUT_PATH}"
    )


# ============================================================
# PANDAS SQL / MERGE COMPARISON
# ============================================================

def demonstrate_pandas_equivalence():
    """
    Reproduce the SQL JOIN using pandas.merge
    and compare the results.
    """

    connection = sqlite3.connect(
        DB_PATH
    )

    try:

        # ------------------------------
        # SQL JOIN using pd.read_sql
        # ------------------------------

        sql_join = """
            SELECT
                c.category_name,
                b.title,
                b.rating,
                b.price_gbp,
                b.price_inr,
                b.in_stock
            FROM books AS b
            JOIN categories AS c
                ON b.category_id = c.category_id
            ORDER BY
                c.category_name,
                b.rating DESC,
                b.title
            LIMIT 20;
        """

        sql_result = pd.read_sql_query(
            sql_join,
            connection
        )

        # ------------------------------
        # Load tables separately
        # ------------------------------

        books_df = pd.read_sql_query(
            """
            SELECT
                book_id,
                title,
                price_gbp,
                price_inr,
                rating,
                in_stock,
                category_id
            FROM books
            """,
            connection,
        )

        categories_df = pd.read_sql_query(
            """
            SELECT
                category_id,
                category_name
            FROM categories
            """,
            connection,
        )

    finally:

        connection.close()

    # ------------------------------
    # Reproduce JOIN with pandas
    # ------------------------------

    merged = books_df.merge(
        categories_df,
        on="category_id",
        how="inner",
    )

    merged = merged[
        [
            "category_name",
            "title",
            "rating",
            "price_gbp",
            "price_inr",
            "in_stock",
        ]
    ]

    merged = merged.sort_values(
        [
            "category_name",
            "rating",
            "title",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    )

    merged = merged.head(
        20
    ).reset_index(drop=True)

    sql_result = sql_result.reset_index(
        drop=True
    )

    # SQLite stores booleans as 0/1.
    # Convert both to the same Python bool type.
    sql_result["in_stock"] = (
        sql_result["in_stock"].astype(bool)
    )

    merged["in_stock"] = (
        merged["in_stock"].astype(bool)
    )

    equivalent = sql_result.equals(
        merged
    )

    print()
    print("=" * 70)
    print("pd.read_sql JOIN RESULT")
    print("=" * 70)

    print(
        sql_result.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("pd.merge JOIN RESULT")
    print("=" * 70)

    print(
        merged.to_string(
            index=False
        )
    )

    print()
    print(
        "Equivalent outputs:",
        equivalent
    )

    comparison_text = (
        "pd.read_sql JOIN RESULT\n"
        + "=" * 70
        + "\n"
        + sql_result.to_string(
            index=False
        )
        + "\n\n"
        + "pd.merge JOIN RESULT\n"
        + "=" * 70
        + "\n"
        + merged.to_string(
            index=False
        )
        + "\n\n"
        + f"Equivalent outputs: "
        f"{equivalent}\n"
    )

    JOIN_OUTPUT_PATH.write_text(
        comparison_text,
        encoding="utf-8"
    )

    return equivalent


# ============================================================
# VALIDATION
# ============================================================

def validate_clean_data(df):
    """
    Validate the cleaned dataset against the
    assignment requirements.
    """

    print()
    print("=" * 70)
    print("VALIDATING CLEAN DATA")
    print("=" * 70)

    if len(df) < 60:
        raise ValueError(
            f"Only {len(df)} rows found. "
            "At least 60 are required."
        )

    if df["category"].nunique() < 3:
        raise ValueError(
            "Fewer than 3 categories found."
        )

    required_columns = {
        "title",
        "price_gbp",
        "price_inr",
        "rating",
        "in_stock",
        "category",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: "
            f"{missing_columns}"
        )

    if df["price_gbp"].isna().any():
        raise ValueError(
            "price_gbp still contains missing values."
        )

    if df["price_inr"].isna().any():
        raise ValueError(
            "price_inr still contains missing values."
        )

    if df["rating"].isna().any():
        raise ValueError(
            "rating still contains missing values."
        )

    if not df["rating"].between(
        1,
        5
    ).all():
        raise ValueError(
            "rating contains values outside 1-5."
        )

    expected_inr = (
        df["price_gbp"] * GBP_TO_INR
    ).round(2)

    if not (
        df["price_inr"].round(2)
        == expected_inr
    ).all():

        raise ValueError(
            "price_inr does not match "
            "the required 105.50 conversion."
        )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Categories: "
        f"{df['category'].nunique()}"
    )

    print(
        "price_gbp missing:",
        df["price_gbp"].isna().sum()
    )

    print(
        "price_inr missing:",
        df["price_inr"].isna().sum()
    )

    print(
        "rating missing:",
        df["rating"].isna().sum()
    )

    print(
        "Conversion check: PASS"
    )

    print(
        "Validation: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ZEPTO AI/ML CAPSTONE")
    print("MODULE 1 - DATA PIPELINE")
    print("=" * 70)

    # ------------------------------
    # Step 1: Scrape
    # ------------------------------

    raw_df = scrape_books()

    if len(raw_df) < 60:

        raise RuntimeError(
            "Scraping produced fewer than "
            "60 books."
        )

    # ------------------------------
    # Step 2: Clean
    # ------------------------------

    cleaned_df = clean_books(
        raw_df
    )

    # ------------------------------
    # Step 3: Validate
    # ------------------------------

    validate_clean_data(
        cleaned_df
    )

    # ------------------------------
    # Step 4: Save CSV
    # ------------------------------

    cleaned_df.to_csv(
        CLEAN_CSV_PATH,
        index=False
    )

    print()
    print(
        f"Clean CSV saved to: "
        f"{CLEAN_CSV_PATH}"
    )

    print()
    print("Data types:")
    print(
        cleaned_df.dtypes
    )

    # ------------------------------
    # Step 5: Database
    # ------------------------------

    create_database(
        cleaned_df
    )

    # ------------------------------
    # Step 6: SQL
    # ------------------------------

    execute_sql_queries()

    # ------------------------------
    # Step 7: pandas JOIN
    # ------------------------------

    equivalent = (
        demonstrate_pandas_equivalence()
    )

    if not equivalent:

        raise RuntimeError(
            "pd.read_sql and pd.merge "
            "outputs are not equivalent."
        )

    # ------------------------------
    # Final success
    # ------------------------------

    print()
    print("=" * 70)
    print("DATA PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"Database: {DB_PATH}"
    )

    print(
        f"Clean CSV: {CLEAN_CSV_PATH}"
    )

    print(
        f"SQL results: {SQL_OUTPUT_PATH}"
    )

    print(
        f"JOIN comparison: "
        f"{JOIN_OUTPUT_PATH}"
    )

    print(
        "JOIN equivalence: PASS"
    )


if __name__ == "__main__":
    main()

