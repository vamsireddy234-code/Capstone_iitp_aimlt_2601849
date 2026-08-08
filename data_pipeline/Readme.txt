# Data Pipeline

## Overview

This project scrapes book data from **Books to Scrape**, cleans the data using pandas, stores it in SQLite, and runs SQL queries.

## Requirements


beautifulsoup4
pandas
requests
sqlite3
os

## Run

Run the script from the data_pipeline folder:

bash
python scrape_and_load.py


The script automatically:

1. Scrapes books from 3 categories.
2. Collects book title, price, rating, stock status, and category.
3. Calculates the INR price.
4. Creates the SQLite database.
5. Loads the data into the database.
6. Runs the SQL queries.
7. Saves the query outputs.

## Currency Conversion

The project uses the fixed conversion rate:

**1 GBP = 105.50 INR**

text
price_inr = price_gbp × 105.50


## Database

The SQLite database is stored as:

text
books.db


It contains two tables.

### categories

* category_id — Primary Key
* category_name — Unique

### books

* book_id — Primary Key
* title
* price_gbp
* price_inr
* rating
* in_stock
* category_id — Foreign Key

## Data Cleaning

* price_gbp is converted to a numeric value.
* Ratings are converted from words (One to Five) into integers 1–5.
* Stock status is converted to Boolean.
* price_inr is calculated using the fixed exchange rate of 105.50.
* Category IDs are used to connect books with their categories.

## SQL Queries

The project includes SQL queries using:

* WHERE
* ORDER BY
* LIMIT
* DISTINCT
* BETWEEN
* JOIN

The SQL results are saved in the data_pipeline folder.

## JOIN Check

The SQL JOIN result is compared with the equivalent pandas pd.merge() result.

The script checks that both outputs match.

## Git

The data pipeline was developed on a feature branch with at least two commits and then merged into main.
