import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import os
folder = os.path.dirname(os.path.abspath(__file__))

db_path = os.path.join(folder, "books.db")

url = "http://books.toscrape.com/"

response = requests.get(url)

# print(response.status_code)

#print(response.text)

soup = BeautifulSoup(response.text, "html.parser")


# print(soup.prettify()[:30000])

category_section = soup.find("div", class_ ="side_categories")

# print(category_section)

links = category_section.find_all("a")

book_cats = []

for link in links[1:]:
    category_name = link.text.strip()
    category_url = link["href"]
    book_cats.append({"category_name" : category_name, "category_url" :category_url})

# print(book_cats)
all_books = []
for book_cat in book_cats[:3]:
    key = book_cat["category_name"]
    value = book_cat["category_url"]
    next_page = url + value
    while next_page:

        response = requests.get(next_page)

        books_data = BeautifulSoup(response.text,"html.parser")

        # print(books_data.prettify)

        books = books_data.find_all("article", class_="product_pod")


        for book in books:
        # print(books[0].prettify)

            try:
                title = book.find("h3").find("a")["title"]
            except Exception:
                title = None
            try:
                price_gbp = float(book.find("p", class_="price_color").text.strip().strip("Â£"))
            except Exception:
                price_gbp = None
            try:
                rating = book.find("p", class_="star-rating")
                rating = rating["class"][1]
                rating_map = {
                                "One": 1,
                                "Two": 2,
                                "Three": 3,
                                "Four": 4,
                                "Five": 5
                            }

                rating = rating_map[rating]
            except Exception:
                rating = None

            

            try:
                in_stock = book.find("p", class_="instock availability").text.strip()

                

            except Exception:

                in_stock = None

            book_data = {
                
                "title" : title,
                "price_gbp" : price_gbp,
                "rating" : rating,
                "in_stock" : in_stock,
                "category_name" : key
            }

            all_books.append(book_data)
            # print(in_stock)

        next_button = books_data.find("li", class_="next")

        if next_button:

            next_url = next_button.find("a")["href"]

            print(next_url)

            current_url = next_page.rsplit("/", 1)[0]

            print(current_url)

            next_page = current_url + "/" + next_url

            print(next_page)

        else:
            next_page = None

        

# Convert list of dictionaries into dataframe
df = pd.DataFrame(all_books)

# Replace missing numeric values with median
df["price_gbp"] = df["price_gbp"].fillna(df["price_gbp"].median())
df["rating"] = df["rating"].fillna(df["rating"].median())
df["in_stock"] = df["in_stock"].astype(bool)
#df["in_stock"] = df["in_stock"].fillna(df["in_stock"].median())
        


# print(all_books)
print(len(all_books))

conversion_rate = 105.50

df["price_inr"] = df["price_gbp"] * conversion_rate

print(df.head())

categories_df = df[["category_name"]].drop_duplicates()

categories_df["category_id"] = range(1, len(categories_df) + 1)

print(categories_df.head())

df = df.merge(categories_df, on="category_name", how="left")

print(df.dtypes)

connection = sqlite3.connect(db_path)

cursor = connection.cursor()




cursor.execute("""
    CREATE TABLE IF NOT EXISTS categories(
    category_id INTEGER PRIMARY KEY,
    category_name TEXT UNIQUE)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS books(
    book_id INTEGER PRIMARY KEY,
    title TEXT,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock INTEGER,
    category_id INTEGER,
    FOREIGN KEY(category_id) REFERENCES categories(category_id)
)
""")

cursor.execute("DELETE FROM books")
cursor.execute("DELETE FROM categories")

for _,row in categories_df.iterrows():
    cursor.execute(

        """
        INSERT INTO categories(category_id, category_name)
        VALUES (?,?)
        """,
        (row["category_id"], row["category_name"])

    )

for _, row in df.iterrows():
     cursor.execute(
        """
        INSERT INTO books(
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            row["title"],
            row["price_gbp"],
            row["price_inr"],
            row["rating"],
            row["in_stock"],
            row["category_id"]
        )
    )


connection.commit()


query1 = """ SELECT title , price_gbp

FROM books
 
WHERE in_stock = 1; """


query2 = """ SELECT title , rating

FROM books
 
ORDER BY rating DESC; """


query3 = """ SELECT title , rating

FROM books
 
ORDER BY rating DESC

LIMIT 10; """

query4 = """ SELECT DISTINCT category_name
FROM categories; """

query5 = """ SELECT title , price_gbp

FROM books
 
WHERE price_gbp BETWEEN 40 AND 50; """


query6 = """ SELECT books.title , books.price_inr , books.rating , categories.category_name

FROM books

JOIN categories ON books.category_id = categories.category_id; """


queries = {
    "query1_select_where": query1,
    "query2_order_by": query2,
    "query3_limit": query3,
    "query4_distinct": query4,
    "query5_between": query5,
    "query6_join": query6
}


results = {}

for name, query in queries.items():
    output = pd.read_sql_query(

        query,connection
    )

    results[name] = output

for name, output in results.items():
    output_path = os.path.join(folder, f"{name}.csv")
    output.to_csv(output_path,index=False)


sql_Q1 = pd.read_sql(query1, connection)
sql_Q6 = pd.read_sql(query6, connection)

pandas_join = df.merge(categories_df,on="category_name",how="inner")

pandas_join = pandas_join[

    [
        "title",
        "price_inr",
        "rating",
        "category_name"

    ]
]


pandas_join = pandas_join.sort_values("title",ignore_index=False)

sql_Q6 = sql_Q6.sort_values("title",ignore_index=False)

print(pandas_join)

print(sql_Q6)

if (pandas_join.equals(sql_Q6)):
    print("Both are equal")