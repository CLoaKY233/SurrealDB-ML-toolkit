# CSV to SurrealDB Data Import

## Overview

This Python script demonstrates how to import data from a CSV file into a SurrealDB database. It uses the SurrealDB Python client and the built-in CSV module.

## Key Components

### SurrealDB Connection
```python
async with Surreal("ws://localhost:8000/rpc") as db:
    await db.signin({"user": "root", "pass": "root"})
    await db.use("test", "test")
```
This code establishes a connection to SurrealDB, signs in, and selects a namespace and database.

### CSV Reading
```python
with open("datasets/fossil.csv", encoding="utf-8") as file:
    csv_reader = csv.DictReader(file)
```
The script opens and reads a CSV file named "fossil.csv" using the CSV module's DictReader.

### Data Insertion
```python
for row in csv_reader:
    await db.create("watches", {
        "title": row["title"],
        "price": row["price"] if row["price"] else None,
        "rating": row["rating"] if row["rating"] else None,
        "review_count": int(row["review_count"]) if row["review_count"] else None
    })
```
This loop iterates through each row of the CSV and inserts it into the "watches" table in SurrealDB.

## Learning Prompts

1. How would you modify the script to handle different CSV file structures or column names?

2. Can you explain the purpose of the `encoding="utf-8"` parameter when opening the CSV file?

3. How would you implement error handling for cases where the CSV file is not found or is improperly formatted?

4. What are the advantages of using `csv.DictReader` over other methods of reading CSV files?

5. How would you modify the script to allow users to specify the CSV file path and table name as command-line arguments?

6. Can you explain the significance of the conditional statements (e.g., `if row["price"] else None`) in the data insertion step?

7. How would you adapt this script to handle very large CSV files that might not fit into memory?

8. What strategies could you employ to optimize the performance of bulk data insertion into SurrealDB?

9. How would you modify the script to update existing records in SurrealDB instead of creating new ones if a record with the same key already exists?

10. Can you explain how to use SurrealDB's query capabilities to verify the imported data and perform basic analytics on it?
