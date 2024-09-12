from surrealdb import Surreal  # type: ignore
import csv
import asyncio

async def main():
    """
    Main function to demonstrate database operations with SurrealDB.

    This function connects to a SurrealDB instance, signs in, selects a namespace and database,
    then reads data from a CSV file and inserts it into the database.
    """
    # Connect to SurrealDB
    async with Surreal("ws://localhost:8000/rpc") as db:
        # Sign in to the database
        await db.signin({"user": "root", "pass": "root"})

        # Select the namespace and database
        await db.use("test", "test")

        # Open and read the CSV file
        with open("datasets/fossil.csv", encoding="utf-8") as file:
            csv_reader = csv.DictReader(file)

            # Iterate through each row in the CSV
            for row in csv_reader:
                # Create a new record in the 'watches' table
                await db.create("watches", {
                    "title": row["title"],
                    "price": row["price"] if row["price"] else None,
                    "rating": row["rating"] if row["rating"] else None,
                    "review_count": int(row["review_count"]) if row["review_count"] else None
                })

            print("Data insertion complete")

# Run the main function using asyncio
asyncio.run(main())
