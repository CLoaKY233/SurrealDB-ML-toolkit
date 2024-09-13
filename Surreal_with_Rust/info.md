# Amazon Product Scraper with SurrealDB Integration (Rust)

## Overview
This Rust program scrapes product information from Amazon search results and stores the data in a SurrealDB database. It demonstrates web scraping, concurrent programming, and database integration.

## Key Components

### 1. Data Structures
```rust
struct Product {
    title: String,
    price: f64,
    rating: f32,
    review_count: u32,
}

struct Selectors {
    prod: Selector,
    title: Selector,
    price: Selector,
    rating: Selector,
    review_count: Selector,
}
```
- `Product`: Represents scraped product data.
- `Selectors`: Holds CSS selectors for HTML parsing.

### 2. Web Scraping Function
```rust
async fn fetch_page(
    thread_num: &u32,
    url: &str,
    client: Client,
    sel: Arc<Selectors>,
) -> Vec<Product> {
    // ... (implementation details)
}
```
This function fetches and parses a single Amazon search results page.

### 3. Database Upload Function
```rust
async fn upload_surreal(products: &Vec<Product>) -> surrealdb::Result<()> {
    // ... (implementation details)
}
```
Uploads scraped product data to SurrealDB.

### 4. Main Function
```rust
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ... (implementation details)
}
```
Orchestrates the entire scraping and uploading process.

## Core Concepts

1. **Asynchronous Programming**:
   - Uses `async/await` syntax for non-blocking operations.
   - Tokio runtime (`#[tokio::main]`) manages async tasks.

2. **Concurrent Scraping**:
   - Creates multiple tasks to scrape pages simultaneously.
   - Uses `join_all` to wait for all tasks to complete.

3. **Error Handling**:
   - Uses `Result` types to handle potential errors.
   - Provides fallback values for missing data.

4. **Web Scraping Basics**:
   - Sends HTTP GET requests to Amazon.
   - Parses HTML responses using CSS selectors.

5. **Database Integration**:
   - Connects to SurrealDB asynchronously.
   - Inserts scraped data as records.

## Code Walkthrough

1. **Setup**:
   - Initializes HTTP client and user input.
   - Prepares CSS selectors for scraping.

2. **Scraping Process**:
   - Creates a task for each page to scrape.
   - Each task runs `fetch_page` concurrently.

3. **Data Collection**:
   - Collects results from all scraping tasks.
   - Combines data into a single vector.

4. **Database Upload**:
   - Connects to SurrealDB.
   - Inserts each product as a new record.

5. **Output and Timing**:
   - Optionally prints scraped data.
   - Measures and reports total execution time.

## Learning Prompts

1. How would you modify the `Product` struct to include more details like product description?

2. Explain the purpose of `Arc` (Atomic Reference Counting) in `fetch_page`.

3. How does the code handle cases where product information is missing from the HTML?

4. What is the role of the `#[tokio::main]` attribute? How does it affect program execution?

5. How would you adapt the scraping logic if Amazon changes its page structure?

6. Describe the error handling in `fetch_page`. How could it be improved?

7. What are the advantages and potential risks of concurrent web scraping?

8. How would you modify `upload_surreal` to update existing records instead of creating new ones?

9. Explain the purpose of the `Selectors` struct and how it's used in scraping.

10. Implement a simple retry mechanism for failed HTTP requests in `fetch_page`.
