use futures::future::join_all;
use reqwest::Client;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use std::io::{stdin, stdout, Write};
use std::sync::Arc;
use std::time::Instant;
use surrealdb::engine::remote::ws::Ws;
use surrealdb::opt::auth::Root;
use surrealdb::sql::Thing;
use surrealdb::Surreal;
use tokio;

// Define structures for database records and product information
#[derive(Debug, Deserialize)]
struct Record {
    #[allow(dead_code)]
    id: Thing,
}

#[derive(Debug, Serialize)]
struct Product {
    title: String,
    price: f64,
    rating: f32,
    review_count: u32,
}

// Define a structure to hold CSS selectors for web scraping
struct Selectors {
    prod: Selector,
    title: Selector,
    price: Selector,
    rating: Selector,
    review_count: Selector,
}

// Helper function to read string input from user
fn read_input_str(prompt: &str) -> String {
    print!("{}", prompt);
    stdout().flush().unwrap();
    let mut input = String::new();
    stdin().read_line(&mut input).unwrap();
    input.trim().to_string()
}

// Helper function to read boolean input from user
fn read_input_bool(prompt: &str) -> bool {
    loop {
        let input = read_input_str(prompt).to_lowercase();
        match input.as_str() {
            "y" => return true,
            "n" => return false,
            _ => println!("Please enter 'y' or 'n'"),
        }
    }
}

/*
 * Function to fetch and parse a single page of product data
 *
 * This function performs the following steps:
 * 1. Sends a GET request to the specified URL with custom headers
 * 2. Parses the HTML content of the response
 * 3. Extracts product information using CSS selectors
 * 4. Returns a vector of Product structs
 *
 * Error handling is implemented for network requests and HTML parsing.
 */
async fn fetch_page(
    thread_num: &u32,
    url: &str,
    client: Client,
    sel: Arc<Selectors>,
) -> Vec<Product> {
    let mut products = Vec::new();

    // Fetch the page content
    let response = match client
        .get(url)
        .header(reqwest::header::USER_AGENT, "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        .header(reqwest::header::ACCEPT, "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8")
        .header(reqwest::header::ACCEPT_LANGUAGE, "en-US,en;q=0.5")
        .send()
        .await {
            Ok(response) => response,
            Err(e) => {
                eprintln!("Error fetching page: {}", e);
                return products;
            }
        };

    // Parse the HTML content
    let body = match response.text().await {
        Ok(body) => body,
        Err(e) => {
            eprintln!("Error reading response body: {}", e);
            return products;
        }
    };

    let document = Html::parse_document(&body);

    // Extract product information using CSS selectors
    for product in document.select(&sel.prod) {
        // Extract and process title, price, rating, and review count
        // Each extraction uses a similar pattern:
        // 1. Select the element using the appropriate selector
        // 2. Extract the text content
        // 3. Process the text (e.g., parse numbers, remove commas)
        // 4. Provide a default value if extraction fails

        let title = product
            .select(&sel.title)
            .next()
            .map(|el| el.text().collect::<String>())
            .unwrap_or_else(|| "N/A".to_string());

        let price = product
            .select(&sel.price)
            .next()
            .and_then(|el| {
                el.text()
                    .collect::<String>()
                    .replace(",", "")
                    .parse::<f64>()
                    .ok()
            })
            .unwrap_or(0.0);

        let rating = product
            .select(&sel.rating)
            .next()
            .and_then(|el| {
                el.text()
                    .collect::<String>()
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse::<f32>().ok())
            })
            .unwrap_or(0.0);

        let review_count = product
            .select(&sel.review_count)
            .next()
            .and_then(|el| {
                el.text()
                    .collect::<String>()
                    .replace(",", "")
                    .parse::<u32>()
                    .ok()
            })
            .unwrap_or(0);

        products.push(Product {
            title,
            price,
            rating,
            review_count,
        });
    }
    println!("Thread {} finished", thread_num);
    products
}

// Function to upload product data to SurrealDB
#[allow(unused_variables)]
async fn upload_surreal(products: &Vec<Product>) -> surrealdb::Result<()> {
    // Connect to the SurrealDB server
    let db = Surreal::new::<Ws>("127.0.0.1:8000").await?;

    // Sign in as a root user
    db.signin(Root {
        username: "root",
        password: "root",
    })
    .await?;

    // Select the namespace and database
    db.use_ns("test").use_db("test").await?;

    // Insert product data into the database
    for product in products {
        let records: Vec<Record> = db
            .create("product")
            .content(Product {
                title: product.title.clone(),
                price: product.price,
                rating: product.rating,
                review_count: product.review_count,
            })
            .await?;
    }

    Ok(())
}

/*
 * Main function: Orchestrates the web scraping, data processing, and storage
 *
 * This function performs the following steps:
 * 1. Initializes the HTTP client and gathers user input
 * 2. Sets up CSS selectors for web scraping
 * 3. Creates and executes concurrent tasks for fetching product data
 * 4. Collects and processes the results
 * 5. Optionally prints the output
 * 6. Uploads data to SurrealDB
 *
 * Error handling is implemented throughout the process.
 */
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Arc::new(Client::builder().gzip(true).build()?);

    // Gather user input for search parameters
    let product_name = read_input_str("Enter the product name: ");
    let total_pages: u32 = read_input_str("Enter the total pages: ").parse()?;
    let print_output = read_input_bool("Do you want to print the output (y/n): ");

    let start = Instant::now();
    let mut all_products = Vec::new();

    // Initialize CSS selectors
    let selectors = Arc::new(Selectors {
        prod: Selector::parse("div[data-component-type='s-search-result']").unwrap(),
        title: Selector::parse("h2 span.a-text-normal").unwrap(),
        price: Selector::parse("span.a-price-whole").unwrap(),
        rating: Selector::parse("span.a-icon-alt").unwrap(),
        review_count: Selector::parse("span.a-size-base").unwrap(),
    });

    // Create tasks for concurrent page fetching
    let mut tasks = Vec::with_capacity(total_pages as usize);
    println!("Fetching data from Amazon.in");
    println!("Creating {} threads for {} pages", total_pages, total_pages);
    for i in 1..=total_pages {
        let url = format!("https://www.amazon.in/s?k={}&page={}", product_name, i);
        let client = Arc::clone(&client);
        let selectors = Arc::clone(&selectors);

        let task =
            tokio::spawn(async move { fetch_page(&i, &url, (*client).clone(), selectors).await });
        tasks.push(task);
    }

    // Execute all tasks concurrently and collect results
    let results = join_all(tasks).await;
    for result in results {
        match result {
            Ok(products) => all_products.extend(products),
            Err(e) => eprintln!("Error in task: {:?}", e),
        }
    }

    // Print output if requested
    if print_output {
        for product in &all_products {
            println!("Title: {}", product.title);
            println!("Price: {:.2}", product.price);
            println!("Rating: {:.1}", product.rating);
            println!("Review Count: {}", product.review_count);
            println!();
        }
    }

    println!("uploading data to database");

    // Upload data to SurrealDB
    let upload_res = upload_surreal(&all_products).await;
    match upload_res {
        Ok(_) => println!("Data uploaded to database"),
        Err(e) => eprintln!("Error uploading data to database: {}", e),
    }

    let duration = start.elapsed();
    println!("Time elapsed: {:?}", duration);
    Ok(())
}
