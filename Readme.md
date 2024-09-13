# SurrealDB Integration Examples

## Overview

SurrealDB is a multi-model database that combines the capabilities of a document, graph, and relational database. This repository demonstrates how to integrate SurrealDB with Python and Rust for various applications, including data import, web scraping, and machine learning.

## New Features of SurrealDB

- Multi-model support (document, graph, and relational)
- Real-time capabilities
- Built-in authentication and authorization
- Support for complex queries and graph traversals
- Scalable from edge devices to cloud deployments

## Contents

1. [Python Integration](#python-integration)
2. [Rust Integration](#rust-integration)
3. [Getting Started](#getting-started)
4. [Running the Examples](#running-the-examples)
5. [Contributing](#contributing)

---

## Python Integration

### CSV to SurrealDB Import
- **Location**: `Database/Surreal_with_Python/`
- **Description**: Demonstrates how to import data from a CSV file into SurrealDB using Python.
- **Key Features**: CSV reading, data insertion, error handling
- **Info File**: [Surreal_with_Python/info.md](Surreal_with_Python/info.md)


### Web Scraping with SurrealDB
- **Location**: `Database/Surreal_with_ML/Webscrape_Surreal/`
- **Description**: Scrapes product data from Amazon and stores it in SurrealDB.
- **Key Features**: Web scraping, asynchronous database operations, data structuring
- **Info File**: [Surreal_with_ML/Webscrape_Surreal/info.md](Surreal_with_ML/Webscrape_Surreal/info.md)

### PyTorch and SurrealDB for Linear Regression
- **Location**: `Database/Surreal_with_ML/PyTorch_Surreal_LR/`
- **Description**: Integrates SurrealDB with PyTorch for a linear regression task.
- **Key Features**: Data preprocessing, PyTorch model creation, SurrealDB data storage and retrieval
- **Info File**: [Surreal_with_ML/PyTorch_Surreal_LR/info.md](Surreal_with_ML/PyTorch_Surreal_LR/info.md)

---

## Rust Integration

### Amazon Product Scraper with SurrealDB
- **Location**: `Database/Surreal_with_Rust/`
- **Description**: A Rust program that scrapes Amazon product data and stores it in SurrealDB.
- **Key Features**: Concurrent web scraping, asynchronous programming, SurrealDB integration in Rust
- **Info File**: [Surreal_with_Rust/info.md](Surreal_with_Rust/info.md)

Each folder contains the corresponding code and an `info.md` file with detailed explanations and learning prompts.

---

## Getting Started

1. Clone this repository
2. Install SurrealDB following the [official documentation](https://surrealdb.com/docs/surrealdb/installation)
3. Navigate to the desired example directory
4. Follow the setup instructions below for your chosen integration

### Python Setup
For all Python examples:
1. Ensure you have Python 3.7+ installed
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install required packages:
   ```
   pip install surrealdb pandas requests beautifulsoup4 torch scikit-learn
   ```

#### CSV to SurrealDB Import
No additional setup required.

#### Web Scraping with SurrealDB
No additional setup required.

#### PyTorch and SurrealDB for Linear Regression
No additional setup required.

### Rust Setup
For the Rust example:
1. Ensure you have Rust and Cargo installed. If not, install from [rustup.rs](https://rustup.rs/)
2. Navigate to the `Database/Surreal_with_Rust/` directory
3. Build the project:
   ```
   cargo build
   ```
4. Run the project:
   ```
   cargo run
   ```

---

## Running the Examples

### Python Examples
1. Navigate to the respective directory
2. Run the Python script:
   ```
   python script_name.py
   ```

### Rust Example
1. Navigate to the `Database/Surreal_with_Rust/` directory
2. Run the project:
   ```
   cargo run
   ```

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
