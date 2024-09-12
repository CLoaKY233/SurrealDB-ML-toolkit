import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from surrealdb import Surreal

async def main():
    """
    Main function to demonstrate integration of SurrealDB with PyTorch for a linear regression task.
    This function handles database operations, data preparation, model training, and prediction.
    """

    # Connect to SurrealDB
    async with Surreal("ws://localhost:8000/rpc") as db:
        # Authenticate and select database
        await db.signin({"user": "root", "pass": "root"})
        await db.use("test", "test")

        # Sample data for houses
        # Each dictionary represents a house with its size (in sq ft), number of bedrooms, and price
        houses = [
            {"size": 1000, "bedrooms": 2, "price": 200000},
            {"size": 1500, "bedrooms": 3, "price": 300000},
            {"size": 2000, "bedrooms": 4, "price": 400000},
            {"size": 2500, "bedrooms": 5, "price": 500000},
            {"size": 3000, "bedrooms": 6, "price": 600000},
        ]

        # Insert sample data into SurrealDB
        # Each house is created as a new record in the 'house' table
        for house in houses:
            await db.create("house", house)

        # Retrieve all house data from SurrealDB and convert to a pandas DataFrame
        result = await db.select("house")
        df = pd.DataFrame(result)

        # Prepare features (X) and target (y) for the model
        # X contains 'size' and 'bedrooms', y contains 'price'
        X = df[['size', 'bedrooms']].values
        y = df['price'].values

        # Split data into training and testing sets
        # 80% of the data is used for training, 20% for testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the features using StandardScaler
        # This ensures all features are on the same scale, which is important for many ML algorithms
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Convert data to PyTorch tensors
        # PyTorch models operate on tensors, so we need to convert our numpy arrays
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

        # Define the Linear Regression model
        # This model subclasses nn.Module, which is the base class for all neural network modules in PyTorch
        class LinearRegression(nn.Module):
            def __init__(self, input_dim):
                super(LinearRegression, self).__init__()
                self.linear = nn.Linear(input_dim, 1)  # Single output for regression

            def forward(self, x):
                return self.linear(x)

        # Initialize the model
        # We use 2 as the input dimension because we have two features: size and bedrooms
        model = LinearRegression(input_dim=2)

        # Define loss function and optimizer
        # MSELoss is Mean Squared Error, suitable for regression problems
        # SGD is Stochastic Gradient Descent, a common optimization algorithm
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Train the model
        num_epochs = 1000
        for epoch in range(num_epochs):
            # Forward pass: Compute predicted y by passing x to the model
            outputs = model(X_train_tensor)

            # Compute loss
            loss = criterion(outputs, y_train_tensor)

            # Backward pass and optimize
            optimizer.zero_grad()  # Zero the parameter gradients
            loss.backward()        # Compute gradients
            optimizer.step()       # Update parameters

            # Print loss every 100 epochs to track progress
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Evaluate the model
        # We use torch.no_grad() to disable gradient calculation for inference
        with torch.no_grad():
            predicted = model(X_test_tensor)
            mse = criterion(predicted, y_test_tensor)
            print(f"Mean Squared Error on test set: {mse.item():.4f}")

        # Make a prediction for a new house
        # We first scale the new house data using the same scaler as before
        new_house = torch.FloatTensor(scaler.transform([[2200, 4]]))
        predicted_price = model(new_house).item()
        print(f"Predicted price for a house with 2200 sqft and 4 bedrooms: ${predicted_price:.2f}")

        # Close the database connection
        await db.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
