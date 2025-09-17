import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

def train_and_log_model(data_path, experiment_name="Model Monitoring Project"):
    """
    Trains a Linear Regression model and logs metrics and artifacts to MLflow.
    
    Args:
        data_path (str): The file path to the training data.
        experiment_name (str): The name of the MLflow experiment.
    """
    # Set the MLflow experiment name
    mlflow.set_experiment(experiment_name)
    
    # Start an MLflow run to log all steps
    with mlflow.start_run():
        print(f"Starting MLflow run for experiment: {experiment_name}")
        
        # Log the data path as a parameter
        mlflow.log_param("data_path", data_path)
        
        # Load data
        try:
            df = pd.read_csv(data_path)
        except FileNotFoundError:
            print(f"Error: Data file not found at {data_path}")
            return
            
        # Prepare features and target
        X = df[['x1', 'x2']]
        y = df['y']
        
        # Split data for training and evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions and evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        # Log metrics and model
        mlflow.log_metric("mse", mse)
        print(f"Logged MSE: {mse}")
        
        mlflow.sklearn.log_model(model, "linear-regression-model")
        print("Logged model as artifact.")
        
        # Get the run ID for later use
        run_id = mlflow.active_run().info.run_id
        print(f"MLflow run completed. Run ID: {run_id}")
        
        return run_id

if __name__ == "__main__":
    # Get the data directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Corrected path to navigate from the script location to the project root
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_dir = os.path.join(project_root, 'data')
    
    original_data_path = os.path.join(data_dir, 'original_data.csv')
    
    # Run the training process for the original model
    train_and_log_model(original_data_path)