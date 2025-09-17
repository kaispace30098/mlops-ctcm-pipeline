import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import os

def train_and_log_model(data_path, experiment_name):
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
        
        # Load data
        try:
            df = pd.read_csv(data_path)
        except FileNotFoundError:
            print(f"Error: Data file not found at {data_path}")
            return
            
        # Log data path as a parameter
        mlflow.log_param("data_path", data_path)
        
        # Prepare features and target
        # For this example, we will only use x1 and x2 for training
        if 'x2' in df.columns:
            X = df[['x1', 'x2']]
        else:
            X = df[['x1']]
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
        
        print("MLflow run completed.")

if __name__ == "__main__":
    # Define the path to the original dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')
    original_data_path = os.path.join(data_dir, 'original_data.csv')
    
    # Run the training process for the original model
    train_and_log_model(original_data_path, "Model Monitoring Project")
