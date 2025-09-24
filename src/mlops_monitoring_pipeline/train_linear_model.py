# src/mlops_monitoring_pipeline/train_linear_model.py
import pandas as pd
import numpy as np
import os
import yaml
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

def train_and_log_linear_model(data_path, params_path="configs/params.yaml"):
    """
    Performs hyperparameter tuning via GridSearchCV, trains the best model,
    and logs the result to MLflow.
    """
    # Load configuration from params.yaml
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
        train_params = params['train_params']
        model_params = params['linear_model']
    
    # Set the MLflow experiment
    mlflow.set_experiment(train_params['experiment_name'])
    
    # Start a new MLflow run
    with mlflow.start_run():
        print("Starting MLflow run for Linear Regression model...")
        
        # Log all training parameters
        mlflow.log_params(train_params)
        
        # Load data
        try:
            df = pd.read_csv(data_path)
        except FileNotFoundError:
            print(f"Error: Data file not found at {data_path}")
            return
            
        X = df[['x1', 'x2']]
        y = df['y']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=train_params['test_size'], random_state=train_params['random_state']
        )
        
        # Define and run GridSearchCV for hyperparameter tuning
        model = LinearRegression()
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=model_params['grid_search'],
            scoring='neg_mean_squared_error',
            cv=5
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Log the best hyperparameters and score
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_mse", -grid_search.best_score_)
        print(f"Logged best hyperparameters: {grid_search.best_params_}")
        print(f"Logged best cross-validation MSE: {-grid_search.best_score_}")
        
        # Make predictions and evaluate the best model on test set
        y_pred = best_model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_pred)
        
        # Log the final test metric
        mlflow.log_metric("test_mse", test_mse)
        print(f"Logged test MSE: {test_mse}")
        
        # Log and register the best model to MLflow Model Registry
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path=model_params['name'],
            registered_model_name=model_params['name']
        )
        print("Logged and registered best model to MLflow Model Registry.")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_dir = os.path.join(project_root, 'data')
    current_data_path = os.path.join(data_dir, 'current_data.csv')
    
    train_and_log_linear_model(current_data_path)