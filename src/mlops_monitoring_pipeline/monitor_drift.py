# src/mlops_monitoring_pipeline/monitor_drift.py
import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error

def monitor_production_model(model_name):
    """
    Loads the production model from MLflow Model Registry and monitors its performance.
    """
    # Start a new MLflow run for monitoring
    with mlflow.start_run(run_name=f"Monitoring_{model_name}"):
        print(f"Starting monitoring run for model: {model_name}")

        try:
            # Load the latest Production model from MLflow Model Registry
            model_uri = f"models:/{model_name}/Production"
            model = mlflow.sklearn.load_model(model_uri)
            mlflow.log_param("monitored_model", model_name)
            print(f"Successfully loaded production model: {model_name}")

        except Exception as e:
            print(f"Error loading production model '{model_name}': {e}")
            print("Could not perform monitoring. Skipping run.")
            return

        # Load the latest data for monitoring
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        data_dir = os.path.join(project_root, 'data')
        data_path = os.path.join(data_dir, 'current_data.csv')

        try:
            df = pd.read_csv(data_path)
            X = df[['x1', 'x2']]
            y = df['y']
            print(f"Loaded data from {data_path} with {len(df)} samples.")
        except FileNotFoundError:
            print(f"Error: Data file not found at {data_path}. Monitoring aborted.")
            return

        # Predict and calculate performance metric
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)

        # Log the monitoring metric
        mlflow.log_metric("mse_on_latest_data", mse)
        print(f"Logged latest MSE on production model: {mse}")
        
        # TODO: Add logic for data drift detection (e.g., using Evidently AI)
        # and log reports as artifacts.

        print("Monitoring run completed.")

if __name__ == "__main__":
    # You can choose which model to monitor here
    monitor_production_model("linear-regression-model")