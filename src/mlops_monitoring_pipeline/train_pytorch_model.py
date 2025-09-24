# src/mlops_monitoring_pipeline/train_pytorch_model.py
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.pytorch

# A simple Feed-Forward Neural Network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def train_and_log_pytorch_model(data_path, params_path="configs/params.yaml"):
    """
    Performs hyperparameter tuning for a PyTorch model and logs the best model.
    """
    # Load configuration
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
        train_params = params['train_params']
        model_params = params['pytorch_model']
        
    mlflow.set_experiment(train_params['experiment_name'])
    
    # Start the parent MLflow run for hyperparameter tuning
    with mlflow.start_run(run_name="pytorch_tuning"):
        print("Starting parent run for PyTorch model tuning...")
        
        # Log parent run parameters
        mlflow.log_params(train_params)
        
        # Load data
        df = pd.read_csv(data_path)
        X = df[['x1', 'x2']].values
        y = df['y'].values
        
        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=train_params['test_size'], random_state=train_params['random_state']
        )
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        best_test_mse = float('inf')
        best_run_id = None

        # Loop through all hyperparameter combinations
        for lr in model_params['hyperparameters']['learning_rate']:
            for epochs in model_params['hyperparameters']['epochs']:
                # Start a nested run for each combination
                with mlflow.start_run(nested=True, run_name=f"lr_{lr}_epochs_{epochs}"):
                    model = SimpleNN()
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(), lr=lr)

                    # Log hyperparameters for this specific run
                    mlflow.log_param("epochs", epochs)
                    mlflow.log_param("learning_rate", lr)

                    # Train the model
                    for epoch in range(epochs):
                        optimizer.zero_grad()
                        outputs = model(X_train_tensor)
                        loss = criterion(outputs, y_train_tensor)
                        loss.backward()
                        optimizer.step()
                    
                    # Evaluate on test set
                    model.eval()
                    with torch.no_grad():
                        test_outputs = model(X_test_tensor)
                        test_loss = criterion(test_outputs, y_test_tensor).item()
                        
                    mlflow.log_metric("test_mse", test_loss)
                    print(f" - Run with lr={lr}, epochs={epochs}: test_mse={test_loss}")

                    # Check if this is the best model so far
                    if test_loss < best_test_mse:
                        best_test_mse = test_loss
                        best_run_id = mlflow.active_run().info.run_id

        print(f"\nBest run found with MSE: {best_test_mse}")

        # Log and register the best model from the nested runs
        if best_run_id:
            # Need to get the full URI to the best model
            best_model_uri = f"runs:/{best_run_id}/pytorch-model"
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=model_params['name'],
                registered_model_name=model_params['name']
            )
            print("Logged and registered best PyTorch model.")
            
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_dir = os.path.join(project_root, 'data')
    current_data_path = os.path.join(data_dir, 'current_data.csv')
    
    train_and_log_pytorch_model(current_data_path)