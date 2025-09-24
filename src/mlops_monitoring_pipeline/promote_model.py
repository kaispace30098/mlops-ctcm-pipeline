import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Replace 'linear-regression-model' with your model name
# Replace '2' with the version number you want to promote
client.transition_model_version_stage(
    name="linear-regression-model",
    version=2,
    stage="Production"
)

print("Successfully transitioned linear-regression-model version 2 to Production stage.")