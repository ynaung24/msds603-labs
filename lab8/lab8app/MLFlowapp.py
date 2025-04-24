from fastapi import FastAPI
import uvicorn
import mlflow
import os
from pydantic import BaseModel

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5001")

app = FastAPI(
    title="MLflow Model Classifier",
    description="FastAPI wrapper for a model loaded from MLflow.",
    version="0.1",
)

# Load model from MLflow (random_forest_wine version 3)
model_uri = os.environ.get("MLFLOW_MODEL_URI", "models:/random_forest_wine/3")
model = None

# Defining path operation for root endpoint
@app.get('/')
def main():
    return {'message': 'This is a model loaded from MLflow'}

class request_body(BaseModel):
    features: list

@app.on_event('startup')
def load_artifacts():
    global model
    try:
        print(f"Loading model from {model_uri}")
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        model = mlflow.pyfunc.load_model(model_uri)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Available registered models:")
        try:
            for rm in mlflow.tracking.MlflowClient().search_registered_models():
                print(f"- {rm.name}")
        except Exception as e2:
            print(f"Could not list models: {e2}")

# Defining path operation for /predict endpoint
@app.post('/predict')
def predict(data: request_body):
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        predictions = model.predict(data.features)
        
        # Convert to list if it's a numpy array
        if hasattr(predictions, 'tolist'):
            predictions = predictions.tolist()
            
        return {'Predictions': predictions}
    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000) 