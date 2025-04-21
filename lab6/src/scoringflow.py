from metaflow import FlowSpec, step, Parameter
import pandas as pd
import mlflow
import numpy as np
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import scale_data

class WineScoringFlow(FlowSpec):
    """
    A Metaflow flow for making predictions with a trained wine classifier model.
    
    This flow:
    1. Loads a registered model from MLFlow
    2. Processes input data
    3. Makes predictions
    4. Displays the results
    """
    
    # Parameters for the flow
    model_name = Parameter('model_name', default='random_forest_wine', type=str)
    model_version = Parameter('model_version', default='latest', type=str)
    input_data_path = Parameter('input_data_path', default='../data/wine/test_data.parquet', type=str)
    
    @step
    def start(self):
        """
        Start step: Load the model and input data
        """
        print(f"Starting the Wine Scoring Flow for model: {self.model_name}")
        
        # Set up MLFlow with the running server
        mlflow.set_tracking_uri("http://localhost:5001")
        
        # Load the model from MLFlow
        if self.model_version == 'latest':
            self.model = mlflow.pyfunc.load_model(f"models:/{self.model_name}/latest")
        else:
            self.model = mlflow.pyfunc.load_model(f"models:/{self.model_name}/{self.model_version}")
        
        print(f"Model loaded successfully: {self.model_name}")
        
        # Move to the next step
        self.next(self.process_input)
    
    @step
    def process_input(self):
        """
        Process input data step: Load and preprocess the input data
        """
        print("Processing input data")
        
        # Load the input data based on file extension
        file_ext = os.path.splitext(self.input_data_path)[1].lower()
        if file_ext == '.csv':
            # Load CSV data
            self.input_data = pd.read_csv(self.input_data_path, header=None)
            # Add column names for wine dataset
            column_names = ['class', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash',
                           'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
                           'proanthocyanins', 'color_intensity', 'hue', 
                           'od280/od315_of_diluted_wines', 'proline']
            self.input_data.columns = column_names
        elif file_ext == '.parquet':
            # Load parquet data
            self.input_data = pd.read_parquet(self.input_data_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Use .csv or .parquet")
        
        print(f"Input data loaded with shape: {self.input_data.shape}")
        print(f"Columns in input data: {self.input_data.columns.tolist()}")
        
        # Check for NaN values in the entire dataset
        print("\nChecking for NaN values in the dataset:")
        nan_counts = self.input_data.isna().sum()
        print("NaN counts by column:")
        for col, count in nan_counts.items():
            if count > 0:
                print(f"  {col}: {count} NaN values ({count/len(self.input_data)*100:.2f}%)")
        
        # Separate features and target (if present)
        # First try 'class' column (wine dataset standard)
        if 'class' in self.input_data.columns:
            self.X = self.input_data.drop('class', axis=1)
            self.y = self.input_data['class']
            self.has_target = True
            print("\nUsing 'class' column as target")
            print(f"Target value counts:\n{self.y.value_counts()}")
            print(f"NaN values in target: {self.y.isna().sum()} ({self.y.isna().sum()/len(self.y)*100:.2f}%)")
        # Then try 'target' column (alternative)
        elif 'target' in self.input_data.columns:
            self.X = self.input_data.drop('target', axis=1)
            self.y = self.input_data['target']
            self.has_target = True
            print("\nUsing 'target' column as target")
            print(f"Target value counts:\n{self.y.value_counts()}")
            print(f"NaN values in target: {self.y.isna().sum()} ({self.y.isna().sum()/len(self.y)*100:.2f}%)")
        else:
            self.X = self.input_data
            self.has_target = False
            print("\nNo target column found, running in prediction-only mode")
        
        # Check for NaN values in features
        print("\nChecking for NaN values in features:")
        feature_nan_counts = self.X.isna().sum()
        for col, count in feature_nan_counts.items():
            if count > 0:
                print(f"  {col}: {count} NaN values ({count/len(self.X)*100:.2f}%)")
        
        if self.X.isna().any().any():
            print("\nWarning: NaN values found in features. Filling with mean values.")
            self.X = self.X.fillna(self.X.mean())
        
        # Check for NaN values in target if present
        if self.has_target and self.y.isna().any():
            print("\nWarning: NaN values found in target. Removing rows with NaN targets.")
            valid_indices = ~self.y.isna()
            self.X = self.X[valid_indices]
            self.y = self.y[valid_indices]
            if len(self.y) == 0:
                print("Error: All target values are NaN. Cannot proceed.")
                raise ValueError("All target values are NaN")
        
        # Scale the features
        self.X_scaled = scale_data(self.X)
        
        print("\nInput data processed successfully")
        
        # Move to the next step
        self.next(self.make_predictions)
    
    @step
    def make_predictions(self):
        """
        Make predictions step: Generate predictions using the loaded model
        """
        print("Making predictions")
        
        # Make predictions
        self.predictions = self.model.predict(self.X_scaled)
        
        # Create a DataFrame with predictions
        self.results = pd.DataFrame({
            'predicted_class': self.predictions
        })
        
        # Calculate metrics if target is available and not NaN
        if self.has_target and not self.y.isna().all():
            # Filter out NaN values for metric calculation
            valid_indices = ~self.y.isna()
            valid_y = self.y[valid_indices]
            valid_pred = self.predictions[valid_indices]
            
            if len(valid_y) > 0:
                self.metrics = {
                    'accuracy': accuracy_score(valid_y, valid_pred),
                    'precision': precision_score(valid_y, valid_pred, average='weighted'),
                    'recall': recall_score(valid_y, valid_pred, average='weighted'),
                    'f1': f1_score(valid_y, valid_pred, average='weighted')
                }
                
                print("Prediction metrics:")
                for metric, value in self.metrics.items():
                    print(f"{metric}: {value:.4f}")
                
                # Add actual class to results only for valid values
                self.results['actual_class'] = pd.Series(self.y, index=self.results.index)
            else:
                print("No valid target values available for metric calculation")
        else:
            print("Running in prediction-only mode (no valid target values)")
        
        print("Predictions made successfully")
        
        # Move to the next step
        self.next(self.end)
    
    @step
    def end(self):
        """
        End step: Finalize the flow
        """
        print("Wine Scoring Flow completed successfully")
        print("\nPrediction Results:")
        print(self.results.head())

if __name__ == '__main__':
    WineScoringFlow() 